#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

int copy_one_range(const std::uint8_t* src,
                   std::uint8_t* dst,
                   const int64_t* src_ids,
                   int64_t block_count,
                   int64_t block_bytes,
                   int64_t begin,
                   int64_t end)
{
    if (src == nullptr || dst == nullptr || src_ids == nullptr) {
        return -1;
    }
    if (block_count < 0 || block_bytes <= 0) {
        return -2;
    }
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
        const int64_t src_idx = src_ids[out_idx];
        if (src_idx < 0 || src_idx >= block_count * 1024LL * 1024LL) {
            return -3;
        }
        std::memcpy(dst + out_idx * block_bytes,
                    src + src_idx * block_bytes,
                    static_cast<size_t>(block_bytes));
    }
    return 0;
}

int copy_two_range(const std::uint8_t* src0,
                   const std::uint8_t* src1,
                   std::uint8_t* dst0,
                   std::uint8_t* dst1,
                   const int64_t* src_ids,
                   int64_t block_count,
                   int64_t block_bytes,
                   int64_t begin,
                   int64_t end)
{
    if (src0 == nullptr || src1 == nullptr || dst0 == nullptr || dst1 == nullptr
        || src_ids == nullptr) {
        return -1;
    }
    if (block_count < 0 || block_bytes <= 0) {
        return -2;
    }
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
        const int64_t src_idx = src_ids[out_idx];
        if (src_idx < 0 || src_idx >= block_count * 1024LL * 1024LL) {
            return -3;
        }
        std::memcpy(dst0 + out_idx * block_bytes,
                    src0 + src_idx * block_bytes,
                    static_cast<size_t>(block_bytes));
        std::memcpy(dst1 + out_idx * block_bytes,
                    src1 + src_idx * block_bytes,
                    static_cast<size_t>(block_bytes));
    }
    return 0;
}

int memcpy_bytes_range(const std::uint8_t* src,
                       std::uint8_t* dst,
                       int64_t total_bytes,
                       int64_t begin,
                       int64_t end)
{
    if (src == nullptr || dst == nullptr) {
        return -1;
    }
    if (total_bytes < 0 || begin < 0 || end < begin || end > total_bytes) {
        return -2;
    }
    const int64_t bytes = end - begin;
    if (bytes == 0) {
        return 0;
    }
    std::memcpy(dst + begin, src + begin, static_cast<size_t>(bytes));
    return 0;
}

template <typename CopyFn>
int run_per_call_threads(int64_t block_count, int32_t threads, CopyFn copy_fn)
{
    if (block_count == 0) {
        return 0;
    }
    const int32_t worker_count = static_cast<int32_t>(
        std::min<int64_t>(static_cast<int64_t>(std::max(1, threads)), block_count));
    std::atomic<int> status{0};

    auto copy_range = [&](int64_t begin, int64_t end) {
        const int ret = copy_fn(begin, end);
        if (ret != 0) {
            int expected = 0;
            status.compare_exchange_strong(expected, ret, std::memory_order_relaxed);
        }
    };

    if (worker_count == 1) {
        copy_range(0, block_count);
        return status.load(std::memory_order_relaxed);
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int32_t worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = block_count * worker / worker_count;
        const int64_t end = block_count * (worker + 1) / worker_count;
        workers.emplace_back(copy_range, begin, end);
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return status.load(std::memory_order_relaxed);
}

class PersistentPacker {
public:
    explicit PersistentPacker(int32_t threads)
        : worker_count_(std::max(1, threads))
    {
        workers_.reserve(static_cast<size_t>(worker_count_));
        for (int32_t worker = 0; worker < worker_count_; ++worker) {
            workers_.emplace_back([this, worker]() { worker_loop(worker); });
        }
    }

    ~PersistentPacker()
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    int pack_one(const void* src_base,
                 void* dst_base,
                 const int64_t* src_ids,
                 int64_t block_count,
                 int64_t block_bytes)
    {
        return submit(1, src_base, nullptr, dst_base, nullptr, src_ids, block_count, block_bytes);
    }

    int pack_two(const void* src0,
                 const void* src1,
                 void* dst0,
                 void* dst1,
                 const int64_t* src_ids,
                 int64_t block_count,
                 int64_t block_bytes)
    {
        return submit(2, src0, src1, dst0, dst1, src_ids, block_count, block_bytes);
    }

    int memcpy_bytes(const void* src_base, void* dst_base, int64_t total_bytes)
    {
        return submit_memcpy(src_base, dst_base, total_bytes);
    }

private:
    struct Job {
        int parts = 0;
        const std::uint8_t* src0 = nullptr;
        const std::uint8_t* src1 = nullptr;
        std::uint8_t* dst0 = nullptr;
        std::uint8_t* dst1 = nullptr;
        const int64_t* src_ids = nullptr;
        int64_t block_count = 0;
        int64_t block_bytes = 0;
        int64_t total_bytes = 0;
    };

    int submit(int parts,
               const void* src0,
               const void* src1,
               void* dst0,
               void* dst1,
               const int64_t* src_ids,
               int64_t block_count,
               int64_t block_bytes)
    {
        if (src0 == nullptr || dst0 == nullptr || src_ids == nullptr) {
            return -1;
        }
        if (parts == 2 && (src1 == nullptr || dst1 == nullptr)) {
            return -1;
        }
        if (block_count < 0 || block_bytes <= 0) {
            return -2;
        }
        if (block_count == 0) {
            return 0;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (remaining_ != 0) {
                return -10;
            }
            job_ = Job{
                parts,
                static_cast<const std::uint8_t*>(src0),
                static_cast<const std::uint8_t*>(src1),
                static_cast<std::uint8_t*>(dst0),
                static_cast<std::uint8_t*>(dst1),
                src_ids,
                block_count,
                block_bytes,
                0,
            };
            status_ = 0;
            remaining_ = worker_count_;
            ++generation_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this]() { return remaining_ == 0; });
        return status_;
    }

    int submit_memcpy(const void* src_base, void* dst_base, int64_t total_bytes)
    {
        if (src_base == nullptr || dst_base == nullptr) {
            return -1;
        }
        if (total_bytes < 0) {
            return -2;
        }
        if (total_bytes == 0) {
            return 0;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (remaining_ != 0) {
                return -10;
            }
            job_ = Job{
                3,
                static_cast<const std::uint8_t*>(src_base),
                nullptr,
                static_cast<std::uint8_t*>(dst_base),
                nullptr,
                nullptr,
                0,
                0,
                total_bytes,
            };
            status_ = 0;
            remaining_ = worker_count_;
            ++generation_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this]() { return remaining_ == 0; });
        return status_;
    }

    void worker_loop(int32_t worker)
    {
        std::uint64_t seen_generation = 0;
        while (true) {
            Job local_job;
            std::uint64_t local_generation = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this, seen_generation]() {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                local_job = job_;
                local_generation = generation_;
            }

            const int64_t begin = local_job.block_count * worker / worker_count_;
            const int64_t end = local_job.block_count * (worker + 1) / worker_count_;
            int ret = 0;
            if (local_job.parts == 3) {
                const int64_t begin = local_job.total_bytes * worker / worker_count_;
                const int64_t end = local_job.total_bytes * (worker + 1) / worker_count_;
                ret = memcpy_bytes_range(local_job.src0,
                                         local_job.dst0,
                                         local_job.total_bytes,
                                         begin,
                                         end);
            } else if (local_job.parts == 1) {
                ret = copy_one_range(local_job.src0,
                                     local_job.dst0,
                                     local_job.src_ids,
                                     local_job.block_count,
                                     local_job.block_bytes,
                                     begin,
                                     end);
            } else {
                ret = copy_two_range(local_job.src0,
                                     local_job.src1,
                                     local_job.dst0,
                                     local_job.dst1,
                                     local_job.src_ids,
                                     local_job.block_count,
                                     local_job.block_bytes,
                                     begin,
                                     end);
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                if (ret != 0 && status_ == 0) {
                    status_ = ret;
                }
                seen_generation = local_generation;
                --remaining_;
                if (remaining_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    const int32_t worker_count_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;
    std::uint64_t generation_ = 0;
    int32_t remaining_ = 0;
    int status_ = 0;
    Job job_;
};

}  // namespace

extern "C" int kv_staging_pack_blocks(const void* src_base,
                                       void* dst_base,
                                       const int64_t* src_ids,
                                       int64_t block_count,
                                       int64_t block_bytes,
                                       int32_t threads)
{
    if (src_base == nullptr || dst_base == nullptr || src_ids == nullptr) {
        return -1;
    }
    if (block_count < 0 || block_bytes <= 0 || threads <= 0) {
        return -2;
    }
    if (block_count == 0) {
        return 0;
    }

    const auto* src = static_cast<const std::uint8_t*>(src_base);
    auto* dst = static_cast<std::uint8_t*>(dst_base);
    return run_per_call_threads(block_count, threads, [&](int64_t begin, int64_t end) {
        return copy_one_range(src, dst, src_ids, block_count, block_bytes, begin, end);
    });
}

extern "C" int kv_staging_pack_blocks2(const void* src0_base,
                                        const void* src1_base,
                                        void* dst0_base,
                                        void* dst1_base,
                                        const int64_t* src_ids,
                                        int64_t block_count,
                                        int64_t block_bytes,
                                        int32_t threads)
{
    if (src0_base == nullptr || src1_base == nullptr || dst0_base == nullptr
        || dst1_base == nullptr || src_ids == nullptr) {
        return -1;
    }
    if (block_count < 0 || block_bytes <= 0 || threads <= 0) {
        return -2;
    }
    if (block_count == 0) {
        return 0;
    }

    const auto* src0 = static_cast<const std::uint8_t*>(src0_base);
    const auto* src1 = static_cast<const std::uint8_t*>(src1_base);
    auto* dst0 = static_cast<std::uint8_t*>(dst0_base);
    auto* dst1 = static_cast<std::uint8_t*>(dst1_base);
    return run_per_call_threads(block_count, threads, [&](int64_t begin, int64_t end) {
        return copy_two_range(src0, src1, dst0, dst1, src_ids, block_count, block_bytes, begin, end);
    });
}

extern "C" int kv_staging_memcpy_bytes(const void* src_base,
                                        void* dst_base,
                                        int64_t total_bytes,
                                        int32_t threads)
{
    if (src_base == nullptr || dst_base == nullptr) {
        return -1;
    }
    if (total_bytes < 0 || threads <= 0) {
        return -2;
    }
    if (total_bytes == 0) {
        return 0;
    }

    const auto* src = static_cast<const std::uint8_t*>(src_base);
    auto* dst = static_cast<std::uint8_t*>(dst_base);
    const int32_t worker_count = static_cast<int32_t>(
        std::min<int64_t>(static_cast<int64_t>(std::max(1, threads)), total_bytes));
    std::atomic<int> status{0};
    auto copy_range = [&](int32_t worker) {
        const int64_t begin = total_bytes * worker / worker_count;
        const int64_t end = total_bytes * (worker + 1) / worker_count;
        const int ret = memcpy_bytes_range(src, dst, total_bytes, begin, end);
        if (ret != 0) {
            int expected = 0;
            status.compare_exchange_strong(expected, ret, std::memory_order_relaxed);
        }
    };
    if (worker_count == 1) {
        copy_range(0);
        return status.load(std::memory_order_relaxed);
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(copy_range, worker);
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return status.load(std::memory_order_relaxed);
}

extern "C" void* kv_staging_packer_create(int32_t threads)
{
    try {
        return new PersistentPacker(std::max(1, threads));
    } catch (...) {
        return nullptr;
    }
}

extern "C" void kv_staging_packer_destroy(void* handle)
{
    delete static_cast<PersistentPacker*>(handle);
}

extern "C" int kv_staging_packer_pack_blocks(void* handle,
                                             const void* src_base,
                                             void* dst_base,
                                             const int64_t* src_ids,
                                             int64_t block_count,
                                             int64_t block_bytes)
{
    if (handle == nullptr) {
        return -1;
    }
    return static_cast<PersistentPacker*>(handle)->pack_one(
        src_base, dst_base, src_ids, block_count, block_bytes);
}

extern "C" int kv_staging_packer_pack_blocks2(void* handle,
                                              const void* src0_base,
                                              const void* src1_base,
                                              void* dst0_base,
                                              void* dst1_base,
                                              const int64_t* src_ids,
                                              int64_t block_count,
                                              int64_t block_bytes)
{
    if (handle == nullptr) {
        return -1;
    }
    return static_cast<PersistentPacker*>(handle)->pack_two(
        src0_base, src1_base, dst0_base, dst1_base, src_ids, block_count, block_bytes);
}

extern "C" int kv_staging_packer_memcpy_bytes(void* handle,
                                              const void* src_base,
                                              void* dst_base,
                                              int64_t total_bytes)
{
    if (handle == nullptr) {
        return -1;
    }
    return static_cast<PersistentPacker*>(handle)->memcpy_bytes(src_base, dst_base, total_bytes);
}
