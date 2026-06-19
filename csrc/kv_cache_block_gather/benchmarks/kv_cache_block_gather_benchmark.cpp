#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "acl/acl.h"
#if __has_include("aclnnop/aclnn_kv_cache_block_gather.h")
#include "aclnnop/aclnn_kv_cache_block_gather.h"
#else
#include "aclnn_kv_cache_block_gather.h"
#endif

namespace {

#define CHECK_ACL(expr)                                                        \
    do {                                                                       \
        aclError _ret = (expr);                                                \
        if (_ret != ACL_SUCCESS) {                                             \
            std::cerr << #expr << " failed: " << _ret << std::endl;            \
            return _ret;                                                       \
        }                                                                      \
    } while (0)

struct Options {
    int32_t device_id = 0;
    int32_t num_pages = 1024;
    int32_t selected_blocks = 1024;
    int32_t dst_blocks = 1024;
    size_t elems_per_block = 16384;
    int32_t warmup = 3;
    int32_t iters = 10;
    std::string src_pattern = "random";
    std::string dst_pattern = "random";
};

struct TensorHandle {
    aclTensor *tensor = nullptr;
    void *device_addr = nullptr;
    bool device_allocated = false;
};

struct PagesHandle {
    void *host_addr = nullptr;
    void *device_addr = nullptr;
    bool host_registered = false;
    bool device_allocated = false;
};

int64_t ShapeSize(const std::vector<int64_t> &shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

void *AllocPageAligned(size_t size) {
    const long page = sysconf(_SC_PAGESIZE);
    void *ptr = nullptr;
    if (posix_memalign(&ptr, static_cast<size_t>(page), size) != 0) {
        return nullptr;
    }
    std::memset(ptr, 0, size);
    return ptr;
}

void DestroyTensorHandle(TensorHandle &handle) {
    if (handle.tensor != nullptr) {
        aclDestroyTensor(handle.tensor);
        handle.tensor = nullptr;
    }
    if (handle.device_allocated && handle.device_addr != nullptr) {
        aclrtFree(handle.device_addr);
    }
    handle.device_addr = nullptr;
    handle.device_allocated = false;
}

void DestroyPagesHandle(PagesHandle &pages) {
    if (pages.host_registered && pages.host_addr != nullptr) {
        aclrtHostUnregister(pages.host_addr);
    }
    if (pages.device_allocated && pages.device_addr != nullptr) {
        aclrtFree(pages.device_addr);
    }
    if (pages.host_addr != nullptr) {
        free(pages.host_addr);
    }
    pages = {};
}

void FillPages(std::vector<float> &host_data, int32_t num_pages,
               size_t elems_per_block) {
    host_data.resize(static_cast<size_t>(num_pages) * elems_per_block);
    for (int32_t page = 0; page < num_pages; ++page) {
        for (size_t i = 0; i < elems_per_block; ++i) {
            host_data[static_cast<size_t>(page) * elems_per_block + i] =
                static_cast<float>(page * 100000 + (i % 100000));
        }
    }
}

int32_t PickSourcePage(int32_t block, int32_t num_pages,
                       const std::string &pattern) {
    if (pattern == "sequential") {
        return block % num_pages;
    }
    if (pattern == "reverse") {
        return num_pages - 1 - (block % num_pages);
    }
    if (pattern == "stride") {
        return (block * 13 + 7) % num_pages;
    }
    uint32_t x = static_cast<uint32_t>(block) * 2654435761U + 1013904223U;
    x ^= x >> 16;
    return static_cast<int32_t>(x % static_cast<uint32_t>(num_pages));
}

std::vector<int32_t> BuildSourceBlockIds(const Options &opt) {
    std::vector<int32_t> ids(static_cast<size_t>(opt.selected_blocks));
    for (int32_t block = 0; block < opt.selected_blocks; ++block) {
        ids[static_cast<size_t>(block)] =
            PickSourcePage(block, opt.num_pages, opt.src_pattern);
    }
    return ids;
}

std::vector<int32_t> BuildDestinationBlockIds(const Options &opt) {
    std::vector<int32_t> ids(static_cast<size_t>(opt.dst_blocks));
    std::iota(ids.begin(), ids.end(), 0);

    if (opt.dst_pattern == "reverse") {
        std::reverse(ids.begin(), ids.end());
    } else if (opt.dst_pattern == "stride") {
        std::vector<int32_t> perm;
        perm.reserve(ids.size());
        std::vector<uint8_t> seen(ids.size(), 0);
        int32_t value = 7 % opt.dst_blocks;
        for (int32_t i = 0; i < opt.dst_blocks; ++i) {
            while (seen[static_cast<size_t>(value)] != 0) {
                value = (value + 1) % opt.dst_blocks;
            }
            perm.push_back(value);
            seen[static_cast<size_t>(value)] = 1;
            value = (value + 13) % opt.dst_blocks;
        }
        ids = std::move(perm);
    } else if (opt.dst_pattern == "random") {
        std::mt19937 rng(20260602U);
        std::shuffle(ids.begin(), ids.end(), rng);
    }

    ids.resize(static_cast<size_t>(opt.selected_blocks));
    return ids;
}

int CreateDeviceTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                       size_t element_size, TensorHandle &handle) {
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * element_size;
    CHECK_ACL(aclrtMalloc(&handle.device_addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    handle.device_allocated = true;
    CHECK_ACL(aclrtMemset(handle.device_addr, bytes, 0, bytes));

    const std::vector<int64_t> strides = ContiguousStrides(shape);
    handle.tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                                    strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(),
                                    handle.device_addr);
    if (handle.tensor == nullptr) {
        std::cerr << "aclCreateTensor(device) failed" << std::endl;
        return 1;
    }
    return 0;
}

aclTensor *CreateTensorView(const std::vector<int64_t> &shape,
                            aclDataType dtype, void *addr) {
    const std::vector<int64_t> strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), addr);
}

int CreateMappedHostPages(const std::vector<float> &host_data,
                          PagesHandle &pages) {
    const size_t bytes = host_data.size() * sizeof(float);
    pages.host_addr = AllocPageAligned(bytes);
    if (pages.host_addr == nullptr) {
        std::cerr << "host aligned allocation failed" << std::endl;
        return 1;
    }
    std::memcpy(pages.host_addr, host_data.data(), bytes);
    CHECK_ACL(aclrtHostRegister(pages.host_addr, bytes,
                                ACL_HOST_REGISTER_MAPPED,
                                &pages.device_addr));
    pages.host_registered = true;
    return 0;
}

int CreateDevicePages(const std::vector<float> &host_data, PagesHandle &pages) {
    const size_t bytes = host_data.size() * sizeof(float);
    CHECK_ACL(aclrtMalloc(&pages.device_addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    pages.device_allocated = true;
    CHECK_ACL(aclrtMemcpy(pages.device_addr, bytes, host_data.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    return 0;
}

double Mean(const std::vector<float> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

float Percentile(std::vector<float> values, double p) {
    std::sort(values.begin(), values.end());
    const double pos = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>(values[lo] * (1.0 - frac) + values[hi] * frac);
}

void PrintStats(const std::string &name, const std::vector<float> &ms,
                size_t bytes) {
    const double mean_ms = Mean(ms);
    const double gbps = static_cast<double>(bytes) / mean_ms / 1.0e6;
    std::cout << std::left << std::setw(28) << name
              << " mean_ms=" << std::right << std::setw(9)
              << std::fixed << std::setprecision(3) << mean_ms
              << " p50_ms=" << std::setw(9) << Percentile(ms, 0.50)
              << " p90_ms=" << std::setw(9) << Percentile(ms, 0.90)
              << " p95_ms=" << std::setw(9) << Percentile(ms, 0.95)
              << " p99_ms=" << std::setw(9) << Percentile(ms, 0.99)
              << " GB/s=" << std::setw(9) << std::setprecision(2) << gbps
              << std::endl;
}

int TimeStreamWork(aclrtStream stream, const std::string &name, int32_t warmup,
                   int32_t iters, size_t bytes,
                   const std::function<int()> &enqueue) {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    CHECK_ACL(aclrtCreateEventExWithFlag(&start, ACL_EVENT_TIME_LINE));
    CHECK_ACL(aclrtCreateEventExWithFlag(&end, ACL_EVENT_TIME_LINE));

    for (int32_t i = 0; i < warmup; ++i) {
        const int ret = enqueue();
        if (ret != 0) {
            aclrtDestroyEvent(start);
            aclrtDestroyEvent(end);
            return ret;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int32_t i = 0; i < iters; ++i) {
        CHECK_ACL(aclrtRecordEvent(start, stream));
        const int ret = enqueue();
        if (ret != 0) {
            aclrtDestroyEvent(start);
            aclrtDestroyEvent(end);
            return ret;
        }
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end));
        float elapsed_ms = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsed_ms, start, end));
        samples.push_back(elapsed_ms);
    }

    PrintStats(name, samples, bytes);
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    return 0;
}

int ValidateOutput(void *out_device, const std::vector<int32_t> &src_block_ids,
                   const std::vector<int32_t> &dst_block_ids,
                   size_t elems_per_block) {
    int32_t max_dst = 0;
    for (int32_t dst : dst_block_ids) {
        max_dst = std::max(max_dst, dst);
    }

    const size_t out_elems = static_cast<size_t>(max_dst + 1) * elems_per_block;
    std::vector<float> result(out_elems, 0.0f);
    CHECK_ACL(aclrtMemcpy(result.data(), out_elems * sizeof(float), out_device,
                          out_elems * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    int bad = 0;
    for (size_t pair = 0; pair < src_block_ids.size(); ++pair) {
        const int32_t src_page = src_block_ids[pair];
        const int32_t dst_page = dst_block_ids[pair];
        const size_t dst_base = static_cast<size_t>(dst_page) * elems_per_block;
        for (size_t i = 0; i < elems_per_block; ++i) {
            const float expected =
                static_cast<float>(src_page * 100000 + (i % 100000));
            const float actual = result[dst_base + i];
            if (std::fabs(actual - expected) > 1e-5f) {
                if (bad < 8) {
                    std::cerr << "mismatch[pair=" << pair
                              << ", dst_block=" << dst_page
                              << ", elem=" << i << "] = " << actual
                              << ", expected " << expected << std::endl;
                }
                ++bad;
            }
        }
    }
    if (bad != 0) {
        std::cerr << "validation failed: mismatches=" << bad << std::endl;
        return 2;
    }
    return 0;
}

void PrintUsage(const char *program) {
    std::cerr
        << "usage: " << program << " [device_id] [options]\n"
        << "options:\n"
        << "  --num-pages N           source host/HBM pages, default 1024\n"
        << "  --selected-blocks N     gathered block pairs, default 1024\n"
        << "  --dst-blocks N          output cache blocks, default selected-blocks\n"
        << "  --elems-per-block N     float32 elements per block, default 16384\n"
        << "  --src-pattern NAME      sequential|reverse|stride|random, default random\n"
        << "  --dst-pattern NAME      sequential|reverse|stride|random, default random\n"
        << "  --warmup N              warmup iterations, default 3\n"
        << "  --iters N               measured iterations, default 10\n";
}

bool IsPattern(const std::string &value) {
    return value == "sequential" || value == "reverse" ||
           value == "stride" || value == "random";
}

int ParseOptions(int argc, char **argv, Options &opt) {
    int start_arg = 1;
    bool dst_blocks_set = false;
    if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0) {
        opt.device_id = std::stoi(argv[1]);
        start_arg = 2;
    }
    for (int i = start_arg; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--num-pages" && i + 1 < argc) {
            opt.num_pages = std::stoi(argv[++i]);
        } else if (arg == "--selected-blocks" && i + 1 < argc) {
            opt.selected_blocks = std::stoi(argv[++i]);
        } else if (arg == "--dst-blocks" && i + 1 < argc) {
            opt.dst_blocks = std::stoi(argv[++i]);
            dst_blocks_set = true;
        } else if (arg == "--elems-per-block" && i + 1 < argc) {
            opt.elems_per_block = static_cast<size_t>(std::stoll(argv[++i]));
        } else if (arg == "--src-pattern" && i + 1 < argc) {
            opt.src_pattern = argv[++i];
        } else if (arg == "--dst-pattern" && i + 1 < argc) {
            opt.dst_pattern = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opt.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opt.iters = std::stoi(argv[++i]);
        } else {
            return 1;
        }
    }
    if (!dst_blocks_set) {
        opt.dst_blocks = opt.selected_blocks;
    }
    if (opt.num_pages <= 0 || opt.selected_blocks <= 0 ||
        opt.dst_blocks < opt.selected_blocks || opt.elems_per_block == 0 ||
        opt.warmup < 0 || opt.iters <= 0 || !IsPattern(opt.src_pattern) ||
        !IsPattern(opt.dst_pattern)) {
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    if (ParseOptions(argc, argv, opt) != 0) {
        PrintUsage(argv[0]);
        return 1;
    }

    const size_t block_bytes = opt.elems_per_block * sizeof(float);
    const size_t useful_bytes =
        static_cast<size_t>(opt.selected_blocks) * block_bytes;
    const std::vector<int64_t> pages_shape = {
        static_cast<int64_t>(opt.num_pages),
        static_cast<int64_t>(opt.elems_per_block)};
    const std::vector<int64_t> out_shape = {
        static_cast<int64_t>(opt.dst_blocks),
        static_cast<int64_t>(opt.elems_per_block)};

    aclrtStream stream = nullptr;
    TensorHandle src_ids_device;
    TensorHandle dst_ids_device;
    TensorHandle out;
    PagesHandle mapped_pages;
    PagesHandle device_pages;
    aclTensor *mapped_pages_tensor = nullptr;
    aclTensor *device_pages_tensor = nullptr;
    void *workspace = nullptr;
    uint64_t workspace_size = 0;
    int ret = 0;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(opt.device_id));
    CHECK_ACL(aclrtCreateStream(&stream));

    std::vector<float> host_pages;
    FillPages(host_pages, opt.num_pages, opt.elems_per_block);
    const std::vector<int32_t> src_block_ids = BuildSourceBlockIds(opt);
    const std::vector<int32_t> dst_block_ids = BuildDestinationBlockIds(opt);

    ret = CreateMappedHostPages(host_pages, mapped_pages);
    if (ret != 0) {
        goto cleanup;
    }
    ret = CreateDevicePages(host_pages, device_pages);
    if (ret != 0) {
        goto cleanup;
    }
    ret = CreateDeviceTensor({opt.selected_blocks}, ACL_INT32, sizeof(int32_t),
                             src_ids_device);
    if (ret != 0) {
        goto cleanup;
    }
    ret = CreateDeviceTensor({opt.selected_blocks}, ACL_INT32, sizeof(int32_t),
                             dst_ids_device);
    if (ret != 0) {
        goto cleanup;
    }
    ret = CreateDeviceTensor(out_shape, ACL_FLOAT, sizeof(float), out);
    if (ret != 0) {
        goto cleanup;
    }

    CHECK_ACL(aclrtMemcpy(src_ids_device.device_addr,
                          src_block_ids.size() * sizeof(int32_t),
                          src_block_ids.data(),
                          src_block_ids.size() * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dst_ids_device.device_addr,
                          dst_block_ids.size() * sizeof(int32_t),
                          dst_block_ids.data(),
                          dst_block_ids.size() * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    mapped_pages_tensor =
        CreateTensorView(pages_shape, ACL_FLOAT, mapped_pages.device_addr);
    device_pages_tensor =
        CreateTensorView(pages_shape, ACL_FLOAT, device_pages.device_addr);
    if (mapped_pages_tensor == nullptr || device_pages_tensor == nullptr) {
        std::cerr << "aclCreateTensor(pages) failed" << std::endl;
        ret = 1;
        goto cleanup;
    }

    {
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnKvCacheBlockGatherGetWorkspaceSize(
            src_ids_device.tensor, mapped_pages_tensor, dst_ids_device.tensor,
            out.tensor, &workspace_size, &executor));
        if (workspace_size > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspace_size,
                                  ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }

    std::cout << "shape: pages=" << opt.num_pages
              << ", selected_blocks=" << opt.selected_blocks
              << ", dst_blocks=" << opt.dst_blocks
              << ", block_bytes=" << block_bytes
              << ", useful_MiB=" << std::fixed << std::setprecision(2)
              << static_cast<double>(useful_bytes) / (1024.0 * 1024.0)
              << ", src_pattern=" << opt.src_pattern
              << ", dst_pattern=" << opt.dst_pattern << std::endl;

    ret = TimeStreamWork(
        stream, "mapped-host gather op", opt.warmup, opt.iters, useful_bytes,
        [&]() -> int {
            aclOpExecutor *executor = nullptr;
            uint64_t size = 0;
            CHECK_ACL(aclnnKvCacheBlockGatherGetWorkspaceSize(
                src_ids_device.tensor, mapped_pages_tensor,
                dst_ids_device.tensor, out.tensor, &size, &executor));
            if (size > workspace_size) {
                std::cerr << "workspace grew from " << workspace_size << " to "
                          << size << std::endl;
                return 1;
            }
            CHECK_ACL(aclnnKvCacheBlockGather(workspace, workspace_size,
                                              executor, stream));
            return 0;
        });
    if (ret != 0) {
        goto cleanup;
    }
    ret = ValidateOutput(out.device_addr, src_block_ids, dst_block_ids,
                         opt.elems_per_block);
    if (ret != 0) {
        goto cleanup;
    }

    ret = TimeStreamWork(
        stream, "aclrtMemcpyAsync/page", opt.warmup, opt.iters, useful_bytes,
        [&]() -> int {
            for (int32_t pair = 0; pair < opt.selected_blocks; ++pair) {
                const int32_t src_page =
                    src_block_ids[static_cast<size_t>(pair)];
                const int32_t dst_page =
                    dst_block_ids[static_cast<size_t>(pair)];
                const char *src =
                    static_cast<const char *>(mapped_pages.host_addr) +
                    static_cast<size_t>(src_page) * block_bytes;
                char *dst = static_cast<char *>(out.device_addr) +
                            static_cast<size_t>(dst_page) * block_bytes;
                CHECK_ACL(aclrtMemcpyAsync(dst, block_bytes, src, block_bytes,
                                           ACL_MEMCPY_HOST_TO_DEVICE, stream));
            }
            return 0;
        });
    if (ret != 0) {
        goto cleanup;
    }
    ret = ValidateOutput(out.device_addr, src_block_ids, dst_block_ids,
                         opt.elems_per_block);
    if (ret != 0) {
        goto cleanup;
    }

    ret = TimeStreamWork(
        stream, "aclrtMemcpyAsync/contig", opt.warmup, opt.iters, useful_bytes,
        [&]() -> int {
            CHECK_ACL(aclrtMemcpyAsync(out.device_addr, useful_bytes,
                                       mapped_pages.host_addr, useful_bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
            return 0;
        });
    if (ret != 0) {
        goto cleanup;
    }

    ret = TimeStreamWork(
        stream, "HBM gather op", opt.warmup, opt.iters, useful_bytes,
        [&]() -> int {
            aclOpExecutor *executor = nullptr;
            uint64_t size = 0;
            CHECK_ACL(aclnnKvCacheBlockGatherGetWorkspaceSize(
                src_ids_device.tensor, device_pages_tensor,
                dst_ids_device.tensor, out.tensor, &size, &executor));
            if (size > workspace_size) {
                std::cerr << "workspace grew from " << workspace_size << " to "
                          << size << std::endl;
                return 1;
            }
            CHECK_ACL(aclnnKvCacheBlockGather(workspace, workspace_size,
                                              executor, stream));
            return 0;
        });
    if (ret != 0) {
        goto cleanup;
    }
    ret = ValidateOutput(out.device_addr, src_block_ids, dst_block_ids,
                         opt.elems_per_block);

cleanup:
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (mapped_pages_tensor != nullptr) {
        aclDestroyTensor(mapped_pages_tensor);
    }
    if (device_pages_tensor != nullptr) {
        aclDestroyTensor(device_pages_tensor);
    }
    DestroyTensorHandle(out);
    DestroyTensorHandle(dst_ids_device);
    DestroyTensorHandle(src_ids_device);
    DestroyPagesHandle(device_pages);
    DestroyPagesHandle(mapped_pages);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(opt.device_id);
    aclFinalize();
    return ret;
}
