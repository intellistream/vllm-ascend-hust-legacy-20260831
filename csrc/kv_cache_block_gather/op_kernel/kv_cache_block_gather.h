#ifndef KV_CACHE_BLOCK_GATHER_H
#define KV_CACHE_BLOCK_GATHER_H

#include "kernel_operator.h"
#include "kv_cache_block_gather_tiling_data.h"
#include "kv_cache_block_gather_tiling_key.h"

namespace NsKvCacheBlockGather {
using namespace AscendC;

// This file contains the actual device-side gather algorithm.  GlobalTensor
// wraps device-visible GM addresses; for srcPagesGM_ that address may refer to
// host RAM registered with ACL_HOST_REGISTER_MAPPED by the Torch binding.
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KvCacheBlockGather {
public:
    __aicore__ inline KvCacheBlockGather() {}
    __aicore__ inline void Init(GM_ADDR srcBlockIds, GM_ADDR srcPages,
        GM_ADDR dstBlockIds, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling,
        const KvCacheBlockGatherTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyBlock(int64_t pairIdx);

private:
    // Separate VECIN and VECOUT queues preserve the conventional
    // CopyIn -> local copy -> CopyOut pipeline.  Although the middle copy looks
    // redundant, measurements in README.md show that TQueBind was slower on
    // this mapped-host gather path.  BUFFER_NUM=2 gives each stage ping-pong UB
    // storage; increasing it to 4 also regressed performance.
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<int32_t> srcBlockIdsGM_;
    GlobalTensor<int32_t> dstBlockIdsGM_;
    GlobalTensor<T> srcPagesGM_;
    GlobalTensor<T> outGM_;

    int64_t selectedBlocks_ = 0;
    int64_t elemsPerBlock_ = 0;
    int64_t tileElems_ = 0;
};

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Init(
    GM_ADDR srcBlockIds, GM_ADDR srcPages, GM_ADDR dstBlockIds, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling,
    const KvCacheBlockGatherTilingData* tilingData)
{
    selectedBlocks_ = tilingData->selectedBlocks;
    elemsPerBlock_ = tilingData->elemsPerBlock;
    tileElems_ = tilingData->tileElems;
    // Workspace is present in the ACLNN launch ABI but is not consumed by the
    // current copy algorithm.  Host tiling still requests it, so do not infer
    // from this cast alone that workspace=0 is valid for every runtime.
    (void)workspace;
    (void)tiling;

    srcBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds, selectedBlocks_);
    dstBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds, selectedBlocks_);
    srcPagesGM_.SetGlobalBuffer((__gm__ T*)srcPages);
    outGM_.SetGlobalBuffer((__gm__ T*)out);

    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::CopyBlock(int64_t pairIdx)
{
    // The two ID arrays are the page table for this operation.  One mapping
    // pair says: copy src_pages[srcPageIdx] into out[dstPageIdx].
    int32_t srcPageIdx = srcBlockIdsGM_.GetValue(pairIdx);
    int32_t dstPageIdx = dstBlockIdsGM_.GetValue(pairIdx);
    // GetValue issues a scalar-pipeline load from GM (possibly served by L2)
    // into a scalar register.  It needs no explicit UB buffer or TQue, but it
    // is still a real memory transaction, so this form suits small metadata;
    // bulk payload movement should use DataCopy through UB instead.
    int64_t srcBase = static_cast<int64_t>(srcPageIdx) * elemsPerBlock_;
    int64_t dstBase = static_cast<int64_t>(dstPageIdx) * elemsPerBlock_;

    // Move one flat block in tileElems-sized chunks:
    //   mapped host/device GM -> input UB -> output UB -> NPU GM.
    int64_t copied = 0;
    while (copied < elemsPerBlock_) {
        int64_t remain = elemsPerBlock_ - copied;
        int64_t copyElems = remain < tileElems_ ? remain : tileElems_;

        LocalTensor<T> inLocal = inputQueue_.AllocTensor<T>();
        DataCopy(inLocal, srcPagesGM_[srcBase + copied], copyElems);
        inputQueue_.EnQue(inLocal);

        inLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outputQueue_.AllocTensor<T>();
        DataCopy(outLocal, inLocal, copyElems);
        outputQueue_.EnQue(outLocal);
        inputQueue_.FreeTensor(inLocal);

        outLocal = outputQueue_.DeQue<T>();
        DataCopy(outGM_[dstBase + copied], outLocal, copyElems);
        outputQueue_.FreeTensor(outLocal);

        copied += copyElems;
    }
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Process()
{
    // Grid-stride distribution across AIV cores.  For example, with 40 cores,
    // core 0 handles pairs 0, 40, 80, ... and core 1 handles 1, 41, 81, ... .
    for (int64_t pair = static_cast<int64_t>(AscendC::GetBlockIdx());
         pair < selectedBlocks_;
         pair += static_cast<int64_t>(AscendC::GetBlockNum())) {
        CopyBlock(pair);
    }
}

} // namespace NsKvCacheBlockGather

#endif

/*
 * =============================================================================
 * 学习笔记：从这个 gather 算子理解 Ascend AI Core
 * =============================================================================
 *
 * 这份笔记刻意放在 include guard 之后：它不参与算子实现，只是陪着源码一起
 * 阅读的“小课本”。不同 Ascend 型号的物理结构会有差异，下面描述的是编写
 * AscendC kernel 时最有用的编程模型，而不是某一颗芯片的完整硬件框图。
 *
 * 一、先建立整条执行链
 * --------------------
 *
 *   Python / vLLM
 *       |
 *       | 调用 torch.ops.vllm_ascend.kv_cache_block_gather(...)
 *       v
 *   Torch C++ binding
 *       |
 *       | 注册并映射 host 内存，创建 ACL tensor，查询 workspace
 *       v
 *   op_host tiling
 *       |
 *       | 决定 blockDim、tileElems、tiling key，并序列化 tiling data
 *       v
 *   ACL runtime 把 kernel 放进当前 stream
 *       |
 *       v
 *   多个 AI Vector Core 执行本文件中的 Process() / CopyBlock()
 *
 * Host 侧发起 kernel launch 通常是异步的。对 host 来说“调用已经返回”，并不
 * 等于 NPU 已经完成读取 mapped host memory；源内存在 stream 完成前必须保持
 * 注册、存活，而且 CPU 不能同时修改它，否则会形成数据竞争。
 *
 * 二、一个 AI Core 里可以先想象成什么
 * ------------------------------------
 *
 * 编写这个算子时，可以先采用下面这个简化模型：
 *
 *                         +--------------------------+
 *                         |       one AI Core        |
 *                         |                          |
 *   GM / mapped host ---> | MTE2 ---> local memory   |
 *                         |              (UB)        |
 *                         |               |          |
 *                         |        Vector / Scalar   |
 *                         |               |          |
 *   GM <----------------- | MTE3 <--------+          |
 *                         +--------------------------+
 *
 * 1. Scalar/control 部分
 *    执行循环、地址计算、分支以及 srcPageIdx/dstPageIdx 等标量工作。
 *
 * 2. Vector 部分
 *    对 UB 中的数据执行向量运算。通用算子常在这里做 Add、Mul、Cast 等。
 *    当前 gather 没有数学计算，所以它几乎没有使用 Vector 计算能力。
 *
 * 3. Cube 部分
 *    主要服务矩阵乘等计算，常配合 L1/L0A/L0B/L0C 等片上存储。本算子只是
 *    搬运数据，不需要 Cube；阅读本文件时暂时可以把它放在一边。
 *
 * 4. MTE（Memory Transfer Engine）
 *    负责不同存储层次之间的数据搬运。为了建立直觉，可以把 GM -> UB 的
 *    CopyIn 想成由 MTE2 完成，把 UB -> GM 的 CopyOut 想成由 MTE3 完成。
 *    DataCopy 是我们向这些搬运流水线提交工作的 AscendC 接口。
 *
 * 不同引擎可以异步、并行地工作。因此，源码中的语句顺序不能单独代表数据
 * 已经搬完；TPipe/TQue/TQueBind 还承担 buffer 生命周期和流水事件同步。
 *
 * 三、内存结构：GlobalTensor 和 LocalTensor 到底在哪里
 * -----------------------------------------------------
 *
 * 从远到近，可以先记住下面几层：
 *
 * 1. Host DRAM
 *    CPU 内存。普通情况下 AI Core 不能拿一个 CPU 虚拟地址直接解引用。本项目
 *    先通过 aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED) 注册页面，再取得
 *    device-visible alias。srcPagesGM_ 的地址可以因此落在 mapped host memory。
 *
 * 2. GM（Global Memory）
 *    kernel 可见的全局地址空间，通常指设备侧 HBM/DDR。GlobalTensor<T> 是对
 *    GM 地址的访问包装，不会因为 SetGlobalBuffer 就复制或分配数据。
 *    在这个实验里，同一个 GM 编程接口还包装了 mapped host 的 device alias。
 *
 * 3. L2/cache 等共享层次
 *    位于外部存储与 AI Core 之间，具体行为与芯片有关。普通 AscendC kernel
 *    通常不把它当成一块可随意分配的数组；先把它理解为数据通路上的共享缓存。
 *
 * 4. UB（Unified Buffer）
 *    AI Core 本地的片上 scratchpad，容量小、带宽高。LocalTensor<T> 指向这里。
 *    UB 不是所有 core 共享的：一个 core 不能直接读取另一个 core 的 LocalTensor。
 *    tileElems_ 的意义就是把大 block 切成一块块能够放进 UB 的 tile。
 *
 * 5. L1/L0/寄存器
 *    更靠近计算单元。它们对 matmul/cube kernel 很重要，但当前纯搬运 kernel
 *    的关键路径只有 mapped-host/GM -> UB -> GM。
 *
 * 所以要特别区分“可寻址”与“访问成本”：注册后的 host 页面能作为 GM 地址
 * 访问，不代表它拥有设备 HBM 的延迟和带宽。随机选择 page 的代价主要发生在
 * 外部地址转换和传输链路；进入某个 page 后，按连续 tile 搬运仍然非常重要。
 *
 * 四、三个很容易混淆的 block
 * ----------------------------
 *
 * 1. KV cache block/page
 *    业务数据单位。srcBlockIds[pair] -> dstBlockIds[pair] 描述一条页面映射。
 *
 * 2. kernel block / blockDim
 *    启动并行度。GetBlockIdx() 返回当前逻辑执行 block 的编号，GetBlockNum()
 *    返回本次 launch 的 block 数。op_host 根据可用 AIV core 数和任务量设置它。
 *
 * 3. UB tile
 *    单个 core 一次搬进 UB 的数据块，大小由 tileElems_ 决定。
 *
 * 本算子的 Process() 使用 grid-stride 分工。例如 blockDim=40 时：
 *
 *   core/block 0: pair 0, 40, 80, ...
 *   core/block 1: pair 1, 41, 81, ...
 *   ...
 *
 * 每个执行 block 运行同一份 kernel 程序，只是 GetBlockIdx() 不同。这和 CPU
 * 创建 40 个长期线程并不完全等价，但作为理解此处数据并行的模型已经足够。
 *
 * 五、当前 CopyBlock() 在一颗 core 上发生了什么
 * -----------------------------------------------
 *
 * 对一个 pairIdx：
 *
 *   1. 从 GM 读取 srcPageIdx 和 dstPageIdx；
 *   2. 计算两个页面的线性基址；
 *   3. 从 inputQueue_ 申请一块输入 UB；
 *   4. DataCopy: mapped-host/GM -> input UB；
 *   5. EnQue/DeQue 等待 CopyIn 完成；
 *   6. 从 outputQueue_ 申请一块输出 UB，并做 input UB -> output UB copy；
 *   7. 通过 outputQueue_ 等待局部 copy 完成；
 *   8. DataCopy: output UB -> device GM，然后释放两块 UB。
 *
 * 中间的 UB -> UB copy 看起来多余，我们也实测过 TQueBind 版本：它让同一块
 * UB 同时承担 VECIN/VECOUT，从语义上可以安全地删掉这次 payload copy。但在
 * 910B2/CANN 9.0 上，TQueBind 的 device event 时间反而更长，因此这里有意保留
 * 独立输入/输出队列。完整数据和解释见同目录 README.md。
 *
 * BUFFER_NUM=2 表示每个 queue 都有两块 tile buffer，可以做 ping-pong。理想
 * 流水是：
 *
 *   time --->
 *   MTE2:  [load tile 0] [load tile 1] [load tile 2] ...
 *   MTE3:                [store tile 0][store tile 1] ...
 *
 * 这样不同流水阶段有机会重叠。实测从 1 增加到 2 对 64 KiB block 有约 1.6%
 * event 收益；继续增加到 4 则回退，所以 buffer 深度仍必须通过 benchmark 选择。
 *
 * 六、core 之间怎样通信
 * ----------------------
 *
 * 当前算子刻意采用“不通信”的设计：每个 core 读取自己的 pair，并写入对应的
 * destination page。不同 core 只是在地址空间上共同看见 GM，没有共享 UB。
 *
 * 如果两个 pair 写到同一个 dstPageIdx，就会发生并发写冲突；本 kernel 内没有
 * 自动解决它。正确性依赖调用者提供无冲突的 block mapping，或者保证重复写的
 * 语义确实安全。
 *
 * 通用 Ascend kernel 若真的需要 core 间协作，通常会借助：
 *
 *   - GM/workspace 中的共享数据；
 *   - atomic 操作；
 *   - 芯片和 API 支持的 barrier / sync；
 *   - 拆成多个 kernel，利用 stream 中前后 kernel 的完成关系作为全局边界。
 *
 * 但这些机制比访问本地 UB 昂贵，也容易引入死锁或竞争。能像本算子这样把任务
 * 分成互不依赖的 page copy，通常是最简单、最稳健的并行方式。
 *
 * 七、写 Ascend 搬运算子时最值得盯住的事情
 * ------------------------------------------
 *
 * 1. 正确性
 *    地址范围、src/dst mapping、尾块、dtype，以及 DataCopy 的长度/对齐约束。
 *    如果数据长度不满足基础 DataCopy 的约束，需要使用合适的 pad/ext 接口或
 *    在 tiling/input contract 中保证对齐，不能只看 C++ 循环“似乎没有越界”。
 *
 * 2. 并行度
 *    core 太少无法覆盖延迟；core 太多又可能只是争抢同一条外部带宽，并增加
 *    很多小事务。blockDim 应由数据规模和实测决定，而不是永远取最大值。
 *
 * 3. 连续性和事务大小
 *    page 的选择可以随机，但每次选中后应尽量连续地搬一个足够大的 tile。
 *    这正是 mapped-host gather 能把“随机 page table”变成“多个连续 tile copy”
 *    的地方。
 *
 * 4. UB 使用量
 *    tile 越大，单次搬运效率可能越好，但会减少同一 core 上可用的 buffer 数量，
 *    也可能阻碍双缓冲或其他局部计算。UB 是需要认真预算的片上资源。
 *
 * 5. 流水同步
 *    EnQue/DeQue 不只是代码仪式。删掉队列、复用 LocalTensor 或提前 FreeTensor
 *    之前，必须确认异步 MTE 的生产者/消费者依赖仍然正确。
 *
 * 6. 测量
 *    kernel launch 是异步的，不能只测 host 提交耗时。需要在同一 stream 上用
 *    正确的 event，或做 device synchronization。先验证全量非零数据，再相信
 *    带宽数字；一个漂亮但没有等待 kernel 完成的结果没有意义。
 *
 * 八、带着哪些问题继续读这份代码
 * ------------------------------
 *
 *   - srcBlockIdsGM_.GetValue() 会不会成为大量小标量 GM 读取？
 *   - tileElems_=1024 是否适合 4 KiB、16 KiB、64 KiB 三种 block？
 *   - profiler 能否解释双队列为何比省掉 UB copy 的 TQueBind 更快？
 *   - 更多 AIV core 是提高带宽，还是更快撞上 host interconnect 上限？
 *   - 尾 tile 和 DataCopy 对齐约束是否被所有合法 shape 满足？
 *   - block mapping 若含重复 dst id，调用契约应该禁止还是定义其语义？
 *
 * 这些问题把“代码能运行”推进到“我们理解它为什么正确、为什么快”。
 * =============================================================================
 */
