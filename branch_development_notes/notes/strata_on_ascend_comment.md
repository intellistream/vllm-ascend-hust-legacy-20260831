对，这个 prototype 结果很关键。它基本把问题从 **“device-driven host KV read 在 Ascend 上能不能成立？”** 推进到了 **“它在哪些场景值得默认启用，怎么不把生产栈搅成一锅小电鳗？”**

我的更新结论是：

**继续做 CANN 上的 vLLM/Strata-style custom kernel CPU KV offloading 路线是值得的，尤其是你们已经证明它在零散随机 KV 读里有收益。** 但它应该被定义成：

```text
device-driven scattered host KV onload/gather backend
```

而不是：

```text
所有 KV transfer 的统一替代品
```

也就是说，**改掉 legacy SharedMemory 是对的，保留 KV transfer 价值也是对的；device-driven mmap/read 是一个很锋利的后端，适合小页、碎片、随机、layout transform，不适合无脑替代大块连续 copy。**

---

## 1. Strata 的收益不是“mmap 随机读天然比 copy 快”

Strata 真正的 claim 更精确：

> 传统 CPU→GPU KV load 在 paged KV layout 下被碎片化成很多小 I/O，`cudaMemcpyAsync`/DMA 不能吃满带宽；GPU-assisted I/O kernel 用大量 GPU threads 直接从 registered pinned host memory 读小块，再写到 GPU HBM，从而让小页 KV 也能高并发、高带宽地搬运。

它不是让 attention 直接长期读 host KV 来算。它仍然主要是在 **load/onload 阶段把 KV 从 CPU DRAM 搬到 GPU HBM**，只是搬运动作由 GPU kernel 发起，并顺手做 layout transform。

Strata 在 Introduction 和 Figure 1 里说得很清楚：长上下文场景下，CPU memory 到 GPU HBM 的 KV cache load 会成为主要瓶颈，PagedAttention 把一个 sequence 的 KV 分散在多个非连续 pages 上，导致很多只有几 KB 的小 transfer，带宽吃不满；即使用它的 I/O-only 优化，仍然有最高 24% 的 prefill execution time 卡在 cache loading 上，所以还需要 scheduler 参与。

这对你们的 prototype 很重要：你们已经证明 **零散随机 KV 读有收益**，这刚好命中 Strata 的甜点区。收益不是来自“host DRAM 神秘加速”，而是来自 **用 device 端大规模并发把很多小碎片读请求压成一场有组织的蜂群飞行** 🐝。

---

## 2. Strata 的收益来源拆开看

### A. 小页 KV cache 让传统 DMA 很吃亏

Strata 在 §3.1 用 Little’s Law 分析 I/O throughput，核心公式可以理解为：

```text
throughput ≈ concurrency × transfer_size / latency
```

传统 DMA 路线想吃满 PCIe 5.0，一般需要 MB 级 transfer size。可是 LLM serving 为了 cache hit rate 和内存利用率，喜欢 1 到 32 token 的小 page。Strata 的 Figure 2 显示 page size 变大虽然改善 transfer size，但 cache hit rate 会掉，TTFT 反而上升；Figure 3 进一步显示，加载 Llama-3.1-8B 的 8192 tokens KV、page size 32 时，只能达到 PCIe 5.0 理论带宽大约 22%，在 GH200 这类更高 CPU-GPU 带宽平台上利用率甚至更低。

所以 Strata 不是简单说 “DMA 不行”，而是说：

```text
small page + fragmented layout + many tiny copies
    =>
DMA path underutilizes interconnect
```

这点和 vLLM 那篇文章形成了一个很有意思的对照。

vLLM 的新 CPU KV offloading connector 最后偏向 DMA，是因为它改了 KV cache physical layout，把原本每层/每 K/V 分散的 block 合成更大的物理 block，典型从几 KB 提升到 0.5 到 2MB。vLLM 文章也承认 custom CUDA kernel 对小 block 更好，但 block 变大之后 DMA 更稳，双向 2MB transfer 里 DMA 是 83.4 GB/s，custom kernel 是 68.5 GB/s；端到端并发测试里 custom kernel 会干扰模型计算，DMA 反而有更好 throughput，Llama-3.1-8B 上最高多 32%。([vLLM][1])

这说明两篇文章不是互相打架，而是在说两个不同区域：

```text
vLLM 新 layout：
    把 KV transfer 变成大块连续 copy，所以 DMA 更香。

Strata：
    保留小 page 来保护 hit rate，同时用 device kernel 消化碎片 I/O。
```

你们在 Ascend 上如果面对的是 **零散随机 KV read**，那就更像 Strata；如果后面能把 KV onload 全部 coalesce 成 MB 级连续 block，那就更像 vLLM 新 DMA 路线。

---

### B. GPU-assisted I/O 的核心收益是并发和 layout transform

Strata 的 §4.2 说它不用反复调用小 `cudaMemcpyAsync`，而是启动 CUDA kernel，让大量 threads 从 GPU global memory 或 CPU registered pinned memory 读取小 chunk，再写到目标内存。它列出的三个优势是：更高并发、兼容小粒度 transfer、可以几乎免费做 layout transform。

这第三点很容易被忽略，但对你们改共享内存很关键。

Strata 的 Figure 6 是 **GPU layer-first layout vs host page-first layout**。GPU HBM 里保持 layer-first，因为计算 kernel 喜欢这个 layout；CPU DRAM 或 disk 里用 page-first，因为 transfer 更连续。device-side I/O kernel 在搬的时候顺手做地址变换。Figure 12 里 page-first layout 把从 disk 到 CPU 的 8192-token KV load latency 从 1.687/1.739/2.102 秒降到 0.420/0.638/1.202 秒，模型分别是 Llama-8B、Qwen-14B、Llama-70B。

映射到你们这边，最重要的设计含义是：

```text
Host swap pool 不应该盲目复制 NPU KV layout。
Host swap pool 应该允许 transfer-friendly layout。
Device-side custom kernel 负责 host layout -> NPU layout 的变换。
```

这也是为什么 legacy global SharedMemory 很别扭。它更像“大家共享同一块老式仓库”，而 Strata-style kernel 想要的是 **worker/device 本地拥有、可注册、可按 transfer-friendly layout 组织的 host arena**。你们 RFC 里把第一阶段定成 worker-local host swap pool + baseline copy path，再后接 mapped-host gather/read-write backend，这个顺序正好匹配这个结论。

---

### C. Interference 是 Strata 成败的关键，不是旁枝

Strata 很诚实地承认 GPU-assisted I/O 会抢 GPU resources：register file、execution cycles、cache pollution，都会影响模型计算。它的解决方式是在 H200 上只用很少的 CUDA blocks，让 I/O kernel 被限制在少数 SM 上。Figure 5 显示，使用两个 1024-thread CUDA blocks 时，Strata 接近 50 GB/s transfer throughput，同时 prefill 降低小于 5%，decode 降低约 10%；它默认 HtoD 用两个 blocks，DtoH 用一个 block，端到端影响控制在 5% 内。

这点迁移到 Ascend 上不能照抄。

CUDA 的 “two blocks on a few SMs” 不等于 Ascend 上自然存在一个相同的资源隔离旋钮。你们需要在 CANN/Ascend C 里找到等价的 **AI Core 预算、tiling 策略、stream 优先级、event 同步、cache bypass 或污染控制手段**。如果 prototype 已经能在随机读里跑出收益，说明读路径是通的；但 production 还要证明：

```text
custom kernel 的收益 > 它对 prefill/decode compute 的干扰
```

尤其是 decode-heavy workload。这里最容易出现“microbenchmark 像银色小火箭，端到端像拖着降落伞的滑板车”的现象。

---

### D. Strata 的 headline gain 不是纯 I/O kernel 带来的

Strata 摘要里说长上下文 benchmark 上相比 vLLM + LMCache 最高 5× lower TTFT，相比 TensorRT-LLM 最高 3.75× speedup。这个 headline gain 不是只靠 device-driven host read，而是 **GPU-assisted I/O + cache-aware scheduling** 叠出来的。

拆解实验更能说明问题。Figure 9 里，`Strata-Schedule-Only` 和 `Strata-IO` 分别能带来最高 1.8× 和 2.3× peak throughput 提升；低 request rate 时 scheduling 更重要，高 request rate 时 I/O subsystem 变成主瓶颈，GPU-assisted I/O 更重要。

Figure 11 进一步按 cache distance 拆：在 shuffle/max cache distance 场景里，I/O efficiency 分别带来 76% 和 95% peak throughput improvement；balance batch 再带来 11%/12%，stall hiding 再带来 8%/3%。

所以对你们的 Ascend 集成，我会这样预期：

```text
只做 custom kernel：
    能拿到 Strata-IO 那部分收益，特别是高 I/O 压力、随机小页读。

再做 layout 和 scheduling：
    才有机会接近 Strata headline 的端到端收益。
```

如果只实现 CANN custom kernel，而 scheduler 仍然把 load/cache ratio 很高的请求塞成 loading-bound batch，那系统还是会在 I/O stall 里打转。Strata 的 Figure 1 已经说了，即使 I/O-only 做掉小页 transfer overhead，仍然最高有 24% prefill time 是 loading stall。

---

## 3. CANN/Ascend 上最关键的现实约束

CANN 确实有 `aclrtHostRegister` 这类 host memory registration API。官方文档说它把 Host 内存映射注册为 Device 可访问地址，并且和 `aclrtHostUnregister` 成对使用；支持矩阵上 A2/A3 训练/推理系列支持，但一些 200I/500 A2 推理产品和旧的 Atlas 推理系列不支持。文档还要求 host 地址 4K 页对齐，并说明 OS kernel 5.10 或更低时，非锁页内存会异常，需要通过 `aclrtMallocHost` 申请 Host 内存。([昇腾社区][2])

这带来几个生产判断：

第一，**机型边界要写进 RFC 和 config validation**。如果目标是 A2/A3，路线可行性强很多；如果要覆盖更老推理卡，就要提前明确 fallback。

第二，**host arena 最好用 CANN host allocation 或严格受控的 pinned allocator**。不要依赖 Python SharedMemory 这种不受 worker/device context 约束的生命周期。你们改共享内存的方向更对了。

第三，文档还说 mapped 后的 Device address 不能用于内存复制操作。这个限制反而支持你们的路线：它不是给普通 memcpy 用的，而是需要自定义 device op 去读写 mapped address。你们 prototype 已证明这条 op 路径能成立，但 production 还是要把支持矩阵、alignment、registration/unregistration、fallback 都写硬。

---

## 4. 在 Ascend 推理栈上，什么时候会有好效果？

我会给一个比较明确的判断：

**在你们已经证明“零散随机 KV 读有收益”的前提下，这条路线在 Ascend 生产环境里很可能有价值，但价值集中在特定 workload 区域，而不是全局无条件收益。**

### 高概率有效的区域

```text
1. long-context prefill
2. CPU KV cache hit rate 高
3. HBM 放不下完整 prefix，需要频繁 CPU onload
4. KV block/page 很小或很碎
5. requested CPU blocks 非连续、乱序、partial hit
6. 当前 copy path 需要大量小 transfer 或 gather/scatter
7. host layout 和 NPU layout 不一致，需要顺手做 layout transform
8. prefill compute 足够少，无法掩盖 KV load latency
```

你们 prototype 的“零散随机 KV 读收益”已经覆盖了第 4、5、6 点，这是最平台相关、最难靠论文保证的一块。这里可以把风险评级降一档：**不是 feasibility 风险，而是 productionization 和 policy 风险。**

### 低收益或可能负收益的区域

```text
1. active swap reload 是大块连续 exact reload
2. save/offload 是 NPU -> CPU 大块顺序写
3. KV physical block 已经能合并到 0.5MB 到 2MB
4. decode-heavy workload，custom kernel 抢 AI Core 影响 ITL
5. 短上下文，TTFT 主要不是 KV load
6. batch 中 model compute 已经能完全 hide CPU load
7. registration/window 管理造成额外抖动
```

vLLM 那篇文章给了一个很好的反例：一旦 layout 改成大块物理 block，DMA/copy 在端到端吞吐上反而比 custom kernel 更好，因为 custom kernel 会干扰模型计算。([vLLM][1])

所以生产策略不应该是：

```text
enable_custom_kernel = true
```

而应该是：

```text
backend = policy(copy, custom_kernel)
```

---

## 5. 我建议的 runtime policy

可以把 transfer backend 选择写成这样：

```python
def choose_kv_transfer_backend(plan, runtime):
    if not runtime.custom_kernel_enabled:
        return "copy"

    if plan.direction == "NPU_TO_CPU_SAVE":
        return "copy"  # unless measured otherwise

    if plan.intent_type == "ACTIVE_SWAP_RELOAD" and plan.is_large_contiguous:
        return "copy"

    if plan.avg_contiguous_run_bytes >= runtime.copy_favorable_threshold:
        return "copy"

    if plan.fragment_count >= runtime.scatter_min_fragments \
       and plan.avg_fragment_bytes <= runtime.scatter_fragment_threshold \
       and runtime.io_core_budget_available \
       and runtime.recent_interference_ok:
        return "device_driven_host_gather"

    return "copy"
```

第一版阈值可以通过 microbenchmark profile 自动生成：

```text
copy_favorable_threshold:
    从多少 bytes/run 开始 copy path 追平或超过 custom kernel

scatter_fragment_threshold:
    小于多少 bytes/fragment custom kernel 稳赢

io_core_budget:
    custom kernel 占用多少 AI Core / block / task 时不明显影响 prefill/decode

interference_ok:
    最近窗口内 ITL / prefill latency regression 是否低于阈值
```

这比 feature flag 更像生产系统。feature flag 是开关刀，policy 是调音台。

---

## 6. 你们改 SharedMemory 的动机可以更强地写成 Strata-style

我建议把 RFC 里的动机从：

```text
为了 mapped-host gather，global SharedMemory 生命周期不合适
```

扩展成：

```text
为了实现 Strata-style device-driven host KV I/O，
worker 必须拥有可注册、可布局、可同步、可回收的 host arena。
```

原因是 Strata 的收益要求 host memory layout 可以为了 I/O 优化。legacy global SharedMemory 通常会把大家绑在一个“统一内存池”的抽象上，表面统一，实际不适合：

```text
- CANN host register / unregister 生命周期
- mapped device pointer 生命周期
- worker/device/context 绑定
- page-first host layout
- per-worker NUMA / device affinity
- stream/event 同步
- custom kernel index/workspace buffer 复用
- production fallback 和 metrics
```

你们不是在“放弃 KV transfer”。你们是在把 KV transfer 从旧仓库搬到一个更适合 CANN device-driven I/O 的机房里。旧仓库门牌还在，但电线绕得像章鱼打毛线球。

---

## 7. Strata claim 对 Ascend 的可迁移性评估

我会这样打分：

| Strata claim                             | 迁移到 Ascend 的把握 | 原因                                                               |
| ---------------------------------------- | -------------: | ---------------------------------------------------------------- |
| 小页/碎片 KV load 会让传统 copy/DMA 吃不满带宽        |              高 | 这是 I/O 基本规律，你们 prototype 也侧面验证                                   |
| device kernel 对零散随机 host KV 读有收益         |              高 | 你们 prototype 已经证明                                                |
| layout transform 可以顺手融合进 transfer kernel |             中高 | 需要 Ascend C 地址计算和 layout 校验，但技术上合理                               |
| 低干扰地和 prefill/decode 并行                  |              中 | CUDA 的 SM block 控制不能直接照搬到 AI Core                                |
| Strata-IO alone 的 2.3× peak throughput   |              中 | 取决于 Ascend interconnect、copy baseline、page size、kernel occupancy |
| Strata full 最高 5× TTFT / 3.75× speedup   |             中低 | 这包括 scheduling、workload、H200/GH200、SGLang、长上下文和 1TB pinned DRAM  |
| GH200 near-oracle 结论                     |            低到中 | GH200 CPU-GPU coherent/high-bandwidth 特性和 Ascend 机型不同            |

Strata 的 Figure 14 很有启发：在 GH200 上，SGLang-HiCache 的 sustained bandwidth 是 19.43 GB/s，Strata-IO-GH 是 150.50 GB/s；在 PCIe H200 上，SGLang-HiCache 是 10.80 GB/s，Strata-IO 是 40.30 GB/s。它还说硬件带宽提升本身不够，软件如果仍然小块 DMA，就吃不满；Strata-IO-GH 虽然带宽高，但仍不如 Strata-PCIe，说明 scheduler 也必须跟上。

这对 Ascend 的启示是：

```text
只证明 device read 快还不够。
要证明它能提高端到端 TTFT/throughput，并且不恶化 ITL。
```

---

## 8. 我会怎么调整你们的 RFC

建议新增一个小节，叫：

```text
Strata-Style Device-Driven Host KV I/O
```

里面写清楚：

```text
This project does not abandon KV transfer.
It abandons the legacy global SharedMemory ownership model.

The CANN custom kernel backend targets fragmented, small-page,
non-contiguous CPU KV onload, where baseline copy/DMA-like paths
underutilize bandwidth or require expensive gather/scatter.

The copy path remains the correctness baseline and preferred backend
for large contiguous transfers.
```

再加一个 backend policy：

```text
Use copy path for:
- NPU -> CPU save/offload by default
- large contiguous CPU -> NPU reload
- active swap exact reload unless custom backend is proven better
- unsupported dtype/layout/device/product
- high recent compute interference

Use device-driven host gather for:
- fragmented small-page CPU -> NPU onload
- partial prefix hit
- non-contiguous block order
- host page-first -> NPU layer-first layout transform
- measured cases where p95 TTFT improves without p95 ITL regression
```

还要加几个 Ascend-specific guardrails：

```text
- host memory must satisfy CANN registration requirements
- product support must be checked at startup
- host address alignment must be validated
- registration failures fallback to copy
- mapped device pointer must never outlive worker_generation/context
- custom op must retain host tensors, index buffers, workspace, and ACL handles
- no mapped pointer use after unregister
```

---

## 9. 最小生产 benchmark gate

你们 prototype 已证明“能跑、随机读有收益”。下一步应该把收益边界画出来，不要让它变成玄学。

我建议最小 benchmark matrix 是：

| Benchmark                                                  | 目的                                    |
| ---------------------------------------------------------- | ------------------------------------- |
| fragment size sweep: 128B, 512B, 1KB, 4KB, 16KB, 64KB, 1MB | 找 custom vs copy 交叉点                  |
| fragment count sweep: 1, 8, 32, 128, 512, 2048             | 测 device 并发收益                         |
| contiguous run length sweep                                | 判断何时 copy 更好                          |
| random vs sorted block ids                                 | 看 index locality 和 host DRAM locality |
| HtoD only, DtoH only, bidirectional                        | 判断 save/load 默认策略                     |
| with prefill running                                       | 测 prefill regression                  |
| with decode running                                        | 测 ITL regression                      |
| long-context CPU-hit workload                              | 测 TTFT/throughput                     |
| partial prefix hit workload                                | 测真实收益区                                |
| TP/MLA layout                                              | 防止 layout transform 变成暗礁              |

必须记录这些指标：

```text
- custom kernel effective GB/s
- copy path effective GB/s
- p50/p95/p99 transfer latency
- p50/p95/p99 TTFT
- p50/p95/p99 ITL
- prefill latency regression
- decode latency regression
- NPU core occupancy
- stream wait time
- registration failure/fallback count
- mapped pointer validation failure
```

Gate 可以这样定：

```text
device-driven host gather 可默认启用，当且仅当：
1. targeted scatter workload p95 transfer latency 比 copy path 降低 >= 20%
2. p95 TTFT 改善 >= 10%
3. p95 ITL regression <= 3% 或明确只在 prefill window 使用
4. fallback/error rate 接近 0
5. long-running churn 无 registration/mapping leak
```

阈值可以调整，但一定要有门槛。没有门槛，优化就会变成一只会自己开 PR 的小怪兽。

---

## 最终判断

结合 Strata、vLLM 那篇文章、CANN registration 约束，以及你们 prototype 的结果，我会给这个路线一个更积极的 verdict：

**值得继续做，而且应该作为 worker-local host swap pool 之后的重点优化 backend。**

但我会把它精确定义为：

```text
CANN Strata-style device-driven host KV I/O backend
for fragmented / random / small-page CPU KV onload.
```

不是：

```text
replacement for all KV transfer.
```

也不是：

```text
reason to keep global SharedMemory.
```

生产上最稳的路线仍然是：

```text
1. worker-local host swap pool
2. copy path 行为兼容和 correctness baseline
3. CANN host registration lifecycle
4. device-driven custom kernel backend
5. runtime policy 在 copy 和 custom kernel 之间选择
6. prefix-aware routing 和 I/O-aware scheduling
```

你们 prototype 证明了最闪的一片刀刃确实锋利。现在要做的是给它配刀鞘、限位器和验血报告，让它在生产环境里切瓶颈，而不是切到自己的手。

[1]: https://vllm.ai/blog/2026-01-08-kv-offloading-connector "Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput | vLLM Blog"
[2]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_1804.html "aclrtHostRegister-内存管理-运行时管理-acl API（C）-应用开发接口-API-CANN社区版8.5.0.alpha001开发文档-昇腾社区"
