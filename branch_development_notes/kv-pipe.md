根据 2026-06-17 这版 revised RFC，我会把结论说得很明确：**放弃旧的 device KV transfer 技术路线作为 production 主线是值得的；但不要放弃“KV transfer / bulk copy / RDMA transfer”这个能力本身。** 旧路线可以退成 fallback、baseline、compatibility layer 和 benchmark reference，新主线应该是 worker-local host swap pool，后面再接 device-driven mapped/random host DRAM KV access。RFC 现在也已经把 mapped-host gather/read-write 定位成后续可选 transfer backend，而不是第一阶段承重墙。

我会把判断压缩成一句话：

> **放弃旧实现路线，保留 bulk transfer 能力；让 device-driven random DRAM access 成为“按需稀疏/非连续加载”的优化 backend，而不是替代所有 KV transfer 的圣剑。**

---

## 先澄清：这里有三种“KV transfer”，别把它们揉成一团面团

我们现在讨论的“旧路线”至少可能指三件事：

第一种是 **legacy global SharedMemory CPU KV pool**：central metadata server 分配 CPU KV cache，worker 从 Python SharedMemory handle 重建 tensor，scheduler 询问 metadata server 要 prefix hit 和 CPU block id。这个路线在你们 RFC 里已经被指出和 mapped-host gather 的生命周期不匹配，因为 host registration、mapped pointer、stream sync、unregister 都应该由真正使用内存的 worker 负责。

第二种是 **bulk copy / DMA-style KV transfer**：把 CPU KV cache 作为大块数据搬回 GPU/NPU KV cache，再让 attention 正常从 HBM 读。这条路本身并不坏，vLLM 最新 CPU offloading connector 反而证明它非常有价值，尤其是 block layout 做大以后。vLLM 的公开评测里，CPU KV cache load 相比重新 prefill 可以把单请求 TTFT 降低 2× 到 22×，并且在 10,000 个 512-token 请求的并发测试里，throughput 随 CPU cache hit rate 上升，最高提升到 9×。([vLLM][1])

第三种是 **device-driven host DRAM access / mapped-host gather / random access**：device kernel 直接从 mapped host memory 按 index/gather 读 CPU DRAM 里的 KV block，可能再写入 NPU KV cache，或者更激进地直接参与 attention。这条路不是传统 bulk transfer，它的收益来自“少搬”和“按需”，不是来自“搬得更快”。

所以最终决策不应该是：

```text
KV transfer vs device random DRAM access
```

而应该是：

```text
legacy global/shared-memory ownership  该放弃
bulk copy/DMA transfer                  必须保留
device-driven mapped/random access      作为条件触发的优化 backend
```

---

## 文献和系统工作怎么说：KV transfer 不是小玩具，它确实赚大钱

### 1. vLLM CPU KV offloading：bulk copy 路线仍然很强

vLLM 2026 年的新 KV offloading connector 明确把 CPU DRAM offload 作为 native feature，目标是利用更大的 CPU RAM 避免 preemption 后重算 KV，并通过 asynchronous load/store 让 engine 不被外部 KV I/O 阻塞。它还说明 CPU DRAM 容量通常大于 GPU memory，CPU-GPU transfer 具有较低延迟和较高吞吐，适合处理 request preemption。([vLLM][1])

更关键的是 vLLM 对 transfer 技术路线做了一个很接近我们问题的对比：CPU backend 用 `cudaMemcpyAsync` 走 DMA；另一条路线是 custom CUDA kernel，用 GPU cores 从 host memory 拷贝 16-byte words。结果是，小 block 下 custom kernel 的 microbenchmark 可能更好，但它会占 GPU cores，和模型计算互相干扰；在 2MB block 的双向传输里，DMA 达到 83.4 GB/s，custom kernel 是 68.5 GB/s。vLLM 还把 KV physical block size 从几 KB 改到约 0.5 到 2MB，使 DMA 更吃香，并在 end-to-end 里观察到 DMA 对 Llama-3.1-8B-Instruct 最高比 custom kernel 多 32% throughput，同时 TTFT 相当。([vLLM][1])

这个结论对我们非常重要：**如果“旧 device KV transfer”指的是用 device kernel 做 full-block host↔device copy，那它不是天然赢家。** 只要 layout 能变成大块连续传输，bulk DMA/copy path 往往更稳、更少干扰、更好调度。

### 2. LMCache：KV movement 是 serving substrate，不是边角功能

LMCache 的技术报告把 KV movement 做成独立层，worker 负责在 GPU memory、CPU/disk tiers 和 other workers 之间移动 KV cache，支持 CPU/disk offloading 和 prefill-decode disaggregation。它使用 kernel-optimized GPU buffers、async chunked I/O、layer-wise pipelining，目标是在 vLLM/SGLang 这种 paged memory 场景下也维持接近 GPU-resident bandwidth。([lmcache.ai][2])

LMCache 也点出了一个和我们高度相关的工程规律：小消息传输很亏。报告里给出的 RCCL transfer throughput 从 64KB 时的 4GB/s，涨到 1MB 时 30GB/s，10MB 到 100MB 时约 46 到 49GB/s；这说明 KV transfer 系统通常要做 coalescing、chunking、layer-wise pipeline，而不是把世界拆成一地碎玻璃。([lmcache.ai][2])

在 end-to-end 评测里，LMCache 在五个模型上相对最强 baseline 在相同 TTFT 下有 2.3 到 14× query processing rate 提升，远端 backend 场景也有 1.3 到 3× inference throughput 提升，但报告同时承认 remote backend load latency 会比 CPU memory 更高，短输入或小模型时 load delay 甚至可能超过 prefill delay。([lmcache.ai][2])

这给我们的启示是：**KV transfer 的收益真实存在，但它强依赖 workload、粒度、命中率、介质带宽和是否能 overlap。**

### 3. Mooncake / Dynamo / NIXL：跨 worker、P/D disaggregation 的 KV transfer 收益巨大，但那是另一类问题

Mooncake 是 KVCache-centric disaggregated architecture。它利用 GPU cluster 中 CPU、DRAM、SSD 等资源形成 disaggregated KVCache，并用 KVCache-centric scheduler 在吞吐和 SLO 之间取平衡。论文报告在模拟场景里最高有 525% throughput 增加，真实 Kimi workload 里能处理 75% 更多请求。([arXiv][3])

Mooncake FAST’25 版本还报告了 RDMA transfer engine 的实测：在 4×200Gbps 和 8×400Gbps 网络配置下分别达到 87GB/s 和 190GB/s，约比 TCP 快 2.4× 和 4.6×。FAST slides 里也写到 Mooncake 相比之前基于 vLLM 的系统，让 Kimi 在 A800 和 H800 clusters 上分别多处理 115% 和 107% 请求，并在有效请求容量上最高提升 498%。

NVIDIA Dynamo 的 disaggregated serving 文档也把 efficient KV transfer 作为核心：用 NIXL 将 KV cache 直接从 prefill engine 的 VRAM transfer 到 decode engine 的 VRAM，并且 transfer 是 non-blocking，让 GPU forward pass 可以继续服务其他请求。Dynamo 还让 prefill worker 通过 RDMA read/write 直接读写 remote KV blocks，不需要 remote worker engine 显式参与。([NVIDIA Docs][4])

这些工作不能直接证明你们应该保留 legacy SharedMemory pool，但它们强烈证明一件事：**KV transfer 作为跨阶段、跨节点、跨 worker 的系统能力非常值钱。** 只是它们走的是 RDMA/NIXL/Mooncake Transfer Engine 这类高性能、生命周期清楚的数据面，不是 Python SharedMemory + central metadata server 这种旧式内存主权模型。

### 4. CacheGen / InfiniGen：减少 KV 传输量是主旋律，和 mapped random access 同向

CacheGen 关注“网络上 fetch KV 太大”的问题，用 KV tensor encoder 压缩 cache 表示，并根据带宽调整压缩级别。它报告 KV cache size 降低 3.5 到 4.3×，fetch + context processing 总延迟降低 3.2 到 3.7×，质量影响很小。([arXiv][5])

InfiniGen 更贴近 device/random access 的思想：它指出 CPU-offloaded KV cache 允许更长 context，但从 CPU memory 把巨大 KV cache 搬回 GPU 会成为瓶颈；它通过预测下一层 attention 需要的关键 KV entries，只 prefetch essential KV，而不是整段 KV 都搬回 GPU。论文报告最高 3.00× speedup，并且 accuracy 最多提升 32.6 个百分点。

InfiniGen 对我们很有参考价值，因为它支持一个判断：**对 long-context decode，不是“怎样把全部 KV 搬快一点”这么简单，而是“能不能只搬真正需要的那一小部分”。** 这正是 device-driven random host DRAM KV access 的甜点区。

### 5. Tutti / Grace Hopper：行业趋势确实在朝 device/GPU-centric I/O 走，但硬件条件很挑

Tutti 是 2026 年的 SSD-backed KV cache 工作，它明确把 CPU 从 HBM 和 SSD 之间的关键 data/I/O control path 里拿掉，提供 GPU-native object abstraction、GPU direct object I/O 和 slack-aware I/O scheduling。它报告相比 SSD-backed LMCache with GDS，在 strict SLO 下 TTFT 降低 78.3%，achievable request rate 提升 2×，成本降低 27%。([arXiv][6])

NVIDIA Grace Hopper / Grace Blackwell 的统一内存路线也说明，如果 CPU 和 GPU 之间有 NVLink-C2C 这种 900GB/s coherent interconnect，CPU/GPU 可以共享统一地址空间，避免显式 transfer 和冗余 copy；但这个结论高度依赖硬件互联，不能直接外推到普通 PCIe 或 Ascend NPU 的 host DRAM mapped access。([NVIDIA Developer][7])

这说明 device-driven access 是大方向之一，但不是魔法门。硬件互联不够强时，随机 host DRAM access 很容易把 device kernel 拖成踩棉花。

---

## 所以，为了 device-driven random DRAM KV access 放弃旧路线，值不值？

我的答案是：**放弃旧路线作为主线，值；完全放弃 KV transfer 能力，不值。**

更细一点：

### 值得放弃的部分

**1. 值得放弃 legacy global SharedMemory CPU pool。**
这个路线最大收益是 node-local cross-worker prefix reuse，但代价是 central metadata、global CPU block id、跨进程 shared memory handle、worker 重建 tensor，以及很难和 host registration / mapped pointer / stream lifecycle 对齐。你们 revised RFC 已经明确 worker 应该拥有 physical residency 和 transfer execution，router 只消费 advisory locality signal，而且 compatibility layer 只保留 logical behavior，不保留 global physical ownership。这个判断非常稳。

**2. 值得放弃“device kernel 做 full bulk copy”的旧技术冲动。**
vLLM 的公开数据已经给了提示：custom device copy kernel 在小块 microbenchmark 里可能好看，但 end-to-end 会干扰模型计算；当 KV layout 调整成 0.5 到 2MB 物理块以后，DMA-style copy 更像一条宽河，稳定、低干扰、容易 overlap。([vLLM][1])

**3. 值得把 mapped/random access 放到 worker-local lifecycle 里。**
mapped-host gather 的 host registration、mapped device pointer、stream synchronization、unregister 必须由 worker 本地闭环。你们 RFC 的 Phase 5/6 顺序，也就是先 HostMappingRegistry lifecycle，继续 copy path，再开启 mapped-host device gather/read-write backend，是正确的防爆顺序。

### 不值得放弃的部分

**1. 不应该放弃 baseline copy path。**
RFC 现在已经规定 transfer engine 同时有 baseline copy path 和 mapped-host gather path，mapped gather 只在 capability/layout/dtype 合法时启用，否则 fallback 到 copy path。这个必须保留，而且我会建议长期保留，不只是 migration 阶段。

**2. 不应该放弃 bulk KV transfer 优化。**
对 full-prefix reload、active swap reload、连续大块 load/save、P/D disaggregation、RDMA 跨节点 transfer，bulk transfer 是正道。Mooncake、Dynamo、LMCache 都证明了高性能 KV movement 是生产 serving 的核心能力，不是旧时代遗物。([NVIDIA Docs][4])

**3. 不应该指望 random DRAM access 在所有场景赢。**
如果请求要加载的是完整 prefix 或完整 active swap state，random host access 很可能只是把一个本来可以 DMA 顺滑搬完的大块，拆成许多远端 load。除非硬件互联接近 Grace Hopper 这种 coherent high-bandwidth 形态，或者 selected KV fraction 很小，否则它会输给 bulk copy。([NVIDIA Developer][7])

---

## 一个更工程化的判断公式

可以把选择写成这个简化模型：

```text
copy_cost   = full_bytes / BW_bulk_copy_eff + copy_overhead
gather_cost = selected_bytes / BW_random_host_eff + gather_overhead + device_interference
```

mapped/random access 应该只在下面条件成立时启用：

```text
gather_cost < copy_cost
```

令：

```text
f = selected_bytes / full_bytes
```

那么直觉阈值是：

```text
f < BW_random_host_eff / BW_bulk_copy_eff
```

如果随机 host DRAM effective bandwidth 只有 bulk copy 的 1/4，那么 selected KV 必须少于 25% 才大概率值得；如果 random access 还能和 compute 很好 overlap，阈值可以放宽；如果它占用 device compute/AI core 或造成 stream interference，阈值要更严格。

这个阈值不能拍脑袋，必须由 Ascend/CANN 实测决定。我的建议是在 runtime 做 policy：

```python
if not mapped_gather_enabled:
    use_copy()

elif active_swap_reload:
    # correctness first; full reload usually copy unless proven otherwise
    use_copy_or_exact_mapped_with_strict_validation()

elif contiguous_fraction_high and bytes_large:
    use_copy()

elif selected_fraction <= measured_sparse_threshold and block_order_noncontiguous:
    use_mapped_gather()

else:
    use_copy()
```

这才是灵巧的刀法，不是把厨房炸成 benchmark 香炉。

---

## mapped/random DRAM access 真正的甜点区

我认为它最值得投的场景是这些：

**第一，partial prefix hit。**
比如 router 选中的 worker 只有一部分 prefix 或部分 blocks ready。copy path 可能要整理、拼接、搬完整块；mapped gather 可以按 block list 抓出来，尤其在 block order non-contiguous 时更自然。

**第二，稀疏 attention / top-k KV onload。**
InfiniGen 这类工作已经证明，long-context 下“只取关键 KV entries”能显著降低 CPU→GPU transfer overhead。mapped random access 很适合这类 selected KV entries，不适合“我要所有历史 token”的老式 full transfer。

**第三，避免 HBM 中间态膨胀。**
如果 device-side op 能直接从 host mapped KV gather 到最终 layout，或者进一步和 attention 结合，可能减少 HBM staging、GPU/NPU allocator fragmentation 和临时 buffer 压力。这一点对长上下文和高并发很有吸引力。

**第四，worker-local CPU swap。**
你们当前 RFC 选 worker-local pool 后，cross-worker 直接共享少了，但生命周期干净了。mapped random access 在这个模型里刚好能自然绑定到 worker/device/context，不需要 central SharedMemory 当内存皇帝。

---

## 它不适合替代的场景

**第一，NPU-to-CPU save/offload。**
save 通常是把刚生成的一批 KV 从 NPU block 写到 CPU arena。它天然是大块顺序写，bulk copy/DMA-style path 更合适。RFC 里也强调 save/offload correctness 很关键，scheduler 不能在 durable local CPU save ack 前 reclaim NPU blocks。

**第二，active swap reload。**
active swap 是 correctness-critical，不是性能缓存。这里通常要求 exact hit、完整 reload、严格 pinning 和 ack。random gather 可以作为实现 backend，但不应该为了“省一点搬运”牺牲可解释性。RFC 也已经把 ACTIVE_SWAP 和 PREFIX_CACHE 分开了，active swap 丢失必须 abort/retry/recompute from safe boundary，不能当普通 cache miss。

**第三，跨节点 P/D transfer。**
这不是 mapped host DRAM 能解决的问题。Dynamo/Mooncake/NIXL/RDMA 这类 transfer engine 还是主角。([NVIDIA Docs][4])

**第四，高命中、完整 prefix 复用。**
如果整个 prefix 都 ready，而且 layout 连续、block 大，copy path 很可能已经足够快。vLLM 的数据说明，layout 变大以后 DMA/end-to-end 表现非常强。([vLLM][1])

---

## 我会给 RFC 加一个明确决策：不是二选一，而是三层架构

建议把 RFC 里的措辞改成：

```text
We abandon the legacy device KV transfer route as the production architecture,
but we do not abandon bulk KV transfer as a transfer backend.

The production transfer engine supports:
1. baseline copy path: correctness baseline and full-block transfer backend;
2. bulk DMA-style optimized path: preferred for large contiguous load/save;
3. mapped-host gather/read-write path: preferred for sparse, partial, or
   non-contiguous host KV access when measured faster.
```

再加一条 invariant：

```text
Mapped-host random access is an optimization selected by policy, not a
replacement for baseline copy semantics.
```

还有一条 benchmark gate：

```text
Mapped gather may become default only for workload regions where it improves
p95 TTFT or transfer latency without regressing p95 ITL, throughput, or
fallback/error rate relative to copy path.
```

---

## 我建议的最小 benchmark 矩阵

为了回答“完全值得吗”，不要只跑一个 happy path。要把不同区域测出来：

| 场景                                         | 预期 winner            | 关键指标                                  |
| ------------------------------------------ | -------------------- | ------------------------------------- |
| full active swap reload, contiguous blocks | copy / DMA           | p95 load latency, ITL regression      |
| full prefix load, large MB blocks          | copy / DMA           | TTFT, transfer BW                     |
| partial prefix hit, scattered blocks       | mapped gather        | selected fraction threshold           |
| sparse top-k KV onload                     | mapped gather        | accuracy + TTFT                       |
| simultaneous save/load                     | copy likely          | bidirectional BW, stream interference |
| long-context Zipfian prefix reuse          | depends              | hit rate, p99 TTFT                    |
| worker restart/stale mapping               | neither, correctness | no silent corruption                  |
| MLA/TP non-contiguous layout               | mapped maybe         | layout transform overhead             |

核心要测两个阈值：

```text
1. selected_fraction_threshold
   selected_bytes / full_bytes 低于多少时 mapped gather 稳赢 copy。

2. interference_threshold
   mapped gather 占用 device 执行资源后，对 decode ITL 和 model forward 的影响。
```

没有这两个阈值，争论会变成玄学茶会。

---

## 最终判断

**为了 device-driven random DRAM KV access，放弃旧的 legacy device KV transfer 路线，是值得的。** 特别是如果旧路线指的是：

```text
global SharedMemory CPU pool
central metadata server authoritative CPU block ids
worker reconstruct shared tensor
device-side custom copy/gather built around that ownership model
```

这条路应该退场。它和 mapped host memory 的生命周期是拧巴的。

但如果“放弃 KV transfer”指的是连下面这些都不要：

```text
baseline copy path
bulk DMA-style host↔device load/save
RDMA/NIXL/Mooncake-style cross-worker KV movement
P/D disaggregation KV handoff
```

那就不值得。公开系统和论文都在说同一件事：**KV transfer 是 LLM serving 的核心燃料管道，只是旧管道的阀门和产权证写错了。**

我建议最终策略是：

```text
1. 放弃 legacy SharedMemory/global-pool production route。
2. 保留 worker-local copy path 作为 correctness baseline。
3. 保留/优化 bulk transfer，尤其用于 full-block、active swap、save/offload。
4. 将 device-driven mapped/random DRAM access 作为 sparse/partial/non-contiguous backend。
5. 用 measured policy 而不是 feature flag 信仰决定 copy vs mapped gather。
```

所以答案不是“完全值得”或“不值得”，而是更锋利一点：

> **放弃旧实现主线完全值得；放弃 bulk KV transfer 能力不值得；把 mapped/random host DRAM access 做成条件触发的高级刀刃，才是最稳的路线。**

[1]: https://vllm.ai/blog/2026-01-08-kv-offloading-connector "Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput | vLLM Blog"
[2]: https://lmcache.ai/tech_report.pdf "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"
[3]: https://arxiv.org/pdf/2407.00079 "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
[4]: https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving "Dynamo Disaggregation: Separating Prefill and Decode for Enhanced Performance | NVIDIA Dynamo Documentation"
[5]: https://arxiv.org/abs/2310.07240 "[2310.07240] CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving"
[6]: https://arxiv.org/html/2605.03375 "Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving"
[7]: https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ "Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing | NVIDIA Technical Blog"
