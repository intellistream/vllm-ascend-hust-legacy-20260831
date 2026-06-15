# SEW-Offload Expert 搬运分解与流水线优化

## 目标

本文记录 2026-06-15 对当前 `research` 分支 MoE expert miss 搬运路径的代码复核、真实 NPU 实验结果和下一步流水线优化方向。重点回答四个问题：

1. 当前缺失 expert 是算完再搬、还是边搬边算？
2. PCIe 搬运过程中，有多少时间更接近真实 payload movement，有多少是固定开销？
3. 当前“一次 miss expert load”到底是按 expert 单个搬运，还是 batch 搬运？
4. 下一步如何把计算和搬运组织成真正的 overlap pipeline？

## 结论摘要

- 当前 miss expert load 是 **按 expert 逐个同步搬运**，不是 batch 搬运。
- 当前 miss expert load 发生在当前层 `fused_experts()` 的 **Stage T**，位于 token dispatch 和 MLP compute 之前；因此是 **先搬后算**，没有和当前层计算重叠。
- 当前单 expert miss load 的真实实现是两次 `copy_`：
  - `slot.w13.copy_(bundle.w13)`
  - `slot.w2.copy_(bundle.w2)`
- 真实代码路径对应的 CPU 源更接近 **no-pin**，因为 `HostExpertStore` 当前使用的是 `.detach().cpu().clone()`，不是 pinned/UVA buffer。
- 对当前真实路径（no-pin）：
  - 事件总时间约 `1.034 ms/expert`
  - size sweep 拟合得到的 size-dependent payload 项约 `0.571 ms`，fixed+residual 约 `0.464 ms`
  - CANN timeline 中，当前两次 copy 的 `record_function` 窗口约 `0.898 ms/expert`，其中 `aclrtMemcpy` span 约 `0.778 ms`
- `single contiguous expert` 和 `batched contiguous experts` 对照显示，当前“两次 copy”路径确实有可观的拆分开销；`8-expert batch` 时每 expert 的窗口时间可降到约 `0.478 ms`。

## 当前代码路径

### 当前不是边搬边算

当前 MoE 执行顺序在 `vllm_ascend/ops/fused_moe/moe_comm_method.py` 中很明确：

1. `fused_experts()` 进入 Stage T。
2. 先执行 `_maybe_apply_moe_offload_plan(...)`。
3. 然后才进入 token dispatch。
4. 再进入 grouped MLP compute。

也就是说，当前 miss expert load 发生在：

- 当前层 dispatch 之前；
- 当前层 MLP compute 之前；
- 没有 dedicated transfer stream + compute stream overlap；
- 没有上一层/下一步提前发起的 async prefetch。

因此当前语义不是“算完再搬”，也不是“边搬边算”，而是：

> 用到这个 expert 时，先同步搬完，再开始当前层后续 dispatch 和计算。

### 当前不是 batch expert 搬运

`MoeOffloadRuntime.prepare_fixed_slot_plan()` 会遍历当前层本轮 `unique_active_experts`。每个 miss expert 都会单独走一次：

1. `slot_bank.allocate_for(...)`
2. `host_store.get(...)`
3. `transfer_engine.load_sync(...)`

`TransferEngine.load_sync(...)` 又会做两次同步 copy：

- `slot.w13.copy_(bundle.w13)`
- `slot.w2.copy_(bundle.w2)`

所以当前路径是：

- 粒度：`per miss expert`
- 次数：`2 * num_miss_experts` 个 `copy_`
- 模式：`sync`

不是“多个 miss expert 打包后统一搬运”。

## 证据边界

本次证据分三层，精度不同：

1. **代码语义证据**
   - 来自 `prepare_fixed_slot_plan()`、`load_sync()`、`fused_experts()` 的调用顺序。
   - 用来判断“先搬后算 / 是否 batch / 是否 overlap”。

2. **CANN runtime timeline**
   - 来自 `trace_view.json` / `ascend_pytorch_profiler.db` 中的 `AscendCL@aclrtMemcpy`。
   - 这是最硬的 timeline 证据，能给出每次 runtime memcpy 的 start/end。
   - 但它是 **runtime memcpy span**，不是逐 DMA descriptor 的裸硬件 wire time。

3. **PCIe 链路采样**
   - 来自 `pcie.csv` / `PCIE` 表。
   - 这是硬件链路计数器，但它是 **采样值**，不是逐 copy 精确归因。
   - 当前导出频率上限约为 `50 Hz`，因此适合看长窗口平均/峰值，不适合当单次 9 MiB copy 的精确起止点。

另外需要特别说明：

- Ascend NPU 当前 **不支持真正 UVA**；仓库里的 `patch_uva.py` 也是 CPU buffer + NPU mirror wrapper。
- 本文中的 `--pin-memory` 只表示 PyTorch CPU allocator flag 的对照实验，**不能**解读成“Ascend 官方 pinned DMA path”。
- 因此，分析“当前实现”时应以 **no-pin** 结果为主。

## 实验设置

- 模型：`/data/shared-models/Qwen3-30B-A3B`
- expert 权重：
  - `w13 = 6 MiB`
  - `w2 = 3 MiB`
  - `total = 9 MiB/expert`
- 设备：NPU 1，`ASCEND_RT_VISIBLE_DEVICES=1`
- profiler：`torch_npu.profiler`，开启 `sys_interconnection=True`
- 产物：
  - `artifacts/sew_offload/transfer_breakdown/qwen3_30b_a3b_expert_npu1_cann_no_pin.json`
  - `artifacts/sew_offload/transfer_breakdown/qwen3_30b_a3b_expert_npu1_cann_pinned.json`

## 三种对照模式

### 1. `single_contiguous_expert`

策略：

- 把一个 expert 的总 payload 视为一个连续 CPU buffer。
- 目标端也是一个连续 NPU buffer。
- 每轮只做一次 `dst.copy_(src)`。

作用：

- 近似回答“如果一个 expert 能用一次大块 copy 搬完，代价是多少？”
- 它不是当前实现，只是单 expert 一次性连续搬运的对照上界。

### 2. `two_tensor_current`

策略：

- 严格模拟当前真实路径。
- 每轮有两个 CPU tensor：`w13` 和 `w2`。
- 目标端对应 slot 的两个 NPU tensor。
- 每轮做两次 `copy_`：
  - `dst_w13.copy_(src_w13)`
  - `dst_w2.copy_(src_w2)`

作用：

- 这是当前 `TransferEngine.load_sync()` 的真实行为。
- 后文分析“当前实现的开销”都应以它为准。

### 3. `batched_contiguous_experts`

策略：

- 把多个 expert payload 打包成一个连续大 buffer。
- 本次实验默认 `batch_experts=8`，总大小 `72 MiB/batch`。
- 每轮只做一次大 `copy_`。

作用：

- 近似回答“如果未来能把多个 miss expert 打包 batch 搬运，开销还能降多少？”
- 这不是当前实现，但它能给 batch pipeline 一个明确的收益上界。

## 实验结果

### A. 当前真实路径的 size sweep 拟合分解（以 no-pin 为主）

当前真实路径对应 `two_tensor_current`，但这里的分解来自另外一层证据：

- 对多个 size factor 做 sweep；
- 拟合 `time_ms = fixed_ms + bytes * slope_ms_per_byte`；
- 再把 9 MiB expert 代回去。

no-pin 结果：

| 指标 | 数值 |
| --- | ---: |
| `expert_event_ms` | `1.034 ms` |
| `payload_movement_ms_from_fit` | `0.571 ms` |
| `fixed_plus_residual_ms` | `0.464 ms` |
| payload 占比 | `55.2%` |
| fixed+residual 占比 | `44.8%` |

这组分解说明：

- 如果只看 size-dependent 部分，当前单 expert miss load 里，真正随字节数线性增长的部分只有大约一半；
- 另一半更像固定开销、driver/runtime 开销、同步残差或 staging 残差。

### B. CANN runtime timeline 分解（当前真实路径）

no-pin 的 `two_tensor_current`：

| 指标 | 每 expert |
| --- | ---: |
| record window | `0.898 ms` |
| `aclrtMemcpy` 总和 | `0.778 ms` |
| `aclrtSynchronizeStream` | `0.030 ms` |
| host other | `0.089 ms` |
| `aclrtMemcpy` 占比 | `86.7%` |
| sync 占比 | `3.4%` |
| host other 占比 | `9.9%` |
| `aclrtMemcpy` 平均每次调用 | `0.389 ms` |

这个分解和上面的 fit 分解看起来不同，但它们并不矛盾：

- fit 分解切的是：`size-dependent payload` vs `fixed/residual`
- profiler 分解切的是：`API window 内部` 的 `aclrtMemcpy / sync / host-other`

因此：

- fit 里的 `fixed+residual 0.464 ms`，不一定全部落在 `aclrtMemcpy` 之外；
- 它很可能有相当一部分 **就包含在 `aclrtMemcpy` span 内部**。

这也是为什么：

- timeline 里 `aclrtMemcpy` 看起来占了 `86.7%`
- 但 fit 仍然给出 `44.8%` 的 fixed/residual

更准确的解释是：

> 当前 runtime memcpy span 很长，但这段 span 内部不全是“纯 PCIe wire payload movement”；其中包含了相当多 size-insensitive 的 runtime/driver/staging 成分。

### C. 三种模式对照（no-pin，最接近当前实现）

| 模式 | record window / expert | `aclrtMemcpy` / expert | memcpy 带宽 | window 带宽 |
| --- | ---: | ---: | ---: | ---: |
| `single_contiguous_expert` | `0.770 ms` | `0.699 ms` | `13.51 GB/s` | `12.25 GB/s` |
| `two_tensor_current` | `0.898 ms` | `0.778 ms` | `12.12 GB/s` | `10.51 GB/s` |
| `batched_contiguous_experts` | `0.478 ms` | `0.452 ms` | `20.87 GB/s` | `19.76 GB/s` |

可直接得到三个观察：

1. 当前“两次 copy”比“单次连续 copy”更慢。
2. 当前“两次 copy”不仅多了 API 次数，也拉低了带宽。
3. 一旦把多个 expert 打成 batch，大块搬运的收益非常明显。

### D. 当前“两次 copy”相比“一次大 copy”的代价

no-pin：

- `two_tensor_current` vs `single_contiguous_expert`
  - record window 多约 `0.127 ms/expert`
  - `aclrtMemcpy` span 多约 `0.080 ms/expert`

这说明当前把 9 MiB expert 切成 `6 MiB + 3 MiB` 两次 copy，本身就带来了不小的拆分成本。

### E. 链路采样

全 profiler 窗口的 `Rx_cpl_avg(MB/s)`：

| 模式集 | sampled avg | sampled max |
| --- | ---: | ---: |
| no-pin | `15.29 GB/s` | `15.77 GB/s` |
| pinned control | `16.21 GB/s` | `16.73 GB/s` |

注意：

- 这是长窗口链路采样；
- 不是单 expert copy 的逐次精确 DMA 带宽；
- 但它和 batch 模式里更高的 `record_window_bandwidth_gbps` 一致，说明更大块的 copy 确实更接近链路上限。

## 问题分析

### 1. 当前缺失 expert 是“先搬后算”，不是 overlap

当前顺序是：

1. 识别 active experts
2. 对 miss experts 逐个 `load_sync`
3. miss 全部到齐后才进入 dispatch / grouped MLP

所以当前 exposed stall 基本就是：

`T_exposed ~= T_all_miss_transfer - T_already_hidden`

而当前实现里几乎没有显式隐藏项，因为根本没有把 transfer 放到独立 stream，也没有让 hit experts 先算。

### 2. 当前按 expert、按 tensor 粒度拆得太碎

当前 miss expert load 的调用数是：

`2 * num_miss_experts`

也就是每个 miss expert 至少两次 `copy_`。这会带来：

- 更多 runtime launch / runtime bookkeeping
- 更差的大块带宽利用率
- 更难形成有效 overlap window

实验已经说明：

- 单 expert 一次连续 copy 优于当前两次 copy
- 多 expert 一次 batch copy 又明显优于单 expert copy

### 3. fit 与 profiler 共同指向“固定开销不可忽视”

当前真实路径上有两个重要现象：

1. fit 分解里 fixed+residual 接近一半
2. timeline 里 `aclrtMemcpy` 占主导，但它不是纯 wire time

这说明优化不能只盯住“PCIe 线速”：

- 需要减少 copy 次数
- 需要减少每次 copy 的 runtime 固定成本
- 需要把这些成本藏到 compute 下面

## 流水线优化方向

### 方向 1：把 `load_sync` 改成 `load_async`

目标：

- 为 miss expert 建 dedicated transfer stream
- 每个 miss slot 绑定一个 ready event
- `prepare_fixed_slot_plan()` 不再同步等 miss 全到齐

需要的最小改动：

1. `TransferEngine.load_sync()` 增加 `load_async(...)`
2. runtime 返回：
   - hit experts
   - miss experts
   - 每个 miss expert 对应的 ready event
3. compute stream 只在真正执行 miss phase 前等待对应 event

这是从“同步搬运”走向“可 overlap 搬运”的第一步。

### 方向 2：把 D.11 phase split 真正用于 hit-first overlap

当前 D.11 只是语义原型；真正要用来隐藏搬运时间，需要把 phase 变成：

1. `hit phase` 先算
2. miss expert 在 transfer stream 继续搬
3. `miss phase` 等对应 ready event 后再算

这样 exposed stall 会变成：

`T_stall = max(0, T_miss_transfer - T_hit_phase_compute)`

如果 hit phase 足够长，就能把一部分 miss load 藏掉。

### 方向 3：把 miss 搬运从 `2 * num_miss` 次 copy 降成少量大 copy

这是本轮实验最直接支持的方向。

现实可行的版本有两种：

1. **按 tensor 类型做 batch**
   - 把所有 miss 的 `w13` 按 slot 顺序排成一个大块，一次 copy
   - 把所有 miss 的 `w2` 按 slot 顺序排成一个大块，再一次 copy
   - 调用数从 `2 * num_miss` 降到 `2`

2. **packed expert layout**
   - host store 和 slot bank 都维护 packed expert buffer
   - 每个 expert 或每个 miss batch 只做一次 packed copy
   - 然后由 view/slice 暴露给 backend

从实现复杂度看，第一种更接近现有代码结构，因为当前 backend 本来就使用分开的 `w1/w2`。

### 方向 4：slot 分配应服务 batch copy，而不是只服务 LRU

如果 miss experts 被分配到离散 slot，batch copy 很难形成连续目标地址。

因此后续 slot 分配策略应额外考虑：

- 同一轮 miss 尽量分配连续 slot
- 同层 miss 的 `w13/w2` 尽量形成连续 batch window

这要求 slot allocator 不再只是“找一个空位/LRU victim”，而要开始为搬运形状服务。

### 方向 5：不要把优化建立在 UVA/pinned 假设上

由于 Ascend NPU 当前不提供真正 UVA，这条线应避免依赖：

- CPU tensor 可被设备直接寻址
- `pin_memory=True` 等价于 CUDA 式 pinned DMA

更稳妥的方向是：

- 继续把 host store 视作普通 CPU storage
- 优先优化 batch、async、slot layout、phase overlap
- 再单独评估是否存在官方支持的 host registration / faster staging 路径

## 建议的 MVP-E 顺序

### E1. async miss transfer

- 新增 `load_async`
- transfer stream + ready event
- 不改 token 语义

### E2. hit-first phased execution

- D.11 phase split 接管 hit/miss 编排
- hit phase 先算
- miss phase 按 event 等待

### E3. batched miss copy

- 先做“按 tensor 类型 batch”版本
- 后续再评估 packed expert layout

### E4. overlap 指标

新增三类指标：

- `miss_transfer_ms`
- `hit_phase_compute_ms`
- `exposed_stall_ms = max(0, miss_transfer_ms - overlap_hidden_ms)`

以及：

- `copies_per_miss_batch`
- `bytes_per_copy`
- `batch_transfer_bandwidth_gbps`

### E5. correctness 门禁

所有 overlap 优化都必须继续满足：

- 不改 router/top-k
- 不 drop token
- strict token-id compare 通过
- resident/full-weight path 与 slot path 均 fail-closed

## 当前结论对论文叙事的意义

这轮实验把问题定义从“PCIe 慢”进一步收窄成了更具体的系统问题：

1. 当前 miss load 发生得太晚：真正用到 expert 时才同步搬。
2. 当前 miss load 太碎：按 expert、按 tensor 两次 copy。
3. 当前没有 overlap：搬运和计算完全串行。
4. 当前 runtime memcpy span 内部仍有大量 size-insensitive 成分，单纯追线速不够。

因此后续 SEW-Offload 的系统贡献应更明确地表述为：

> 在不改变 dynamic-count MoE 语义的前提下，把 miss expert load 从“逐 expert、同步、串行”重构为“batch-aware、async、hit-first phased overlap”的 expert-weight pipeline。

