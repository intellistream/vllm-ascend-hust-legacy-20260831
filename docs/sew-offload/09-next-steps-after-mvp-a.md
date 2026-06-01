# SEW-Offload MVP-A 后续实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已经完成 trace-only MVP-A 的基础上，逐步把 Ascend MoE offloading 从“可观测”推进到“可模拟、可正确执行、可隐藏传输开销”。

**Architecture:** 下一阶段不直接跳到异步 prefetch。先把 routed expert trace 变成可复现实验输入，再用 simulator 定 slot budget 与 replacement policy，随后实现 fixed HBM expert slots 的同步 correctness prototype，最后引入 NPU stream/event 异步加载和 hit-first phased execution。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, vLLM Ascend MoE fused path, pytest, JSONL trace artifacts.

---

## 0. 当前状态基线

MVP-A 已完成：

- `vllm_ascend/moe_offload/config.py`
- `vllm_ascend/moe_offload/trace_collector.py`
- `vllm_ascend/moe_offload/runtime.py`
- `vllm_ascend/envs.py`
- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `tests/ut/moe_offload/`

当前不变量必须继续保持：

- 默认关闭。
- 不修改 router、top-k、gate weights。
- 不 drop token/expert。
- 不让 CPU tensor 进入 `npu_grouped_matmul`。
- scheduler、model runner 主路径暂不改。
- 新增环境变量必须集中在 `vllm_ascend/envs.py`。

## 1. 里程碑顺序

| 里程碑 | 名称 | 是否改变执行 | 目标 |
| --- | --- | --- | --- |
| MVP-B | Trace Export and Minimal Collection | 否 | 把 MVP-A 内存 trace 导出为 JSONL artifact，并跑通 Qwen3-30B-A3B smoke trace |
| MVP-C | Offline Slot Simulator | 否 | 离线评估 slot budget、replacement、miss、bytes、predicted stall |
| MVP-D | Fixed Slot Correctness | 是 | 实现 host expert store、HBM slot bank、同步 miss load，保证 grouped MoE 只消费 NPU slot tensors |
| MVP-E | Async Transfer and Cost Metrics | 是 | 使用 NPU load stream/event，记录 host-to-HBM copy time、wait time、exposed stall |
| MVP-F | Hit-First Phased Execution | 是 | ready experts 先算、miss-ready experts 后算，保持输出语义等价 |
| MVP-G | Ascend Static Window | 是 | 对 slot layout 和 phase shape 做 bucket，使 ACLGraph/NPUGraph 更容易 replay |

## 2. MVP-B: Trace Export and Minimal Collection

### 目标

把当前内存中的 `TraceCollector` 记录导出为稳定 JSONL 格式，并用真实模型 smoke workload 采集一份最小 trace artifact。

### 文件计划

- Modify: `vllm_ascend/moe_offload/trace_collector.py`
- Modify: `vllm_ascend/moe_offload/runtime.py`
- Create: `tools/sew_offload/collect_moe_trace.py`
- Create: `tests/ut/moe_offload/test_trace_export.py`
- Update: `docs/sew-offload/04-reproduction.md`

### 数据格式

每行一个 JSON object：

```json
{"layer_id": 3, "step_id": 11, "mode": "decode", "num_tokens": 2, "top_k": 8, "num_experts": 128, "active_experts": [0, 7, 31], "expert_token_counts": {"0": 1, "7": 3, "31": 2}}
```

### 任务清单

- [ ] 写 `TraceCollector.to_jsonl()` 失败测试，验证多条记录导出为换行分隔 JSON。
- [ ] 实现 `TraceCollector.to_jsonl()` 和 `TraceCollector.write_jsonl(path)`。
- [ ] 写 `MoeOffloadRuntime.export_trace(path)` 失败测试，验证 runtime 能导出当前 collector。
- [ ] 实现 `MoeOffloadRuntime.export_trace(path)`。
- [ ] 新增 `tools/sew_offload/collect_moe_trace.py`，支持参数：
  - `--model /data/shared-models/Qwen3-30B-A3B`
  - `--output artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl`
  - `--max-num-seqs 1`
  - `--max-model-len 512`
  - `--device npu`
- [ ] 使用 `benchmark_config.yaml` 的 synthetic smoke request 跑一次 trace collection。
- [ ] 在 `04-reproduction.md` 写入 trace-only 采集命令和 artifact 位置。

### 验证命令

```bash
${PYTHON:-python} -m pytest -q \
  tests/ut/moe_offload/test_trace_export.py \
  tests/ut/moe_offload/test_trace_collector.py \
  tests/ut/moe_offload/test_runtime_trace_only.py
```

### 完成标准

- JSONL trace 可以被 Python 标准库 `json.loads` 逐行解析。
- 默认关闭时不写 trace。
- trace-only 开启时不改变 `topk_ids/topk_weights` 对象身份。
- smoke artifact 至少包含多层 MoE 的 routed expert records。

## 3. MVP-C: Offline Slot Simulator

### 目标

在不触碰 NPU 执行路径的前提下，回答三个问题：

- 给定 `num_slots`，miss rate 是多少。
- 给定 expert size，host-to-HBM bytes 是多少。
- 哪些 layer/step 有 hit-first phase opportunity。

### 文件计划

- Create: `vllm_ascend/moe_offload/expert_key.py`
- Create: `vllm_ascend/moe_offload/slot_simulator.py`
- Create: `vllm_ascend/moe_offload/policy.py`
- Create: `tools/sew_offload/simulate_expert_slots.py`
- Create: `tests/ut/moe_offload/test_slot_simulator.py`
- Create: `tests/ut/moe_offload/test_policy.py`

### 初始策略

先实现两种 policy：

```text
lru
sticky_layer_lru
```

`sticky_layer_lru` 优先保留下一 decode step 同层重复出现的 experts；没有 locality 信息时退化为 LRU。

### 任务清单

- [ ] 定义 `ExpertKey(layer_id: int, expert_id: int)` dataclass，作为 slot table key。
- [ ] 写 simulator RED 测试：2 个 slots、trace 中 3 个 experts，验证 hit/miss/eviction 计数。
- [ ] 实现 `SlotSimulator.replay(records, num_slots, policy)`。
- [ ] 写 expert size RED 测试：给定 `w13_bytes + w2_bytes`，验证 host-to-HBM bytes 只在 miss 时累计。
- [ ] 实现 `ExpertSizeTable`，默认按 Qwen3-30B-A3B MoE expert 形状估算。
- [ ] 写 CLI 测试，验证 `simulate_expert_slots.py --trace trace.jsonl --num-slots 32` 输出 JSON summary。
- [ ] 输出指标：
  - `total_records`
  - `hit_count`
  - `miss_count`
  - `eviction_count`
  - `host_to_hbm_bytes`
  - `estimated_load_ms`
  - `phase_opportunity_count`
- [ ] 用 MVP-B smoke trace 跑 `num_slots=8/16/32/64` 四组结果。

### 验证命令

```bash
${PYTHON:-python} -m pytest -q \
  tests/ut/moe_offload/test_slot_simulator.py \
  tests/ut/moe_offload/test_policy.py
```

### 完成标准

- simulator 不依赖 torch_npu，可在 CPU-only 单测中跑。
- 同一 trace、同一 policy、同一 slot budget 输出确定性结果。
- 结果能解释后续 fixed slot 需要的最小 slot budget。

## 4. MVP-D: Fixed Slot Correctness

### 目标

实现第一个会改变执行路径的版本：所有 active experts 在 compute 前必须位于固定 NPU slots；miss 使用同步 load，先追求正确性，不追求 overlap。

### 文件计划

- Create: `vllm_ascend/moe_offload/host_store.py`
- Create: `vllm_ascend/moe_offload/slot_bank.py`
- Create: `vllm_ascend/moe_offload/transfer_engine.py`
- Create: `vllm_ascend/moe_offload/layout.py`
- Modify: `vllm_ascend/moe_offload/runtime.py`
- Modify: `vllm_ascend/ops/fused_moe/fused_moe.py`
- Create: `tests/ut/moe_offload/test_host_store.py`
- Create: `tests/ut/moe_offload/test_slot_bank.py`
- Create: `tests/ut/moe_offload/test_transfer_engine.py`

### 关键接口

```python
@dataclass(frozen=True)
class ExpertWeightBundle:
    layer_id: int
    expert_id: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
```

```python
class ExpertSlotBank:
    def ensure_ready(self, layer_id: int, expert_id: int) -> int:
        """Return ready slot_id for the requested expert."""
```

### 执行边界

仍然只在 `AscendUnquantizedFusedMoEMethod.apply()` 附近接入：

```text
select_experts -> moe_offload_runtime.prepare_weights -> build_fused_experts_input -> fused_experts
```

### 任务清单

- [ ] 写 `HostExpertStore.register_layer(layer)` RED 测试，验证能按 `(layer_id, expert_id)` 取出 post-processed `w13/w2`。
- [ ] 实现 `HostExpertStore`，首版只支持 unquantized whole expert。
- [ ] 写 `ExpertSlotBank` RED 测试，验证 slot 地址稳定、state transition、version 不回退。
- [ ] 实现 fixed slot allocation 和 conservative LRU replacement。
- [ ] 写 `LayoutValidator` RED 测试，验证 shape/dtype/device/stride 不匹配会拒绝执行。
- [ ] 实现同步 `TransferEngine.load_sync(bundle, slot)`，使用 `copy_` 把 host bundle 写入 target NPU slot tensors。
- [ ] 在 runtime 中新增非 trace-only guard：只有 `enabled=1`、`trace_only=0`、`num_slots>0` 时才进入 fixed slot path。
- [ ] 在 `fused_moe.py` 中只替换传给 grouped backend 的 weight tensors，不改 `topk_ids/topk_weights`。
- [ ] 在单层 mock MoE 上比较 no-offload 与 fixed-slot output。
- [ ] 在 Qwen3-30B-A3B smoke prompt 上跑 correctness，记录是否能越过当前 native prefetch 的 CPU/NPU tensor mixing 失败点。

### 验证命令

```bash
${PYTHON:-python} -m pytest -q \
  tests/ut/moe_offload/test_host_store.py \
  tests/ut/moe_offload/test_slot_bank.py \
  tests/ut/moe_offload/test_transfer_engine.py \
  tests/ut/moe_offload/test_runtime_trace_only.py
```

### 完成标准

- grouped MoE backend 看到的 `w13/w2` 必须是 NPU tensors。
- slot tensor 地址在初始化后保持稳定。
- 输出与 no-offload 在容差内一致。
- 允许慢，不能错。

### 设计反思补充

MVP-D 不能简单地把 `num_slots` 维度的 slot tensor 替换原始 `num_experts`
维度的 `w13/w2`，同时保持原始 `topk_ids`。现有 grouped MoE backend 会根据
expert id 与 group list 访问权重；如果权重维度变成 slot id，而 `topk_ids`
仍是原 expert id，语义会错。

因此 fixed-slot correctness 的真正接入必须满足二者之一：

1. 构造 slot-backed full expert tensor，仍保持第 0 维可按原 expert id 索引。
2. 使用现有 `log2phy` 机制或等价 remap，把 active expert id 映射到 slot id，
   并确保 token dispatch、group list、combine 仍保持原 token 语义。

当前实现应先完成 HostExpertStore、SlotBank、LayoutValidator、TransferEngine
这些底座，并让 runtime 的 non-trace fixed-slot path fail closed；在完成
expert-to-slot remap 设计前，不应把 slot weights 接入 `fused_moe.py` 主执行路径。

### 设计反思补充 2：log2phy 之外还需要 physical expert count

进一步核对 `TokenDispatcherWithAllGather`、`TokenDispatcherWithAll2AllV` 和
`TokenDispatcherWithMC2` 后确认：`log2phy` 只能把 `topk_ids` 从 logical expert id
转换为 physical id，但 dispatcher 仍可能按原 MoE config 的 expert 数生成
`expert_tokens/group_list`。如果 `w13/w2` 已变为 `[num_slots, ...]`，而
`group_list` 仍按 logical experts 组织，grouped matmul 仍会出现语义错配。

因此 fixed-slot 主路径接入必须同时满足：

1. slot-backed `w13/w2` 的第 0 维为 `num_slots`；
2. `topk_ids` 经 `logical_to_physical` remap；
3. token dispatch 的 `expert_num/group_list` 使用 `num_slots` 这个 physical expert count。

当前已新增 `PreparedSlotWeights.physical_expert_count` 与
`MoERoutingParams.physical_expert_count`，但只应先用于单卡 AllGather、无
`expert_map`、无 redundant experts 的窄路径。All2All、MC2、EP/EPLB 与 quant
路径需要后续单独验证，不应在 MVP-D correctness 原型中顺手打开。

### 当前 MVP-D.3 接入范围

当前 fixed-slot apply wiring 只允许：

- `MoECommType.ALLGATHER`；
- unquantized whole-expert `w13/w2`；
- 无 `expert_map`；
- 无 redundant experts；
- 无 expert bias；
- 无 force load balance；
- 无 zero-expert path。

其它路径必须在 backend 前 fail closed。这个限制是 correctness 约束：MC2、All2All、
EP/EPLB、quant、bias、zero-expert 都会引入额外 metadata、权重量或输出合并语义，
不能仅凭 `log2phy` 与 `physical_expert_count` 推断已经语义等价。

## 5. MVP-E: Async Transfer and Metrics

### 目标

把 MVP-D 的同步 miss load 改成可度量的异步 load。首版不做复杂 prediction，只要能发起 load、记录 event、在需要前 wait。

### 文件计划

- Modify: `vllm_ascend/moe_offload/transfer_engine.py`
- Create: `vllm_ascend/moe_offload/cost_model.py`
- Create: `vllm_ascend/moe_offload/metrics.py`
- Create: `tests/ut/moe_offload/test_cost_model.py`
- Create: `tests/ut/moe_offload/test_metrics.py`

### 任务清单

- [ ] 定义 `LoadTicket`，包含 `expert_key`、`slot_id`、`start_ns`、`end_ns`、`bytes`、`done_event`。
- [ ] 使用 `torch.npu.Stream` 创建独立 load stream。
- [ ] 实现 `load_async(bundle, slot) -> LoadTicket`。
- [ ] 实现 `wait(ticket)`，返回 wait time。
- [ ] 记录 `load_time_ms`、`wait_time_ms`、`exposed_stall_ms`。
- [ ] 将 metrics 输出到 runtime snapshot，后续 benchmark runner 可读取。

### 完成标准

- 异步路径有同步 fallback。
- 任何 slot 覆盖前必须确认没有 in-flight compute user。
- 所有新指标能被单测构造并导出。

## 6. MVP-F: Hit-First Phased Execution

### 目标

当当前层 active experts 中一部分 ready、一部分 miss 时，先执行 ready experts，让 miss load 有机会并行完成。

### 文件计划

- Create: `vllm_ascend/moe_offload/phase_scheduler.py`
- Modify: `vllm_ascend/moe_offload/runtime.py`
- Modify: `vllm_ascend/ops/fused_moe/fused_moe.py`
- Create: `tests/ut/moe_offload/test_phase_scheduler.py`

### 任务清单

- [ ] 定义 `PhasePlan`：`phase_id`、`expert_ids`、`reason`、`requires_wait`。
- [ ] 实现默认 two-phase plan：hit experts first，miss-ready experts second。
- [ ] 加入阈值：如果 phase split overhead 预计大于 overlap，fallback single phase。
- [ ] 验证 phase split 后 token output 写回位置与 single phase 等价。
- [ ] 记录 `phase_count`、`overlap_time_ms`、`exposed_stall_ms`。

### 完成标准

- 不引入 per-expert tiny kernels。
- 默认最多 2 个 phases。
- 不改变 token output 顺序。

## 7. MVP-G: Ascend Static Window

### 目标

在 correctness 和 async path 都成立后，再做 Ascend-specific graph/static window 优化。

### 文件计划

- Create: `vllm_ascend/moe_offload/static_window.py`
- Modify: `vllm_ascend/moe_offload/phase_scheduler.py`
- Create: `tests/ut/moe_offload/test_static_window.py`

### 任务清单

- [ ] 定义 phase size buckets。
- [ ] 给每个 bucket 生成 graph replay eligibility metadata。
- [ ] 检查 slot layout signature 是否稳定。
- [ ] 对比 eager fixed slot 与 bucketed static window 的 launch overhead。

### 完成标准

- graph/static window 是优化层，不是 correctness 依赖。
- 不为每个 expert 生成 graph fragment。

## 8. 近期两周建议排期

| 时间 | 目标 |
| --- | --- |
| Day 1-2 | MVP-B trace export、JSONL schema、collector CLI |
| Day 3-4 | Qwen3-30B-A3B smoke trace 采集，整理 artifact |
| Day 5-7 | MVP-C simulator、LRU/sticky policy、slot budget sweep |
| Day 8-10 | MVP-D HostExpertStore、SlotBank、LayoutValidator UT |
| Day 11-14 | 同步 fixed slot prototype 接入 mock/smoke correctness |

## 9. 质量门禁

每个 MVP 合并前必须满足：

- `pytest tests/ut/moe_offload -q` 通过。
- `git diff --check` 通过。
- 默认关闭路径行为不变。
- 新增 NPU 相关路径至少有 mock UT；真实 NPU 路径要有 smoke artifact。
- 任何 CPU tensor 进入 grouped MoE backend 都视为 hard failure。

## 10. 当前最优下一步

立即执行 MVP-B，而不是直接做 fixed slot：

```text
MVP-B trace export -> MVP-C simulator -> MVP-D fixed slot correctness
```

理由：

- trace export 让后续策略评估可复现。
- simulator 能先回答 slot 数和 replacement 是否值得做。
- fixed slot 是第一个会改变执行的阶段，必须带着 trace 和 policy 证据进入。
