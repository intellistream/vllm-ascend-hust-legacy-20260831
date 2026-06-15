# SEW-Offload 项目计划

## 目标

在 `vllm-ascend-hust` 中以高内聚、低耦合、默认关闭的方式推进 Ascend 类 NPU 上的 MoE expert offloading 研究，并产出：

1. CCF-A 类会议论文：`paper/`
2. 仿 `moe_serving_report.tex` 风格的汇报 slide：`slide/`
3. 基于 vLLM Ascend 的低侵入实现、实验与复现材料。

## 当前阶段

阶段 15：**MVP-E**（async transfer + hit-first overlap + batched miss copy）
已完成问题复核和 expert transfer breakdown 实验，进入方案冻结与实现规划阶段。
当前已确认：

- miss expert load 在 `fused_experts()` 中发生于 Stage T，位于 token dispatch 和 MLP compute 之前；
- 当前路径按 miss expert 逐个同步搬运，每个 expert 两次 `copy_`，没有 transfer/compute overlap；
- no-pin 真实路径下，单 expert miss load 的 size sweep 拟合为
  `1.034 ms = 0.571 ms size-dependent payload + 0.464 ms fixed/residual`；
- CANN timeline 下，当前 `two_tensor_current` 窗口约 `0.898 ms/expert`，
  其中 `aclrtMemcpy` span 约 `0.778 ms`；
- `single_contiguous_expert` 与 `batched_contiguous_experts` 对照进一步表明：
  当前两次 copy 的拆分成本明确存在，而 batched contiguous copy 可把每 expert
  窗口时间降到约 `0.478 ms`。

因此下一阶段主线不是继续做语义脚手架，而是把 D.11 相位切分和异步搬运真正接成
overlap pipeline。

## 阶段 12：MVP-D.9 任务表

| 子任务 | 状态 | 说明 |
| --- | --- | --- |
| 规划三文件同步（差距表、d9_first） | complete | task_plan / findings / progress |
| Tiered residency config（默认关） | complete | `RESIDENT_LAYER_IDS`、`RELEASE_ORIGINAL_EXPERT_WEIGHTS` |
| 容量模型 + per-layer vs global | complete | `compare_slot_budget_models`、13.5GB budget 参考 |
| Partial release（opt-in + guard） | complete | `release_original_expert_weights_if_ready`、零元素占位 Parameter |
| 动态 count 路径复核 | complete | 确认常规 MoE 是 dropless dynamic count；固定 token capacity 不再作为默认主线 |
| D.9 分段耗时 + ledger 打点 | complete | runtime profiling + cross-process JSONL artifact |
| 小范围 release=1 NPU smoke | complete | 1 个 non-resident 层 release=1 通过 |
| no-offload vs release=1 对照 | complete | token id 一致；reported weight 56.9001GB vs 42.3454GB |
| 真实 trace 采集 | complete | `artifacts/sew_offload/traces/d9_trace_short_20260602/trace.jsonl`，6192 records |
| resident/slot/global pool sweep | complete | `sweep_summary.json`：per-layer 小 slots 在 prefill fail-closed；naive global LRU 不足 |
| prefill/resident-aware 策略设计 | complete | 离线 analyzer +真实 trace sweep；global 小池反例，per-layer decode slots 有效 |
| prefill/resident-aware runtime 原型 | superseded | 升级为阶段 13 MVP-D.10 dynamic-count layered runtime |

## 阶段 13：MVP-D.10 任务表

| 子任务 | 状态 | 说明 |
| --- | --- | --- |
| 下一步规划同步 | complete | 把 dynamic-count staged 主线写入三份 `.planning` 文件 |
| 已验证 plugin 路线映射 | complete | 用户提供过往成功方案：load_model 后 CPU expert store + per-layer NPU pool + patch `MoECommMethod.fused_experts` remap |
| Runtime decision contract | complete | `MoeOffloadDecisionPath` / `MoeOffloadPathDecision`：`full_weight_path`、`slot_cache_path`、`fail_closed` |
| fused_experts 边界内生化 | complete | `AscendUnquantizedFusedMoEMethod.apply()` 只传 offload metadata；`MoECommMethod.fused_experts` 内按 `topk_ids` 决策 full/slot/fail，并在 token dispatch 前准备 slot plan |
| Env/config 默认关闭 | complete | `VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=0` 默认关；`FANOUT_THRESHOLD` 默认 0（回退 `num_slots`） |
| Full-weight readiness guard | complete | high fan-out 且原始 full expert 权重已 release 时 fail closed |
| Slot-cache path 接入 | complete | low fan-out 复用现有 fixed-slot sync + `log2phy/physical_expert_count`，不改 token count |
| Decision observability | complete | profile JSONL 记录 active count、path、full/slot readiness、reason 和 ledger |
| UT 回归 | complete | 默认路径、low fanout slot path、high fanout full path、released fail-closed、smoke env/metrics |
| 小范围 NPU smoke | complete | NPU 6：1-token 与 1 prompt × 8-token no-offload strict token-id 对照通过；输出 throughput/TTFT/TPOT |
| post-downsink NPU smoke | complete | 2026-06-02 NPU 6 资源窗口恢复后补跑下沉版 smoke：1-token 与 8-token candidate 均 status ok，与 no-offload baseline strict token-id 对照全部通过（1-token `[353]`；8-token `[353,91957,9,0,358,2776,501,311]`）；首跑曾因 NPU 6 残留幽灵上下文偶发 engine init 失败，约 30s 自行释放后重跑成功 |
| D.10 复盘门禁 | complete | 下沉后 fused boundary path decision 在真实 NPU 上验证正确：短 prompt 全程走 `slot_cache_path`/`low_fanout_slot_cache_ready`；同步 slot path 性能仍显著慢于 no-offload（8-token candidate throughput 1.475 tok/s vs no-offload 5.512 tok/s），只作为 offloading 通路与 observability 闭环，不作为性能收益；correctness 门禁已过，可进入 D.11 phase split |

## 阶段 14：MVP-D.11 任务表

| 子任务 | 状态 | 说明 |
| --- | --- | --- |
| D.11 范围冻结 | complete | 只做 dispatch 后 phase split 语义原型；不做 async transfer、不做性能优化、不改 router/top-k/token count |
| Phase split contract | complete | `MoEPhasePlan` / `MoEPhase` / 回填 contract 已定义在 `vllm_ascend/moe_offload/phase_split.py`；`MoEPhasePlan` 包含 phases、hit/miss count、total_tokens、reason |
| Dispatch-output expert slicing | complete | `compute_expert_token_slices()` 支持 group_list_type=0/1；AllGather 窄路径，复杂通信 fail closed |
| Phase planner | complete | `plan_hit_miss_phases()` 基于同步 `slot_readiness` 字典做 hit/miss 切分；默认单 phase fallback；`max_phases=1` 强制单 phase |
| Partial MLP execution seam | complete | `_extract_phase_tokens()`、`_build_phase_group_list()`、`_slice_expert_weights()` 构造子 MLP 输入；`execute_phased_mlp()` 对每个 phase 调用 `_apply_mlp_fn` |
| Full-buffer scatter/gather | complete | `_scatter_phase_output()` 对各 phase 输出按 `token_slices` 回填到完整输出 buffer；单 phase fast-path 跳过 scatter |
| Equivalence UT | complete | 31 个 UT 通过：覆盖 type 0/1 slicing、hit/miss/all-hit/all-miss/空 expert 等价性、group_list 构建、权重切片、scatter、profile JSONL、空 phase fallback、乱序/缺失 slice fail-closed |
| Boundary integration guard | complete | `MoECommMethod._maybe_plan_phase_split()` + `fused_experts()` 集成；默认 `VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT=0` 关闭；只在 AllGather+unquantized+no bias+no EP 窄路径启用；否则 fail closed |
| Observability | complete | `PhaseSplitProfileEvent` 写入 profile JSONL 记录 phase plan、fail reason、layer_id；与现有 `VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` 复用同一 JSONL 流 |
| NPU semantic smoke | complete | 2026-06-05 NPU 2：1-token strict compare **status=ok**（baseline `[26288]` vs candidate `[26288]`）；8-token strict compare **status=ok**（baseline `[26288,102064,104949,9370,104034,20074,89161,102021]` vs candidate 完全一致）；throughput 1-token 2.16 tok/s（vs baseline 2.29）、8-token 4.71 tok/s（vs baseline 5.60） |
| D.11 复盘门禁 | complete | Python 多 phase 开销确认：phase_split=1 下每层每 token 触发 phase plan（96 事件/1-token，432 事件/8-token），8-token throughput 下降约 16%（4.71 vs 5.60 tok/s）。记录为语义脚手架；MVP-E async 前保留 single-phase fallback 作为 correctness 底座 |

## 阶段 15：MVP-E 任务表

| 子任务 | 状态 | 说明 |
| --- | --- | --- |
| transfer breakdown 结论冻结 | complete | 已形成 `docs/sew-offload/11-expert-transfer-breakdown-and-pipeline.md`：当前是先搬后算、逐 expert 两次 sync copy、无 overlap |
| 证据边界澄清 | complete | Ascend 当前不提供真正 UVA；`--pin-memory` 只作控制实验，当前实现分析以 no-pin 为准 |
| `load_async` 设计 | pending | `TransferEngine` 增加 dedicated transfer stream + ready event；runtime 返回 miss readiness，而不是同步等待 miss 全到齐 |
| hit-first overlap 接线 | pending | 把 D.11 phase split 从语义脚手架升级为真正的 hit phase / miss phase pipeline；计算 stream 只在 miss phase 前等待对应 ready event |
| batched miss copy 设计 | pending | 优先做按 tensor 类型的 batch copy（`w13` 一次、`w2` 一次）；后续再评估 packed expert layout |
| slot allocator for batch | pending | 研究让同轮 miss expert 尽量落到连续 slot，服务大块 copy，而不是只做 LRU victim 选择 |
| overlap observability | pending | 新增 `miss_transfer_ms`、`hit_phase_compute_ms`、`exposed_stall_ms`、`copies_per_miss_batch`、`bytes_per_copy` |
| NPU overlap smoke | pending | 单 expert miss / 多 expert miss / all-hit / all-miss 四类场景分别做 strict token-id compare + timeline artifact |
| batch-vs-sync 对照 | pending | 固定 num_slots 和 active expert fanout，比较 `2*num_miss` copy、2-copy batch、packed batch 三组 |

## 阶段清单

| 阶段 | 状态 | 目标产物 |
| --- | --- | --- |
| 0. 上下文建立 | complete | 仓库结构、MoE 代码边界、paper/slide 目录、工程规则 |
| 1. 研究问题冻结 | complete | RQ、假设链、贡献点、非目标 |
| 2. 论文蓝图 | in_progress | `paper/outline.md`、`paper/related_work_matrix.md`、LaTeX skeleton |
| 3. Slide 蓝图 | pending | `slide/sew_offload_report.tex` skeleton 与图表清单 |
| 4. Runtime 架构设计 | complete | 高内聚低耦合模块边界、配置开关、集成点 |
| 5. MVP-0/MVP-1 实现计划 | complete | routing trace、expert store、slot manager、fixed window |
| 6. 实验计划 | complete | baseline、workload、metrics、ablation、artifact layout |
| 6.5. 最小 Benchmark 协议 | complete | `docs/sew-offload/06-benchmark-design.md`、`docs/sew-offload/benchmark_config.yaml`：只固定模型、数据集、workload buckets、13.5GB offload budget、指标 |
| 7. 质量门禁 | in_progress | correctness、性能、citation、复现、默认关闭验证 |
| 8. 现有 offloading 实测 | complete | Qwen3-30B-A3B 单卡 baseline、UVA/offload-prefetch 日志、失败/瓶颈定位 |
| 9. Native offload benchmark pilot | complete | `tools/sew_offload/run_minimal_offload_benchmark.py`、native prefetch expert/all-param 失败证据、no-offload 三指标 sanity |
| 10. SEW runtime MVP | in_progress | MVP-A/B/C 完成；MVP-D fixed-slot sync correctness 最小闭环已通过真实 NPU smoke；MVP-D.11 phase split 语义原型代码/UT 已完成 |
| 11. MoE Offload 支持核实与总体架构 | complete | `docs/sew-offload/08-ascend-moe-offload-architecture.md`：不支持证据链、系统架构、控制/数据面图、路线图 |
| 12. Dynamic-count layered runtime | complete | MVP-D.10 path selector 与 fused_experts 下沉代码/UT 完成；post-downsink 1-token/8-token 真实 NPU smoke 与 strict compare 全部通过 |
| 13. Dispatch 后 phase split | complete | MVP-D.11 代码/UT/smoke 全部完成：1-token & 8-token strict compare **ok**；复盘确认 Python 多 phase 16% 开销，记录为语义脚手架 |
| 14. Expert transfer overlap pipeline | in_progress | 当前问题已量化：先搬后算、逐 expert 两次 sync copy、无 overlap；下一步进入 `load_async`、hit-first overlap、batched miss copy |

## 当前 MVP-A 状态

- 已实现 `vllm_ascend/moe_offload/` trace-only runtime、集中 env 配置和 `AscendUnquantizedFusedMoEMethod.apply()` 的低侵入 hook。
- MVP-A 严格不移动权重、不改 router/top-k、不改 dispatch/grouped matmul、不拆分 execution phase，只记录 routed expert working set。
- 已通过新 MoE offload UT、现有 env UT、语法检查、Markdown fence 检查和 `git diff --check`。
- `tests/ut/ops/test_fused_moe.py` 在当前环境收集阶段缺少 `pytest_mock`，未能作为额外既有测试运行；MVP-A 的 fused MoE hook 已由 `tests/ut/moe_offload/test_runtime_trace_only.py` 覆盖。

## 当前 MVP-B 状态

- 已实现 `TraceCollector.to_jsonl()`、`TraceCollector.write_jsonl(path)` 和 `MoeOffloadRuntime.export_trace(path)`。
- 已新增 `tools/sew_offload/collect_moe_trace.py`，支持准备 synthetic smoke manifest、加载 manifest、启用 trace-only、运行 vLLM 并导出 JSONL trace。
- 已更新 `docs/sew-offload/04-reproduction.md` 的 trace-only 复现命令。
- 已通过 MVP-B UT、py_compile、`git diff --check` 和 CLI `--prepare-only` smoke；真实 Qwen3-30B-A3B NPU trace collection 尚未运行。

## 当前 MVP-C 状态

- 已实现 `ExpertKey`、`LruPolicy`、`StickyLayerLruPolicy`、`SlotSimulator` 和 `ExpertSizeTable`。
- 已新增 `tools/sew_offload/simulate_expert_slots.py`，支持从 JSONL trace 输出 slot simulation JSON summary。
- 已更新 `docs/sew-offload/04-reproduction.md` 的 offline simulator 复现命令。
- 已通过 MVP-C UT、py_compile、`git diff --check` 和 simulator CLI sample。
- 下一步进入 MVP-D：HostExpertStore、ExpertSlotBank、LayoutValidator、同步 TransferEngine 与 fixed slot correctness。

## 当前 MVP-D 状态

- 已实现 fixed-slot 安全底座：`HostExpertStore`、`ExpertSlotBank`、`LayoutValidator`、`TransferEngine`。
- 已新增 expert-to-slot remap 安全计划：`ExpertSlotMapping`、`PreparedSlotWeights`、稳定 `[num_slots, ...]` backing tensors、runtime `prepare_fixed_slot_plan(...)`。
- 已新增 AllGather slot-dispatch metadata 契约：`MoERoutingParams.physical_expert_count`，避免 `log2phy` 已 remap 但 `group_list` 仍按 logical expert count 生成。
- 已完成单卡 AllGather、unquantized、无 EP/EPLB/bias 的窄路径主接入；其它通信/EP/EPLB/quant/bias 路径仍 fail closed。
- 已补默认路径回归测试：offload 关闭时仍传原始 `layer.w13_weight/layer.w2_weight`，`log2phy/physical_expert_count` 默认为 `None`，AllGather dispatcher 默认仍使用 local logical expert count。
- 已补 `PreparedSlotWeights.validate_backend_ready(...)`，进入 backend 前检查 slot tensor 设备类型和 `physical_expert_count == w1.shape[0] == w2.shape[0]`。
- 已新增 `tools/sew_offload/run_fixed_slot_smoke.py` 和复现命令，用于真实 Qwen3-30B-A3B fixed-slot sync smoke artifact。
- 已处理 vLLM profile/dummy run 的 `enable_force_load_balance` 冲突：fixed-slot profile 只在 dummy 路径中把 active experts 约束到 slot budget，并要求 `num_slots >= top_k`；真实请求不改 router/top-k 语义。
- 已补齐 Ascend `NPUModelRunner.load_model()` 中现有 weight `PrefetchOffloader.post_init()` 生命周期，避免 prefetch forward hook 在 profile run 中因 static buffer pool 未分配而失败；默认 NoopOffloader 路径为空操作。
- 已修正 `_torch_cuda_wrapper()` 的 Event 映射：退出 wrapper 后仍保持 `torch.cuda.Event = torch.npu.Event`，避免现有 prefetch 初始 onload 中 NPU stream 与 placeholder event 混用。
- 已修正 Ascend MoE runner internal-router gate 生命周期：`AscendMoERunner._forward_impl()` 对齐上游，在调用 layer forward 前执行 `self.gate(hidden_states)`，避免 Qwen3 把 hidden states 当 router logits 产生越界 expert id。
- 真实 NPU smoke 已通过：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_runnergate_slots8_inline_prefetch/summary.json` 状态 `ok`，1 个请求、1 个输出 token。
- 已新增 MVP-D.5 correctness 对照工具：`run_fixed_slot_smoke.py --mode no_offload|trace_only|fixed_slot_sync` 写出 `outputs.jsonl`，`tools/sew_offload/compare_smoke_outputs.py` 对 baseline/candidate 做严格 token id 比较。
- 真实 no-offload vs fixed-slot 1-token 对照已通过：baseline `artifacts/sew_offload/runs/no_offload_smoke_20260601_inline_1tok_cleanenv/outputs.jsonl` 与 candidate `artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare/outputs.jsonl` 均输出 token id `[353]`，`correctness_compare.json` 状态 `ok`。
- 真实 no-offload vs fixed-slot 2 short prompt × 8 token 对照已通过：baseline `artifacts/sew_offload/runs/no_offload_smoke_20260601_2short_8tok/outputs.jsonl` 与 candidate `artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/outputs.jsonl` 两条输出 token id 均严格一致，`correctness_compare.json` 状态 `ok`、`matched=2`。
- 真实 2 prompt × 8 token 中较长 prompt 在 `num_slots=8` 下触发 `active expert working set size 46 exceeds num_slots=8`，这是容量不足的 fail-closed 行为；不能通过 drop/clamp/remap active experts 改变 router/top-k 语义来绕过。
- 关键设计结论：slot tensor 若以 `num_slots` 为第 0 维，必须同步 remap `topk_ids/group_list` 到 slot id；不能保持原 expert id 直接替换权重。
- 已新增 fixed-slot memory ledger：runtime 可只读报告原始 expert 参数、CPU host store clone 与 slot bank backing tensors 的字节账本；离线工具 `tools/sew_offload/estimate_fixed_slot_memory.py` 可估算 Qwen3-30B-A3B 的 per-layer slot bank 成本。
- 已新增原始 expert 参数释放 readiness guard：`plan_original_weight_release(...)` 只读返回 blockers/layers_ready，当前不释放、不替换任何参数；默认需要证明默认路径已保留、所有目标层均已注册，且 host store 自检完整。
- 已把 `host_store_is_complete` 从人工布尔前置条件升级为 runtime 自检：`HostExpertStore` 会按 expected layers 检查 layer 注册、expert 覆盖、shape/dtype/stride 与 CPU host bundle，不满足则 fail closed。
- 架构澄清：MVP-D 当前 fixed-slot bank 是 CPU-backed expert cache，不代表最终系统要把所有专家都卸载到 CPU；后续应支持 NPU pinned/full-resident experts 与 CPU-backed cache slots 并存。
- MVP-D.9（2026-06-01）：`tiered_residency.py`、`expert_weight_release.py`；`MoeOffloadConfig.resident_layer_ids` / `release_original_expert_weights`；resident 层不走 `prepare_fixed_slot_plan`；release 后 `memory_ledger().original_expert_weight_bytes` 归零（UT）；`compare_slot_budget_models` 对比 per-layer slot bank vs global pool。
- 2026-06-02 设计修正：固定的是 expert weight slot / 稳定权重地址，不固定每个 expert 的 token 容量；默认继续沿用现有 dropless dynamic count/group_list 模式，`expert_capacity/drop_pad_mode` 只作为可选实验分支。
- 2026-06-02 D.9 小范围 release=1 smoke：NPU 6，1 个 non-resident layer（layer 0）+ 47 个 resident layers，release=1 成功；profile JSONL 显示 register layer 0 用时约 2.03s，release 约 0.0004s，ledger 中 original expert bytes 从约 1.208GB 降到 0；no-offload vs candidate 1-token 严格 token-id 对照通过。
- 2026-06-02 D.10 dynamic-count layered runtime：默认关闭的 path selector 已在旧 hook 位置通过真实 NPU 1-token/8-token smoke 和 strict token-id compare；8-token candidate throughput `1.675 tok/s`、TTFT `868.72 ms`、TPOT `558.14 ms`，no-offload baseline throughput `6.337 tok/s`、TTFT `445.15 ms`、TPOT `116.71 ms`。
- 2026-06-02 D.10 fused boundary 下沉：`apply()` 不再直接做 slot 权重准备，而是把 offload metadata 传入 fused input；`MoECommMethod.fused_experts()` 在 dispatch 前根据 `topk_ids` 做 path decision、slot plan 和 fail-closed。UT/compile/diff 已通过；post-downsink 真实 smoke 被 NPU 6 残留 HBM 上下文阻塞。
- 下一步顺序更新：D.11 已于 2026-06-05 完成 NPU semantic smoke（1-token & 8-token strict compare 全部 ok）；下一步进入 MVP-E async transfer 与 overlap metrics；最后考虑 D.12 staging-aware fused/custom op 或 window-aware global pool。

## MVP-D.11 实现摘要（2026-06-04）

- 新增 `vllm_ascend/moe_offload/phase_split.py`：
  - `MoEPhase` / `MoEPhasePlan`：phase split contract dataclass，含 `to_jsonable()`。
  - `compute_expert_token_slices()`：支持 group_list_type 0（cumsum）和 1（count）。
  - `plan_hit_miss_phases()`：基于 slot_readiness map 做 hit/miss 切分；支持 max_phases=1 强制单 phase。
  - `_extract_phase_tokens()`：从 sorted hidden_states 按 token_slices 抽取并 concat。
  - `_build_phase_group_list()`：构造只包含子集 expert 的 group_list（兼容 type 0/1）。
  - `_slice_expert_weights()`：对 MoEWeights 做 expert 维度的索引切片。
  - `_scatter_phase_output()`：把 phase 输出回填到完整 output buffer。
  - `execute_phased_mlp()`：顶层编排器；单 phase fast-path 直接委托 `_apply_mlp_fn`。
  - `PhaseSplitProfileEvent` + `_write_phase_split_profile_jsonl()`：observability。
- 修改 `vllm_ascend/ops/fused_moe/moe_comm_method.py`：
  - 新增 `_maybe_plan_phase_split()` 方法：检查 phase_split 启用、窄路径 gate（AllGather+unquant+no bias+no EP）、构建 phase plan、写 profile JSONL。
  - `fused_experts()` 中在 `build_mlp_compute_input` 后分支：phase_split 启用 → `execute_phased_mlp()`；否则 → 原 `_apply_mlp()`。
- 环境变量：`VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT`（默认 0）。
- 配置：`MoeOffloadConfig.phase_split_enabled`。
- UT：`tests/ut/moe_offload/test_phase_split.py`，31 个测试全部通过。
  - 覆盖：type 0/1 slicing、hit/miss/all-hit/all-miss 等价性、空 expert、group_list 构建、权重切片、scatter、profile JSONL、空 phase fallback、乱序/缺失 slice fail-closed。
- 回归：现有 moe_offload 测试 133 passed（2 预存在 import error + 1 预存在 env monkeypatch 失败未计入）。

### NPU Semantic Smoke（2026-06-05）

- 环境：NPU 2（910B3，60.40GB 空闲 HBM），Qwen3-30B-A3B，`ASCEND_RT_VISIBLE_DEVICES=2`。
- **1-token strict compare**: baseline `[26288]` == candidate `[26288]` ✅ `status=ok, matched=1`
- **8-token strict compare**: baseline `[26288,102064,104949,9370,104034,20074,89161,102021]` == candidate ✅ `status=ok, matched=1`
- Profile JSONL: 96 phase_split events (1-token), 432 (8-token)，均为 all-hit single phase fast-path。
- 开销：8-token phase_split throughput 4.71 tok/s vs baseline 5.60 tok/s（~16% Python 层开销，属语义脚手架预期范围）。
- 工具：新增 `tools/sew_offload/run_phase_split_smoke.py`。
- Ascend NPU 设备隔离需使用 `ASCEND_RT_VISIBLE_DEVICES`（非 `CUDA_VISIBLE_DEVICES`）。

## 关键约束

- 后续可比较的 SEW-Offload 实验都必须先检查 `docs/sew-offload/benchmark_config.yaml` 是否匹配；当前配置只固定模型、数据集、workload buckets、13.5GB offload budget 和指标，暂不引入复杂 benchmark framework。
- 不重训练、不微调、不修改 router，不改变 top-k expert 激活语义。
- 第一目标是单机单卡 HBM 不足时的 Qwen3-30B-A3B expert offloading。
- 所有 runtime 功能默认关闭，通过 `VLLM_ASCEND_*` 环境变量或显式配置启用。
- 尽量不改 scheduler、worker 主路径和现有 fused MoE C++ kernel；先以 Python 封装与现有 MoE boundary 接入。
- 新环境变量必须集中定义在 `vllm_ascend/envs.py`。
- 新功能必须有 UT；NPU 路径需要 E2E 或手工验证脚本。

## 初始设计判断

| 项目 | 初始判断 |
| --- | --- |
| 系统名 | SEW-Offload: Staged Expert-Weight Offloading for Dynamic-Count MoE on Ascend NPUs |
| 主要代码位置 | 新包 `vllm_ascend/moe_offload/` |
| 最小集成点 | `vllm_ascend/ops/fused_moe/fused_moe.py` 中 expert 执行边界 |
| 配置入口 | `vllm_ascend/envs.py` + 独立 config dataclass |
| 第一阶段 | 只做 routing/expert 工作集观测，不改变执行 |
| 第二阶段 | whole-expert host store + fixed weight slot + synchronous load |
| 第三阶段 | dynamic-count staged expert compute + weight residency/cache/prefetch |

## 三线交付规划

### A. 论文线

- 目标：以 CCF-A 系统/体系结构会议论文口径组织，先完成英文 LaTeX skeleton 与中文研究备忘。
- 核心叙事：GPU MoE offloading 默认以动态 expert cache/prefetch 为中心；Ascend 类 NPU 已经有 per-expert count/grouped MoE 后端，但这个后端假设 expert 权重常驻。HBM 受限时，新的控制面不是重新发明 count 化 dispatch，也不是固定每个 expert 的 token 容量，而是把 expert 权重驻留重构为固定地址的 weight slot，并在 dispatch 之后按动态 group_list 做 resident/staged/miss 分阶段执行，通过 Ascend 数据搬运与计算流水隐藏 host-to-HBM 预取开销。
- 论文非目标：不训练 router、不改 top-k、不做 expert drop、不把精度风险伪装成系统优化。
- 目标文件：`paper/outline.md`、`paper/related_work_matrix.md`、`paper/sew_offload.tex`、`paper/sew_offload.bib`、`paper/experiment_plan.md`。

### B. Slide 线

- 目标：仿 `moe_serving_report.tex` 的中文技术报告风格，围绕“问题、观察、旧工作为何不够、Ascend 机会、SEW-Offload 设计、实验计划”展开。
- 目标文件：`slide/sew_offload_report.tex`，以及后续 `slide/figures/`。

### C. 工程线

- 目标：在 `vllm-ascend-hust` 内以默认关闭、单包高内聚、少量边界 hook 的方式推进；不重复实现现有 per-expert count dispatch。
- 首选包：`vllm_ascend/moe_offload/`。
- 首选 hook：短期仍在 `AscendUnquantizedFusedMoEMethod.apply()` 到 `moe_comm_method.fused_experts()` 的 expert execution boundary 做权重所有权和 slot remap；真正的 staged compute 切入点应下沉到 `MoECommMethod.fused_experts()` 内部的 `token_dispatch_output -> build_mlp_compute_input -> _apply_mlp` 边界。
- 严控边界：暂不改 `model_runner` 主路径、暂不改 scheduler、暂不大改 Ascend C kernels。
- 目标文档：`docs/sew-offload/00-charter.md`、`01-system-design.md`、`02-implementation-plan.md`、`03-experiment-plan.md`、`04-reproduction.md`。

## 当前文档状态

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| `docs/sew-offload/00-charter.md` | complete | 项目目标、研究问题、目标模型、贡献与非目标 |
| `docs/sew-offload/01-system-design.md` | complete | 固定 slot、deadline-aware prefetch、hit-first phased execution |
| `docs/sew-offload/02-implementation-plan.md` | complete | runtime 模块、TDD 任务、hook、测试与 commit 切分 |
| `docs/sew-offload/03-experiment-plan.md` | complete | workloads、baselines、metrics、ablation、图表计划 |
| `docs/sew-offload/04-reproduction.md` | complete | 环境检查、trace、simulator、NPU 实测、故障排查 |
| `docs/sew-offload/05-existing-offload-baseline.md` | complete | Qwen3-30B-A3B 单卡现有 offload 实测结果与瓶颈诊断 |
| `docs/sew-offload/06-benchmark-design.md` | complete | 最小可复用 benchmark：固定模型、数据集、workload buckets、13.5GB offload budget、指标 |
| `docs/sew-offload/benchmark_config.yaml` | complete | 上述最小 benchmark 的机器可读配置 |
| `docs/sew-offload/07-native-offload-benchmark-results.md` | complete | 最小 benchmark runner、native prefetch offload 失败结果、no-offload throughput/TTFT/TPOT sanity |
| `docs/sew-offload/08-ascend-moe-offload-architecture.md` | complete | 再次核实现有能力缺口，并给出 Ascend NPU MoE Offloading 整体架构设计与 Mermaid 架构图 |
| `docs/sew-offload/09-next-steps-after-mvp-a.md` | complete | MVP-A 之后的 MVP-B 到 MVP-G 实施计划、文件计划、测试门禁和两周排期 |
| `docs/sew-offload/11-expert-transfer-breakdown-and-pipeline.md` | complete | 当前 expert miss 搬运分解、CANN profiler 证据、three-pattern 对照、MVP-E overlap 优化方向 |

## 未决问题

- 目标 CCF-A 会议优先级：系统会议优先还是体系结构会议优先。
- 论文初稿语言：英文为主，是否保留中文 slide。
- 是否允许在 `paper/` 和 `slide/` 下立即创建 skeleton。
- 单卡 HBM 不足的实验模拟方式：真实限制 cache budget、保留 KV cache 压力、或人工减少 expert resident slots。
- 分层驻留策略如何配置：按层保留完整 experts、按全局热点 expert 保留、还是二者组合；需要 trace/simulator 给出保留预算与命中/搬运收益曲线。
- 参数所有权转移应支持 partial release：未卸载专家继续由 NPU 原始参数/常驻 bank 拥有，卸载专家由 HostExpertStore + ExpertSlotBank 拥有，避免错误地把“释放全部 expert 参数”当作唯一优化目标。
- activation/token capacity 默认不固定：当前常规 MoE 已使用 `expert_tokens_num_type=1` + `group_list_type=1` 的动态 count 路径；SEW-Offload 默认只做 expert weight residency/remap，不引入 drop/pad 改变 token 语义。
- 2026-06-02 源码复核修正：`Static Expert Window` 不应再解释为固定每专家 token 容量；项目主线改为 `Dynamic-Count Staged Expert-Weight Offloading`，即保留现有 dynamic count dispatch/grouped matmul 语义，在 dispatch 后围绕专家权重驻留、搬运和分阶段 MLP 计算做系统设计。

## 当前实测任务：现有 offloading baseline

- 模型路径：用户给出 `/data/Qwen3-30B-A3B`，当前机器实际存在 `/data/shared-models/Qwen3-30B-A3B`。
- 运行栈：`/root/miniconda3/envs/vllm-hust-dev/bin/python`，editable `vllm-hust` + `vllm-ascend-hust`。
- 首选设备：NPU 4，其次 NPU 6/3；避免占用较高的 0/1/2/5/7。
- 目标：用现有 vLLM weight offloading 方法跑 baseline，不修改 runtime 主代码；收集启动、加载、OOM/报错、HBM 曲线、吞吐/延迟数据，判断现有方法在 Ascend 单卡 MoE expert offloading 场景到底缺什么。

## 阶段 8 实测结论

- artifact：`artifacts/sew_offload/existing_offload_20260529T143705Z`
- no-offload：成功，`LOAD_OK 49.062s`，`GENERATE_OK 17.158s`，日志显示模型权重 `56.9001 GB`，NPU 4 HBM 峰值约 `63886 MB`。
- UVA expert offload：失败于 `get_accelerator_view_from_cpu_tensor`，当前 NPU 平台不支持该 UVA accelerator view。
- prefetch expert offload：能把模型权重驻留降到 `43.4001 GB`、HBM 峰值约 `50697 MB`，但原始路径失败于裸 `torch.cuda.is_current_stream_capturing()`；补齐 CUDA-to-NPU wrapper 后，进一步失败于 `npu_grouped_matmul` 收到 CPU weight。
- 研究判断：现有 offload 能证明“省 HBM 有价值”，但不是 Ascend MoE 可直接采用的运行时抽象。SEW-Offload 应聚焦 MoE execution boundary 上的 fixed expert slot、post-processed layout-stable NPU buffers、expert-aware prefetch 和 hit-first phased execution。
