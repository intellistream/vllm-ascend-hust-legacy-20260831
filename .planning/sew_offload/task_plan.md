# SEW-Offload 项目计划

## 目标

在 `vllm-ascend-hust` 中以高内聚、低耦合、默认关闭的方式推进 Ascend 类 NPU 上的 MoE expert offloading 研究，并产出：

1. CCF-A 类会议论文：`paper/`
2. 仿 `moe_serving_report.tex` 风格的汇报 slide：`slide/`
3. 基于 vLLM Ascend 的低侵入实现、实验与复现材料。

## 当前阶段

阶段 10/11：MVP-A/B/C 已完成，MVP-D fixed-slot correctness 已跑通 Qwen3-30B-A3B 单卡 prefetch+fixed-slot 1-token smoke；MVP-D.5 已补 no-offload vs fixed-slot 独立进程 correctness 对照工具，并通过真实 1-token 与 2 short prompt × 8 token 的 token-id 严格对照。当前重点是审查 fixed-slot working-set 容量边界、释放/替换原始 expert 参数前的生命周期，以及 MVP-E async transfer 设计。

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
| 10. SEW runtime MVP | in_progress | MVP-A/B/C 完成；MVP-D fixed-slot sync correctness 最小闭环已通过真实 NPU smoke |
| 11. MoE Offload 支持核实与总体架构 | complete | `docs/sew-offload/08-ascend-moe-offload-architecture.md`：不支持证据链、系统架构、控制/数据面图、路线图 |

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
- 下一步：在通过更多默认路径验证后，设计 post-load 后“分层驻留”的参数所有权转移方案，包括保留若干完整层/热点专家在 NPU、只释放 cold/offloaded experts、并让 fixed-slot cache 处理 CPU expert miss；之后再进入 MVP-E async transfer。

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
| 系统名 | SEW-Offload: Static Expert-Window Scheduling for Prefetch-Hidden MoE Offloading on Ascend NPUs |
| 主要代码位置 | 新包 `vllm_ascend/moe_offload/` |
| 最小集成点 | `vllm_ascend/ops/fused_moe/fused_moe.py` 中 expert 执行边界 |
| 配置入口 | `vllm_ascend/envs.py` + 独立 config dataclass |
| 第一阶段 | 只做 routing/expert 工作集观测，不改变执行 |
| 第二阶段 | whole-expert host store + fixed slot + synchronous load |
| 第三阶段 | fixed expert-slot residency window + cache/prefetch + graph replay |

## 三线交付规划

### A. 论文线

- 目标：以 CCF-A 系统/体系结构会议论文口径组织，先完成英文 LaTeX skeleton 与中文研究备忘。
- 核心叙事：GPU MoE offloading 默认以动态 expert cache/prefetch 为中心；Ascend 类 NPU 已经有 per-expert count/grouped MoE 后端，但这个后端假设 expert 权重常驻。HBM 受限时，新的控制面不是重新发明 count 化 dispatch，而是把 expert 权重驻留重构为固定地址的 expert-slot window，并通过 deadline-aware prefetch/orchestration 与 hit-first phased execution 隐藏 host-to-HBM 预取开销。
- 论文非目标：不训练 router、不改 top-k、不做 expert drop、不把精度风险伪装成系统优化。
- 目标文件：`paper/outline.md`、`paper/related_work_matrix.md`、`paper/sew_offload.tex`、`paper/sew_offload.bib`、`paper/experiment_plan.md`。

### B. Slide 线

- 目标：仿 `moe_serving_report.tex` 的中文技术报告风格，围绕“问题、观察、旧工作为何不够、Ascend 机会、SEW-Offload 设计、实验计划”展开。
- 目标文件：`slide/sew_offload_report.tex`，以及后续 `slide/figures/`。

### C. 工程线

- 目标：在 `vllm-ascend-hust` 内以默认关闭、单包高内聚、少量边界 hook 的方式推进；不重复实现现有 per-expert count dispatch。
- 首选包：`vllm_ascend/moe_offload/`。
- 首选 hook：`AscendUnquantizedFusedMoEMethod.apply()` 之后、`moe_comm_method.fused_experts()` 之前的 expert execution boundary；MVP-0 可复用 routed experts capturer 思路。
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

## 未决问题

- 目标 CCF-A 会议优先级：系统会议优先还是体系结构会议优先。
- 论文初稿语言：英文为主，是否保留中文 slide。
- 是否允许在 `paper/` 和 `slide/` 下立即创建 skeleton。
- 单卡 HBM 不足的实验模拟方式：真实限制 cache budget、保留 KV cache 压力、或人工减少 expert resident slots。
- 分层驻留策略如何配置：按层保留完整 experts、按全局热点 expert 保留、还是二者组合；需要 trace/simulator 给出保留预算与命中/搬运收益曲线。
- 参数所有权转移应支持 partial release：未卸载专家继续由 NPU 原始参数/常驻 bank 拥有，卸载专家由 HostExpertStore + ExpertSlotBank 拥有，避免错误地把“释放全部 expert 参数”当作唯一优化目标。

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
