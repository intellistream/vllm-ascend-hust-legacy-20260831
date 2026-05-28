# SEW-Offload 项目计划

## 目标

在 `vllm-ascend-hust` 中以高内聚、低耦合、默认关闭的方式推进 Ascend 类 NPU 上的 MoE expert offloading 研究，并产出：

1. CCF-A 类会议论文：`paper/`
2. 仿 `moe_serving_report.tex` 风格的汇报 slide：`slide/`
3. 基于 vLLM Ascend 的低侵入实现、实验与复现材料。

## 当前阶段

阶段 0：上下文建立与项目规划。

## 阶段清单

| 阶段 | 状态 | 目标产物 |
| --- | --- | --- |
| 0. 上下文建立 | in_progress | 仓库结构、MoE 代码边界、paper/slide 目录、工程规则 |
| 1. 研究问题冻结 | pending | RQ、假设链、贡献点、非目标 |
| 2. 论文蓝图 | pending | `paper/outline.md`、`paper/related_work_matrix.md`、LaTeX skeleton |
| 3. Slide 蓝图 | pending | `slide/sew_offload_report.tex` skeleton 与图表清单 |
| 4. Runtime 架构设计 | pending | 高内聚低耦合模块边界、配置开关、集成点 |
| 5. MVP-0/MVP-1 实现计划 | pending | routing trace、expert store、slot manager、fixed window |
| 6. 实验计划 | pending | baseline、workload、metrics、ablation、artifact layout |
| 7. 质量门禁 | pending | correctness、性能、citation、复现、默认关闭验证 |

## 关键约束

- 不重训练、不微调、不修改 router，不改变 top-k expert 激活语义。
- 第一目标是单机单卡 HBM 不足时的 Qwen3-30B-A3B expert offloading。
- 所有 runtime 功能默认关闭，通过 `VLLM_ASCEND_*` 环境变量或显式配置启用。
- 尽量不改 scheduler、worker 主路径和现有 fused MoE C++ kernel；先以 Python 封装与现有 MoE boundary 接入。
- 新环境变量必须集中定义在 `vllm_ascend/envs.py`。
- 新功能必须有 UT；NPU 路径需要 E2E 或手工验证脚本。

## 初始设计判断

| 项目 | 初始判断 |
| --- | --- |
| 系统名 | SEW-Offload: Static Expert Windowing for Ascend MoE Offloading |
| 主要代码位置 | 新包 `vllm_ascend/moe_offload/` |
| 最小集成点 | `vllm_ascend/ops/fused_moe/fused_moe.py` 中 expert 执行边界 |
| 配置入口 | `vllm_ascend/envs.py` + 独立 config dataclass |
| 第一阶段 | 只做 routing/expert 工作集观测，不改变执行 |
| 第二阶段 | whole-expert host store + fixed slot + synchronous load |
| 第三阶段 | static window + cache/prefetch + graph replay |

## 未决问题

- 目标 CCF-A 会议优先级：系统会议优先还是体系结构会议优先。
- 论文初稿语言：英文为主，是否保留中文 slide。
- 是否允许在 `paper/` 和 `slide/` 下立即创建 skeleton。
- 单卡 HBM 不足的实验模拟方式：真实限制 cache budget、保留 KV cache 压力、或人工减少 expert resident slots。
