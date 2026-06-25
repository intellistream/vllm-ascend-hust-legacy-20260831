# SEW-Offload 进度记录

## 2026-05-28

- 确认 `vllm-ascend-hust` 当前在 `/root/vllm-ascend-hust`，与其他 repo 并列存在。
- 检查 git 状态，当前分支为 `research`，工作树干净。
- 确认 `paper/` 和 `slide/` 目录存在但为空。
- 阅读 `CLAUDE.md` 和 `AGENTS.md` 的关键工程规则。
- 扫描 MoE 相关 Python 与 C++/Ascend C 文件，确认应以 `vllm_ascend/ops/fused_moe/` 作为首要集成边界。
- 建立 `.planning/sew_offload/` 文件规划区。

## 下一步

- 输出三线项目规划：论文、slide、runtime 实现。
- 等用户确认后，创建 `paper/`、`slide/`、`docs/` 下的正式 skeleton。

## 2026-05-28 追加

- 阅读 `planning-with-files-zh`、`brainstorming`、`context-engineering`、`academic-paper`、`academic-pipeline` 的执行约束。
- 阅读参考 slide `/root/slide/moe_serving_report.tex`，确认应沿用中文技术报告结构。
- 阅读 `vllm_ascend/ops/fused_moe/fused_moe.py`、`token_dispatcher.py`、`moe_runtime_args.py` 与 routed expert capture patch，确认首要集成边界。
- 阅读 graph mode 与 weight prefetch 文档，确认 Ascend 的 static kernel、ACLGraph/Npugraph_ex、MTE/权重预取可作为论文与实现的硬件控制面。
- 等用户确认当前设计后，再创建 paper/slide/docs skeleton；运行时代码暂不修改。

## 2026-05-28 关键修正

- 根据用户指出的问题，重新审视 `token_dispatcher.py`、`moe_comm_method.py`、`moe_mlp.py` 与 `moe_grouped_matmul`。
- 确认现有 Ascend MoE 后端已经实现 per-expert token count/group_list 表示，不能把它作为 SEW-Offload 新贡献。
- 修正研究定位：SEW-Offload 应聚焦 HBM 受限 offloading 下的固定 expert-slot residency window、稳定权重地址、slot remap/prefetch，以及可选 capacity-tier graph replay。

## 2026-05-28 slide 落盘

- 已创建正式 slide 文件：`slide/sew_offload_report.tex`。
- 采用与 `moe_serving_report.tex` 相近的中文技术叙事结构，并把研究主线修正为“已有 grouped count 后端 + 新增 fixed expert-slot residency window”。
- 已安装 TeX/PDF 编译环境并生成正式 PDF：`slide/sew_offload_report.pdf`。
- 已安装/确认的关键包包括 `texlive-xetex`、`texlive-ctex`、`texlive-fandol`、`texlive-beamer`、`texlive-pgf`、`latexmk`、`texlive-everysel`、`texlive-everyshi`。
- 额外安装 Noto CJK 字体包，并在 `slide/sew_offload_report.tex` 中显式指定 Noto CJK 字体，避免“昇腾”等中文字符缺字。
- 已用 `latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error sew_offload_report.tex` 验证可编译，输出 16 页 PDF。

## 2026-05-28 paper design 落盘

- 根据用户修正，将 SEW-Offload 的论文核心目标从“固定 slot/cache”进一步收敛为“通过编排隐藏 offloading 预取时间”。
- 新主线确定为：固定 slot 解决 Ascend 稳定执行；deadline-aware prefetch/orchestration 优化 offloading 暴露开销；hit-first phased execution 在 miss 已经发生时继续隐藏等待。
- 已创建 `paper/sew_offload_design.tex`，作为英文 LaTeX 论文设计草稿，覆盖 motivation、problem statement、old assumptions vs new control surface、design overview、components、scheduling algorithm、Ascend-specific rationale、vLLM Ascend integration、correctness、evaluation plan、limitations。
- 已创建 `paper/README.md`，记录当前 paper 目录文件和中文设计主线。
- 已用 `latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error sew_offload_design.tex` 编译验证，输出 6 页 PDF：`paper/sew_offload_design.pdf`。编译仅有 underfull hbox / float placement 级别排版警告，无 fatal error。
- 核对时曾用不兼容的 `rg` 正则检查占位符并触发 regex parse error；已改用固定字符串检查，未发现 `TODO`、`TBD`、`\cite{` 或 `placeholder`。

## 2026-05-28 research question 重定义

- 按用户要求调研 CCF-A 新硬件/新设备系统论文的问题定义写法，提炼出“旧假设失效 -> 新硬件暴露新控制面 -> 旧系统无法直接利用 -> 新系统抽象”的写作模式。
- 已创建 `paper/research_question_reframing.md`，记录代表论文、可复用模式、对 SEW-Offload 的启发、主 RQ、子问题和审稿人压力测试。
- 已更新 `paper/sew_offload_design.tex`，将 `Problem Statement` 改为 `Research Question and Problem Statement`，并写入新版研究问题。
- 已更新 `paper/README.md`，加入当前研究问题摘要。
- 已重新编译 `paper/sew_offload_design.tex`，输出 6 页 `paper/sew_offload_design.pdf`；清理 LaTeX 临时文件后确认 PDF 文件有效。

## 2026-05-28 research problem 陈述式修正

- 根据用户反馈，确认顶会论文的问题定义不应写成“如何做”的方案导向问句，而应写成陈述式问题链：现有方法怎么做、依赖什么假设、为什么在新硬件上失效、导致什么系统性结果。
- 已将 `paper/research_question_reframing.md` 中的 `Proposed Research Question` 改为 `Proposed Problem Definition`，并把 Main RQ/Short/System version 改为 Main problem statement/Short paper version/Thesis statement。
- 已将 `paper/sew_offload_design.tex` 中 `Research Question and Problem Statement` 改为 `Problem Definition`，并替换为陈述式问题定义。
- 已同步更新 `paper/README.md` 和 `findings.md`。

## 2026-05-28 abstract 问题定义修正

- 根据用户进一步指出的问题，将 `paper/sew_offload_design.tex` 摘要中的问题定义段落改为“现有方法怎么做 -> 为什么在 Ascend 上不行 -> 导致什么结果”的通用顶会摘要句式。
- 新摘要明确写出：现有 MoE offloading 把 expert weights 当作 dynamically cached device objects，通过 cache replacement、host-to-device prefetching、stream-level copy/compute overlap 降低传输开销；但 limited HBM 下动态加载到任意 device buffers 会破坏 Ascend 依赖的 stable weight addresses、fixed execution windows、graph replay、explicit movement scheduling；结果是 host-to-HBM loading 暴露在关键路径上，或放弃 NPU 高效执行需要的静态规律。

## 2026-05-29 当前设备与模型规划调研

- 按用户要求聚焦“单机单卡 Ascend NPU 上 MoE offloading 如何通过编排隐藏预取时间”重新梳理实验规划。
- 检查当前设备：8 张 Ascend 910B3，每张 64GB HBM；当前部分卡被大进程占用。
- 检查软件栈：CANN 8.5.1 / npu-smi 25.3.rc1 / openEuler 24.03；默认 Python 环境缺少 torch/torch_npu/vllm/transformers，需要后续切到正确运行环境。
- 检查本地模型：未发现 Qwen3-30B-A3B；发现 Qwen3.5-122B-A10B（234G，qwen3_5_moe，256 experts，top-8）和 Qwen3 dense 系列。
- 阅读本地 vLLM Ascend weight prefetch、ACLGraph、MoE prefetch 代码，确认现有能力是 HBM/L2 预取与图执行基础，不等价于 host->HBM expert offloading。
- 将 findings 更新为：首选模型分层为 Qwen3-30B-A3B 主模型、Qwen3.5-122B-A10B 压力模型、dense Qwen3 仅做系统基线；硬件控制面分为 ACLGraph/static shape、固定 slot 地址、stream/piecewise graph 约束、npu_prefetch/MTE 数据搬运。

## 2026-05-29 Runtime system design 落盘

- 根据用户要求，将“利用当前 Ascend NPU 特性实现专家高效预取，并通过并行计算掩盖专家加载时间”的方案正式写入 `docs/sew-offload/01-system-design.md`。
- 文档明确拆分两个核心子系统：`Expert Prefetch Planner` 与 `Overlap Execution Scheduler`。
- 文档细化了 fixed expert slots、deadline-aware prefetch、TransferEngine、hit-first phased execution、CostModel、vLLM Ascend 集成边界、correctness 约束、评价指标和分阶段实现路线。
- 已将 `task_plan.md` 当前阶段更新为 Runtime 架构设计审阅，阶段 4 状态标记为 `in_progress`。

## 2026-05-29 项目文档补齐

- 根据用户“继续完成没有做完的工作”的要求，补齐 `docs/sew-offload/` 下的项目文档集。
- 新增 `docs/sew-offload/00-charter.md`，明确项目名称、研究问题、目标硬件/模型、设计原则、贡献、非目标、成功标准和风险。
- 新增 `docs/sew-offload/02-implementation-plan.md`，按照 implementation plan 形式拆解 runtime 模块、测试、hook、simulator、sync slot、async prefetch 和 phased execution 任务。
- 新增 `docs/sew-offload/03-experiment-plan.md`，定义模型、workloads、baselines、ablations、metrics、trace、simulator、NPU 实测和图表计划。
- 新增 `docs/sew-offload/04-reproduction.md`，记录环境检查、模型准备、环境变量、单测、trace-only、simulator、NPU runtime 实测、profiling 数据和故障排查。
- 更新 `task_plan.md`：阶段 4/5/6 标记为 complete，阶段 7 进入 `in_progress`；新增当前文档状态表。
- 已验证 `docs/sew-offload/` 五份 Markdown 文档均存在且非空；对新文档扫描 `TODO|TBD|FIXME|待补充` 未发现遗留占位符。

## 2026-05-29 现有 offloading baseline 实测启动

- 用户要求停止纸面推演，改为用真实 Qwen3-30B-A3B 和单卡 Ascend NPU 跑现有 offloading 方法，找出真实瓶颈。
- 已恢复 `.planning/sew_offload/` 规划上下文，并新增阶段 8：现有单卡 offloading baseline 实测与瓶颈定位。
- 确认正确运行环境为 `/root/miniconda3/envs/vllm-hust-dev/bin/python`，其中 `vllm` 来自 `/root/vllm-hust`，`vllm_ascend` 来自 `/root/vllm-ascend-hust`。
- 用户给出的 `/data/Qwen3-30B-A3B` 当前不存在；实际可用模型路径为 `/data/shared-models/Qwen3-30B-A3B`。
- 模型配置：`qwen3_moe`，48 层，hidden size 2048，128 experts，top-8，MoE intermediate size 768，bf16。
- 当前设备为 8 张 Ascend 910B3；NPU 4/6/3 基本空闲，优先使用 NPU 4。
- 已确认 vLLM CLI 暴露现有 weight offload 参数：`--offload-backend {auto,prefetch,uva}`、`--cpu-offload-gb`、`--cpu-offload-params`、`--offload-group-size`、`--offload-num-in-group`、`--offload-prefetch-step`、`--offload-params`。

## 2026-05-29 现有 offloading baseline 实测完成

- 创建并使用实验脚本 `tools/sew_offload/run_existing_offload_case.py`，所有结果写入 `artifacts/sew_offload/existing_offload_20260529T143705Z`。
- 为跑通当前本地版本组合，修复 `vllm_ascend/ops/fused_moe/fused_moe.py` 中的 MoE runner API 兼容问题：
  - `default_moe_runner` 改为当前 `moe_runner.MoERunner` 路径。
  - 对缺失的 `SharedFusedMoE` 模块加兼容 mixin。
  - 将旧字段 `_routed_input_transform` 改为从当前 runner 参数读取。
  - 对当前 vLLM 已移除的 `reduce_results` 使用单卡安全默认值 `False`。
- 为验证 prefetch 后端，补齐 `vllm_ascend/worker/model_runner_v1.py` 的 `torch.cuda.is_current_stream_capturing -> torch.npu.is_current_stream_capturing` wrapper 映射。
- no-offload case `baseline_no_offload_after_reduce_results_patch` 成功：
  - return code `0`，elapsed `89s`
  - `LOAD_OK seconds=49.062`
  - `GENERATE_OK seconds=17.158`
  - `Loading model weights took 56.9001 GB`
  - NPU 4 HBM peak `63886 MB`
- UVA case `uva_experts_cpu8gb` 失败：
  - return code `1`
  - offloader 被设置为 `UVAOffloader`
  - 失败于 `get_accelerator_view_from_cpu_tensor` 不支持 `npu`
- prefetch case `prefetch_experts_group4_num1_step1` 失败：
  - return code `1`
  - `Loading model weights took 43.4001 GB`
  - NPU 4 HBM peak `50697 MB`
  - 失败于裸 `torch.cuda.is_current_stream_capturing()`
- prefetch wrapper patched case `prefetch_experts_group4_num1_step1_after_cuda_wrapper_patch` 失败：
  - return code `1`
  - `Loading model weights took 43.4001 GB`
  - NPU 4 HBM peak `50696 MB`
  - 失败于 `npu_grouped_matmul` 收到 CPU weight 与 NPU hidden states
- 新增正式记录文档 `docs/sew-offload/05-existing-offload-baseline.md`。
- 更新 `task_plan.md`：阶段 8 标记为 complete，新增阶段 9：SEW runtime MVP。

## 2026-05-29 统一 Benchmark 规范落盘

- 按用户要求将统一 benchmark 正式落到 `docs/sew-offload/06-benchmark-design.md`。
- 新增机器可读配置规范 `docs/sew-offload/benchmark_config.yaml`，固定模型、数据集、request manifest、workload buckets、offload budget、methods、metrics、artifact layout 和 validity gates。
- 将“后续所有 SEW-Offload 实验都必须先读取并遵守 `docs/sew-offload/benchmark_config.yaml`，并记录 `config_sha256`”写入 `AGENTS.md`，作为项目级持久上下文规则。
- 将同一强制规则写入 `.planning/sew_offload/task_plan.md` 的关键约束，确保规划上下文恢复后也会先看到该要求。
- 更新 `.planning/sew_offload/findings.md`，记录 benchmark 固定选择、公平性原则和核心指标。
- 校验 `git diff --check` 通过；系统 `python` 命令不存在，已改用 `/root/miniconda3/envs/vllm-hust-dev/bin/python` 成功解析 `benchmark_config.yaml`。
- 当前 `benchmark_config.yaml` sha256：`980fb359028a14cf5b5f18fd7bbc3d39d0426a8fb771b0ee29aa461848937f26`。

## 2026-05-29 Benchmark 规范精简

- 用户指出上一版 benchmark 设计过重，当前阶段只需要高复用最小定义。
- 已重写 `docs/sew-offload/06-benchmark-design.md`，只保留固定模型、数据集、workload buckets、13.5GB offload budget、指标。
- 已重写 `docs/sew-offload/benchmark_config.yaml`，移除 future SEW methods、artifact layout、validity gates、复杂 metadata 约束。
- 已同步精简 `AGENTS.md` 和 `.planning/sew_offload/task_plan.md` 中的上下文规则：后续可比较实验先检查配置是否匹配，但暂不引入复杂 benchmark framework。

## 2026-05-29 Native Offloading Benchmark Pilot

- 新增 runner：`tools/sew_offload/run_minimal_offload_benchmark.py`，支持按 `benchmark_config.yaml` 生成 smoke manifest 并运行 Qwen3-30B-A3B，输出 throughput、TTFT、TPOT。
- 生成 smoke manifest：`artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/requests.jsonl`，1 条 `short_chat`，128 prompt tokens，128 output tokens。
- 运行 `native_prefetch_experts_short1`：resident weight `43.4001 GB`，失败于 `wrapper__npu_grouped_matmul`，错误为 weight 在 CPU、其他张量在 `npu:0`。
- 运行 `native_prefetch_all_short1`：resident weight `42.9722 GB`，失败于 `wrapper_NPU__matmul`，错误为 CPU tensor 与 NPU tensor 混用，并出现 Ascend vector core / MTE DDR address out-of-range 异常。
- 运行 `no_offload_short1` sanity：resident weight `56.9001 GB`，成功；output throughput `7.6207 tok/s`，TTFT `465.78 ms`，TPOT `128.59 ms`。
- 新增结果记录：`docs/sew-offload/07-native-offload-benchmark-results.md`。
- 更新 `.gitignore`，忽略本地 benchmark artifacts 与 Ascend 异常生成的 `extra-info/` dump。
- 校验 `tools/sew_offload/run_minimal_offload_benchmark.py` 通过 `py_compile`；`git diff --check` 通过。

## 2026-05-29 Git ignore 清理

- 用户指出 git 中堆积了很多实验 txt/log 文件；检查确认这些文件都在 `artifacts/sew_offload/...` 下，属于本地实验输出，包括 `command.txt`、`run.log`、`npu_monitor.txt`、`return_code.txt`、`elapsed_seconds.txt` 等。
- 已在 `.gitignore` 中新增 `/artifacts/`，只忽略本地实验产物目录，不忽略 `docs/sew-offload/` 和 `tools/sew_offload/run_existing_offload_case.py`。
- 验证 `git check-ignore`：`artifacts/.../command.txt` 和 `artifacts/.../run.log` 均由 `.gitignore:207:/artifacts/` 命中。
- 验证 `py_compile`：`vllm_ascend/ops/fused_moe/fused_moe.py`、`vllm_ascend/worker/model_runner_v1.py`、`tools/sew_offload/run_existing_offload_case.py` 语法检查通过。

## 2026-05-31 MoE Offload 支持再核实与架构设计

- 恢复 `.planning/sew_offload/` 上下文，确认前序实测已经证明 native UVA/prefetch offload 不能跑通 Qwen3-30B-A3B 的 Ascend MoE 生成路径。
- 重新检查 `/root/vllm-hust` 的 `vllm/config/offload.py`、`vllm/model_executor/offloader/base.py`、`prefetch.py`、`uva.py`、`prefetch_ops.py`，确认现有 offload 是通用参数/层级抽象，不是 MoE expert working-set runtime。
- 重新检查 `/root/vllm-ascend-hust` 的 `fused_moe.py`、`moe_mlp.py`、`token_dispatcher.py`、`weight_prefetch.py` 和相关文档，确认 vLLM Ascend 有 grouped MoE、EP/EPLB、HBM/cache weight prefetch，但没有 host->HBM expert offload slot runtime。
- 复核本地 artifact 日志：UVA 失败于 NPU 不支持 accelerator CPU tensor view；expert prefetch 降低 resident weight 到 `43.4001 GB` 后失败于 `npu_grouped_matmul` 收到 CPU weight；all-param prefetch 降到 `42.9722 GB` 后失败于 dense matmul CPU/NPU 混用和 Ascend MTE 异常；no-offload 能输出 throughput/TTFT/TPOT。
- 新增 `docs/sew-offload/08-ascend-moe-offload-architecture.md`，包含核实结论、证据链、总体架构、Mermaid 系统图、控制面图、数据面图、状态机、模块设计、执行模式、路线图和风险表。
- 更新 `task_plan.md` 当前阶段为阶段 11，并记录 `08-ascend-moe-offload-architecture.md` 为 complete。
- 更新 `findings.md`，记录 2026-05-31 再核实结论和总体架构决策。

## 2026-05-31 MVP-A Trace-Only 实现

- 将 `docs/sew-offload/08-ascend-moe-offload-architecture.md` 保持为中文技术表达，保留核实结论、证据链、Mermaid 架构图、模块路线和 correctness 不变量。
- 新增 `vllm_ascend/moe_offload/`：
  - `config.py`：`MoeOffloadConfig` 从集中 env 读取配置，默认关闭。
  - `trace_collector.py`：记录 `TraceRecord`、active experts、per-expert token counts，并支持 bounded history 和 JSON 导出。
  - `runtime.py`：提供 trace-only runtime、全局 lazy runtime 和 reset helper。
  - `__init__.py`：导出 MVP-A API。
- 修改 `vllm_ascend/envs.py`，集中注册 `VLLM_ASCEND_MOE_OFFLOAD_ENABLED`、`TRACE_ONLY`、`NUM_SLOTS`、`POLICY`、`MAX_PHASES`、`ASYNC_LOAD`、`TRACE_MAX_RECORDS`。
- 修改 `vllm_ascend/ops/fused_moe/fused_moe.py`，在 `select_experts(...)` 之后接入 `get_moe_offload_runtime().trace_routing(...)`；该 hook 原样返回 `topk_ids/topk_weights`，不改变后续 routed expert capture、zero expert、load balance 或 grouped MoE 执行。
- 新增 UT：`tests/ut/moe_offload/test_config.py`、`test_trace_collector.py`、`test_runtime_trace_only.py`。
- 修复 `tests/ut/test_envs.py` 对 `float(...)` env handler 的测试输入推断，保证新增 env 可进入既有 env 测试。
- 验证结果：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_config.py tests/ut/moe_offload/test_trace_collector.py tests/ut/moe_offload/test_runtime_trace_only.py`：`9 passed, 4 warnings in 0.05s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/test_envs.py`：`2 passed, 4 warnings, 78 subtests passed in 0.05s`。
  - `py_compile` 覆盖新增包、改动源码和相关测试：通过。
  - `git diff --check`：通过。
  - Markdown fence 检查：`mermaid_fences=5`、`total_fences=64`、`balanced=True`、中文关键段落存在。
- 额外验证尝试：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/ops/test_fused_moe.py` 在收集阶段失败，原因是当前环境缺少 `pytest_mock`；未触发本次 fused MoE hook 业务断言。MVP-A hook 已由新增 trace-only UT 覆盖。

## 2026-05-31 MVP-A 后续计划

- 新增 `docs/sew-offload/09-next-steps-after-mvp-a.md`，作为 MVP-A 之后的执行计划。
- 计划明确下一步不直接做 async prefetch，而是按 `MVP-B trace export -> MVP-C offline simulator -> MVP-D fixed slot correctness -> MVP-E async transfer -> MVP-F hit-first phases -> MVP-G static window` 推进。
- MVP-B 目标：导出 JSONL trace、增加 trace collection CLI、采集 Qwen3-30B-A3B smoke trace artifact。
- MVP-C 目标：用离线 simulator sweep slot budget 与 policy，输出 hit/miss、eviction、host-to-HBM bytes、estimated stall 和 phase opportunity。
- MVP-D 目标：实现 HostExpertStore、ExpertSlotBank、LayoutValidator、同步 TransferEngine，并确保 grouped MoE backend 只消费 NPU slot tensors。
- 计划中写入了每个 MVP 的文件计划、测试命令、完成标准和两周建议排期。

## 2026-05-31 MVP-B Trace Export and Collection

- 按 TDD 新增 `tests/ut/moe_offload/test_trace_export.py`，先观察到 `TraceCollector.to_jsonl` 和 `MoeOffloadRuntime.export_trace` 缺失导致的 RED，再实现最小导出能力。
- 修改 `vllm_ascend/moe_offload/trace_collector.py`：
  - 新增 `to_jsonl()`，逐条输出 JSONL。
  - 新增 `write_jsonl(path)`，自动创建父目录并返回记录数。
- 修改 `vllm_ascend/moe_offload/runtime.py`：
  - 新增 `export_trace(path)`，委托 collector 写出 trace。
- 新增 `tools/sew_offload/collect_moe_trace.py`：
  - 支持读取 `benchmark_config.yaml`。
  - 支持 `--prepare-smoke-manifest --prepare-only` 生成 synthetic smoke manifest。
  - 支持加载 manifest、强制开启 `VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1` 与 `TRACE_ONLY=1`、运行 vLLM、导出 JSONL trace。
- 新增 `tests/ut/moe_offload/test_collect_moe_trace.py`，覆盖 manifest 生成、bucket filter、max request limit 和空选择错误。
- 更新 `docs/sew-offload/04-reproduction.md`，把 trace-only 命令修正为 `tools/sew_offload/collect_moe_trace.py`，并同步当前 env 变量名。
- 验证结果：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_trace_export.py tests/ut/moe_offload/test_trace_collector.py tests/ut/moe_offload/test_runtime_trace_only.py tests/ut/moe_offload/test_collect_moe_trace.py`：`12 passed, 4 warnings in 0.34s`。
  - `py_compile` 覆盖 trace collector、runtime、CLI 和新增测试：通过。
  - `git diff --check`：通过。
  - CLI prepare-only smoke：生成 `/tmp/sew_trace_smoke_requests.jsonl`，输出 `PREPARE_OK`，文件 1 行。
- 未运行真实 Qwen3-30B-A3B trace collection；该命令会加载大模型并占用 NPU，后续可在选定空闲卡上执行。

## 2026-05-31 MVP-C Offline Slot Simulator

- 按 TDD 新增 `tests/ut/moe_offload/test_policy.py` 和 `tests/ut/moe_offload/test_slot_simulator.py`，先观察到缺少 `expert_key/policy/slot_simulator` 模块的 RED。
- 新增 `vllm_ascend/moe_offload/expert_key.py`，定义 `ExpertKey(layer_id, expert_id)`。
- 新增 `vllm_ascend/moe_offload/policy.py`：
  - `LruPolicy`：驱逐 `last_used` 最小的 resident expert。
  - `StickyLayerLruPolicy`：优先保留 incoming expert 同层 resident expert，其他层按 LRU 驱逐。
  - `make_policy(name)`：支持 `lru` 与 `sticky_layer_lru`。
- 新增 `vllm_ascend/moe_offload/slot_simulator.py`：
  - `ExpertSizeTable`：默认 Qwen3-30B-A3B whole expert bytes 估计。
  - `SlotSimulator.replay(...)`：输出 total records、hit/miss、eviction、host-to-HBM bytes、estimated load ms、phase opportunity。
- 新增 `tools/sew_offload/simulate_expert_slots.py`，从 JSONL trace 读取 records，输出 `SIM_SUMMARY` 与可选 JSON 文件。
- 更新 `docs/sew-offload/04-reproduction.md` 的 offline simulator 命令，替换旧占位路径。
- 验证结果：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_trace_export.py tests/ut/moe_offload/test_trace_collector.py tests/ut/moe_offload/test_runtime_trace_only.py tests/ut/moe_offload/test_collect_moe_trace.py tests/ut/moe_offload/test_policy.py tests/ut/moe_offload/test_slot_simulator.py`：`18 passed, 4 warnings in 8.72s`。
  - `py_compile` 覆盖 MVP-B/C 新增模块、CLI 与测试：通过。
  - `git diff --check`：通过。
  - simulator CLI sample 输出 `hit_count=1`、`miss_count=3`、`eviction_count=1`、`host_to_hbm_bytes=30`。

## 2026-05-31 MVP-D Fixed Slot Correctness 底座

- 复核 `fused_moe.py`、`moe_runtime_args.py`、`moe_comm_method.py`、`moe_mlp.py` 后确认：现有 backend 以 expert id/group list 访问 `w13/w2`，并已有 `log2phy` remap 入口。
- 按 TDD 新增并通过：
  - `tests/ut/moe_offload/test_host_store.py`
  - `tests/ut/moe_offload/test_layout.py`
  - `tests/ut/moe_offload/test_slot_bank.py`
  - `tests/ut/moe_offload/test_transfer_engine.py`
  - `tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`
- 新增 `vllm_ascend/moe_offload/host_store.py`：

## 2026-06-14 P1 真实数据打穿

- 用户要求停止继续堆模拟，先用真实 Qwen3-30B-A3B 和真实 ShareGPT 数据验证 P1 框架。
- 已确认 NPU 0 空闲，模型 `/data/shared-models/Qwen3-30B-A3B` 存在，ShareGPT 原始数据 `/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json` 存在。
- 已生成真实 ShareGPT smoke manifest：`artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/sharegpt_smoke_2perbucket.jsonl`，共 6 条，覆盖 `short_chat`、`medium_chat`、`decode_heavy`，每条 `dataset=sharegpt`。
- 记录 SHA256：`docs/sew-offload/benchmark_config.yaml` 为 `54534ee7b50548361842d97164258c2aa84ee2c00492003d15d4a5d82c872e18`；manifest 为 `158d939122b4c02605d56f205cfbabe1bcadce8234849c4e3e5f0e205ca5b0ab`。
- 第一次 trace 命令使用 `collect_moe_trace.py --override-max-output-tokens 16` 失败，原因是该脚本没有该参数；改用支持输出 token override 的 `run_fixed_slot_smoke.py --mode trace_only`。
- 真实 trace-only ShareGPT short 1 request / 16 output tokens 已跑通：`artifacts/sew_offload/real_p1_20260614/trace_only_sharegpt_short_1req_16tok`，status ok，1632 条 MoE trace，output throughput `3.518 tok/s`。
- Trace analyzer 结果：grouped_dispatch 816 条，fanout min/mean/max 为 `8/19.2/128`；top grouped signature concentration 仅 `0.6%`，top-3 compute bucket coverage `1.3%`、fallback `98.7%`；analyzer 推荐 P1-RM，而不是 P1-C。
- Slot sweep 结果：`artifacts/sew_offload/real_p1_20260614/sharegpt_short_slot_sweep_1req_16tok.json`。8 slots 下 exposed miss `9914`、exposed H2D bytes `145.5GB`；128 slots 才把 exposed miss 降到 `128`，说明真实 prefill fanout 会严重冲击小 slot offload。
- 真实 compute-only P1-C probe：`artifacts/sew_offload/real_p1_20260614/p1_p1c_probe_baseline_fast_real_sharegpt_1req_16tok`。Baseline `5.177 tok/s`，P1-C fast path `4.259 tok/s`，strict token-id correctness ok；手工解析 profile JSONL 得到 gate total `816`、enabled `11`、enabled_percent `1.35%`，主要 fallback 为 `signature_not_planned` 和 prefill `phase_mismatch`。
- 注意：`run_p1_experiments.py` 当前 aggregate gate summary 只看父进程 `moe_offload_profile.events`，未正确汇总 `moe_offload_profile_jsonl_events`，因此 summary 中 `total=0` 不可信；真实 gate 数据需读 child process profile JSONL。
- 真实 fixed-slot recommended offload case 失败：`artifacts/sew_offload/real_p1_20260614/p1_offload3_real_sharegpt_1req_16tok/p1_fixed_slot_recommended`。模型 reported weight 降到 `6.2751GB`，48 层 fixed-slot register 总耗时约 `42.78s`，ledger 显示 host_store `57.98GB`、slot_bank `3.62GB`，但首次真实 prefill 在 `npu_grouped_matmul` 处失败：`weight is on cpu, different from other tensors on npu:0`。说明当前 fixed-slot/offload path 还没有保证进入 GMM 前使用 NPU slot tensor，执行路径未闭合。
  - `ExpertWeightBundle`
  - `HostExpertStore.register_layer(layer)`，按 expert 维 clone post-processed `w13/w2`。
- 新增 `vllm_ascend/moe_offload/layout.py`：
  - `LayoutSignature`
  - `validate_copy_compatible(...)`
  - `validate_backend_ready(...)`
- 新增 `vllm_ascend/moe_offload/slot_bank.py`：
  - `SlotState`
  - `ExpertSlot`
  - `ExpertSlotBank`，支持 stable slot tensors、LRU ready-slot eviction、computing slot 不驱逐。
- 新增 `vllm_ascend/moe_offload/transfer_engine.py`：
  - `TransferEngine.load_sync(...)`，copy-compatible 校验后同步 copy 并 mark ready。
- 修改 `runtime.py`：
  - 新增 `should_use_fixed_slots`。
  - 新增 `prepare_weights_for_execution(...)` fail-closed guard；在 log2phy expert-to-slot remap 完成前不接入主执行路径。
- 修改 `docs/sew-offload/09-next-steps-after-mvp-a.md`，补充 fixed-slot 不能直接替换权重维度的设计反思。

## 2026-05-31 MVP-D.2 Expert-to-Slot Remap 安全计划

- 按 TDD 新增 `tests/ut/moe_offload/test_slot_mapping.py`，先观察到缺少 `vllm_ascend.moe_offload.slot_mapping` 的 RED。
- 新增 `vllm_ascend/moe_offload/slot_mapping.py`：
  - `ExpertSlotMapping`：从 `ExpertSlotBank` 中 ready slots 构造 `logical_to_physical`，inactive expert 使用 `-1` sentinel。
  - `PreparedSlotWeights`：把稳定 slot backing tensors、`log2phy` 和 `physical_expert_count` 打包，作为后续 fused MoE 接入的单一 payload。
- 修改 `ExpertSlotBank`：
  - 新增 `w13_slots/w2_slots` 作为 `[num_slots, ...]` 稳定 backing tensors。
  - 每个 `ExpertSlot.w13/w2` 是 backing tensor 的 view，避免后续用 `torch.stack` 临时拼权重破坏 slot 地址稳定性。
- 修改 `HostExpertStore.register_layer(...)`，显式将 expert bundle clone 到 CPU host store；这更符合 offload 语义，但当前仍是 correctness 原型，不代表已经释放原始 full expert HBM。
- 修改 `MoeOffloadRuntime`：
  - 新增 `register_layer_for_fixed_slots(layer, slot_device=...)`。
  - 新增 `prepare_fixed_slot_plan(...)`，执行 active working set 去重、slot budget 早失败、同步 miss load、构造 `PreparedSlotWeights`。
  - 保留 `prepare_weights_for_execution(...)` fail-closed，不把 fixed slot 接入 `fused_moe.py` 主路径。
- 按 TDD 扩展 `tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`，先观察到 runtime 缺少 fixed-slot plan API 的 RED，再实现 GREEN。
- 新增 fused MoE routing contract：
  - `MoERoutingParams.physical_expert_count`。
  - `build_fused_experts_input(... physical_expert_count=...)`。
  - `TokenDispatcherWithAllGather` 在该字段存在时把 dispatch `expert_num/active_expert_range` 切到 physical slot space。
- 自我反思后修正设计：单独 `log2phy[topk_ids]` 不足以安全接入 slot weights。dispatcher 的 `expert_num/group_list` 也必须使用 physical slot count，否则会出现 topk 已 remap 但 grouped matmul metadata 仍处于 logical expert space 的语义错配。
- 当前完成的是“safe fixed-slot plan + AllGather slot-dispatch metadata 契约”，仍未把 `PreparedSlotWeights` 接入 `AscendUnquantizedFusedMoEMethod.apply()` 的主执行路径。
- 已验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_slot_mapping.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_preserves_runtime_semantics tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather::test_token_dispatch_uses_physical_expert_count_for_slot_routing`：`10 passed, 4 warnings, 6 subtests passed in 0.07s`。

## 2026-05-31 MVP-D.3 Narrow Fixed-Slot Apply Wiring

- 按 TDD 新增 `tests/ut/moe_offload/test_fixed_slot_apply.py`，先观察到 `AscendUnquantizedFusedMoEMethod.apply()` 仍把原始 `[num_experts, ...]` 权重传给 backend 的 RED。
- 修改 `vllm_ascend/ops/fused_moe/fused_moe.py`：
  - 在 `process_weights_after_loading()` 末尾，如果 fixed-slot 开启则注册 layer 到 `MoeOffloadRuntime`。
  - 在 `apply()` 中复用同一个 runtime；fixed-slot 开启时准备 `PreparedSlotWeights` 并把 `w1/w2/log2phy/physical_expert_count` 传入 `build_fused_experts_input(...)`。
  - 保留窄路径 hard gates：仅 AllGather、无 `expert_map`、无 redundant experts、无 bias、无 force load balance。
  - apply 内保留 lazy register fallback，避免测试或异常生命周期中没有调用 `process_weights_after_loading()` 时直接漏注册。
- 修改 `runtime.py`，新增 `is_layer_registered(layer_id)` 供 fused MoE hook 做 lazy registration 判断。
- 新增负向 UT，确认 fixed-slot 开启后 MC2、expert_map、redundant experts、bias、force load balance 均在 backend 前 `NotImplementedError`，不会静默进入错误路径。
- 追加负向 UT，确认 zero-expert 路径也在 `zero_experts_compute(...)` 前 fail closed；该分支会改写 expert index/scale，当前 fixed-slot working set 不能安全覆盖。
- 自我反思限制：当前 active expert 提取使用 `torch.unique(topk_ids.detach().cpu())`，会产生 NPU->CPU 同步；这在 MVP-D 同步 correctness 原型可接受，但必须在 MVP-E async/hot path 前替换为更合适的 device-side 或 trace-derived working set 机制。
- 当前主路径接入仍只覆盖 mock backend correctness；尚未运行真实 Qwen3-30B-A3B NPU smoke，因此不能声称真实 `npu_grouped_matmul` 已通过。

## 2026-06-01 MVP-D.4 Reverse Compatibility and Smoke Gate

- 按用户要求反向检查“不要影响当前系统其他功能和执行逻辑”，新增默认路径回归测试：
  - `test_default_apply_preserves_original_weights_and_routing`
  - `test_build_fused_experts_input_defaults_to_logical_expert_routing`
  - `test_token_dispatch_defaults_to_local_logical_expert_count`
- 按 TDD 新增 backend-ready RED：`PreparedSlotWeights` 缺少 backend-ready 校验；实现 `validate_backend_ready(expected_device_type=...)` 后 GREEN。
- 修改 `vllm_ascend/moe_offload/slot_mapping.py`：
  - `PreparedSlotWeights.validate_backend_ready(...)` 检查 `physical_expert_count`、`w1/w2` 第 0 维和 backend device type。
- 修改 `vllm_ascend/ops/fused_moe/fused_moe.py`：
  - fixed-slot path 在传给 backend 前调用 `prepared_weights.validate_backend_ready(expected_device_type=x.device.type)`。
  - 默认路径不调用该校验，不改变原始 weights/routing。
- 新增 `tests/ut/moe_offload/test_fixed_slot_smoke.py` 和 `tools/sew_offload/run_fixed_slot_smoke.py`：
  - 支持 prepare synthetic smoke manifest。
  - 支持 fixed-slot sync smoke，输出 `summary.json`。
  - UT mock `LLM.generate`，验证 env、LLM 参数、summary artifact。
- 更新 `docs/sew-offload/04-reproduction.md` 的 9.2 sync-load baseline，加入 fixed-slot smoke prepare/run 命令与 failure interpretation。
- 验证结果：
  - RED 1：`test_fixed_slot_apply_rejects_backend_device_mismatch` 初始失败，原因 `PreparedSlotWeights` 缺少 `validate_backend_ready`。
  - GREEN：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_slot_mapping.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/ops/test_moe_runtime_args.py::TestMoERuntimeArgs::test_build_fused_experts_input_defaults_to_logical_expert_routing tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather::test_token_dispatch_defaults_to_local_logical_expert_count tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather::test_token_dispatch_uses_physical_expert_count_for_slot_routing`：`17 passed, 4 warnings in 0.13s`。
  - 扩大验证：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`61 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.94s`。
  - fixed-slot smoke UT：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_fixed_slot_smoke.py`：`2 passed, 4 warnings in 0.30s`。
  - prepare-only smoke：`/root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/run_fixed_slot_smoke.py --output-dir /tmp/sew_fixed_slot_smoke_prepare --manifest /tmp/sew_fixed_slot_smoke_requests.jsonl --prepare-smoke-manifest --prepare-only --buckets short_chat --smoke-requests-per-bucket 1`：输出 `PREPARE_OK manifest=/tmp/sew_fixed_slot_smoke_requests.jsonl`。
- 未执行真实 Qwen3-30B-A3B NPU smoke；下一步需要选择空闲 NPU 运行 `docs/sew-offload/04-reproduction.md` 中 fixed-slot sync smoke 命令。

## 2026-06-01 MVP-D.4 profile force-load-balance 修正

- 复现当前失败：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_fixed_slot_apply.py` 初始为 `2 failed, 9 passed`，失败均为 `ValueError: w13 backend device mismatch: npu != cpu`。
- 根因分析：测试中的 hidden states 在 CPU；而 `_fixed_slot_device_for_processed_weight()` 对 CPU/offloaded expert weight 选择当前 NPU slot，这是为了真实 prefetch offload 场景中把 host expert 装载到 NPU grouped MoE backend。因此不能为了 CPU mock 测试把实现改成 CPU slot。
- 修正：在两个 CPU mock 正向测试中显式 `register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))`，让测试的 backend-ready 期望和 mock backend device 一致。
- 同时保留 profile dummy routing 设计：fixed-slot profile run 使用 `_build_fixed_slot_profile_topk_ids(...)` 将 dummy active experts 限制到 slot budget，并在 `top_k > num_slots` 时早失败；真实请求路径不改 router/top-k。
- 局部验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_fixed_slot_apply.py`：`11 passed, 4 warnings in 0.12s`。
- 扩大验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`67 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.94s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/slot_mapping.py vllm_ascend/moe_offload/runtime.py vllm_ascend/ops/fused_moe/fused_moe.py tools/sew_offload/run_fixed_slot_smoke.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_fixed_slot_smoke.py`：通过。
  - `git diff --check`：通过。
- 代码级反向审查：`physical_expert_count` 默认 `None`，`build_fused_experts_input(...)` 默认 logical routing 不变；`TokenDispatcherWithAllGather` 只有显式 physical count 时进入 slot-space，且拒绝 expert_map/redundant experts；fixed-slot profile 改写仅在 `enable_force_load_balance and should_use_fixed_slots` 中发生。

## 2026-06-01 真实 NPU smoke 继续推进与 PrefetchOffloader 生命周期修正

- 真实 NPU smoke 命令：
  - `ASCEND_RT_VISIBLE_DEVICES=4 /root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/run_fixed_slot_smoke.py --output-dir artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_profilefix_slots8_inline_prefetch --inline-prompt 'Hello' --inline-max-output-tokens 1 --num-slots 8 --offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params experts --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 --kv-cache-memory-mb 512`
- 结果：失败，但已越过上一轮 `NotImplementedError: MoE offload fixed slots do not support force load balance yet`；日志显示 `Offloader set to PrefetchOffloader`、`Loading model weights took 46.7751 GB`。
- 新失败点：profile run 中现有 `PrefetchOffloader` 的 `start_onload_to_static()` 触发 `AssertionError: Buffer pool not assigned`；summary 写入 `artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_profilefix_slots8_inline_prefetch/summary.json`。
- 根因：Ascend `NPUModelRunner.load_model()` 重写了上游 GPU runner 加载流程，但漏掉上游加载末尾的 `get_offloader().post_init()`；现有 prefetch forward hook 被安装，却没有分配 `StaticBufferPool` 和 static parameter buffers。
- 按 TDD 新增 worker 生命周期测试：
  - RED：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py::TestNPUModelRunnerOffloaderLifecycle::test_load_model_post_inits_weight_offloader` 初始失败，`post_init` 调用次数为 0。
  - GREEN：在 `vllm_ascend/worker/model_runner_v1.py` 导入 `get_offloader`，并在可选 ACLGraph wrapper 后调用 `get_offloader().post_init()`；同一测试变为 `1 passed, 4 warnings in 0.18s`。
- 扩大验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`73 passed, 8 skipped, 4 warnings, 105 subtests passed in 9.03s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/worker/model_runner_v1.py vllm_ascend/moe_offload/slot_mapping.py vllm_ascend/moe_offload/runtime.py vllm_ascend/ops/fused_moe/fused_moe.py tools/sew_offload/run_fixed_slot_smoke.py tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_fixed_slot_smoke.py`：通过。
  - `git diff --check`：通过。
- 自我反思：该修正是补齐现有 vLLM weight prefetch 生命周期，不属于 SEW fixed-slot 特化。默认 offload 关闭时 `NoopOffloader.post_init()` 是空操作，风险面较小；开启 prefetch 时这是正确性前提。

## 2026-06-01 CUDA-to-NPU wrapper Event 修正

- 重跑真实 NPU smoke：
  - artifact log：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_offloaderpostinit_slots8_inline_prefetch.log`
  - summary：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_offloaderpostinit_slots8_inline_prefetch/summary.json`
- 结果：失败，但已越过 `Buffer pool not assigned`；日志显示 `Loading model weights took 46.7751 GB` 和 `[PrefetchOffloader] Initialized 12 modules. Total GPU memory saved: 14.4955 GB, Static buffer pool: 1.2080 GB`。
- 新失败点：`torch.cuda.current_stream().record_event(fork_event)` 调到 NPU stream 后，`event.record(self)` 触发 `TypeError: _EventPlaceholder...<lambda>() takes 0 positional arguments but 1 was given`。
- 根因：`_torch_cuda_wrapper()` 的 finally 中把 `torch.cuda.current_stream/default_stream/stream` 保持映射到 `torch.npu`，但把 `torch.cuda.Event` 设成 placeholder，导致 NPU stream 与 placeholder event API 不兼容。
- 按 TDD 新增 `TestTorchCudaWrapper.test_cuda_event_remains_npu_event_after_wrapper_exit`：
  - RED：初始失败，`torch.cuda.Event` 为 `_EventPlaceholder` 而不是 `torch.npu.Event`。
  - GREEN：把 `_torch_cuda_wrapper()` finally 中的 `torch.cuda.Event` 改为 `torch.npu.Event`；targeted tests `2 passed, 4 warnings in 0.18s`。
- 扩大验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`74 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.96s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/worker/model_runner_v1.py vllm_ascend/moe_offload/slot_mapping.py vllm_ascend/moe_offload/runtime.py vllm_ascend/ops/fused_moe/fused_moe.py tools/sew_offload/run_fixed_slot_smoke.py tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_fixed_slot_smoke.py`：通过。
  - `git diff --check`：通过。
- 自我反思：这仍是现有 prefetch offload 的 Ascend 适配问题，不应算作 SEW fixed-slot 的 correctness 证明；但它是组合路径继续往 MoE backend 推进的必要前置条件。

## 2026-06-01 fixed-slot 越界 expert id 诊断开始

- 恢复计划与 smoke 失败证据：最新真实 NPU fixed-slot+prefetch 已进入 `llm.generate()`，失败于 `MoeOffloadRuntime.prepare_fixed_slot_plan()` 中 `HostExpertStore.get(layer_id=0, expert_id=266)`；Qwen3-30B-A3B 配置只有 128 个 experts，因此 266 必然是非法 routed expert id。
- 按 TDD 增加 fixed-slot 内部安全护栏：
  - RED：`test_runtime_rejects_out_of_range_active_expert_before_host_lookup` 初始失败，当前实现会到 `HostExpertStore.get()` 后抛 `KeyError`。
  - GREEN：`prepare_fixed_slot_plan()` 在 host store lookup 前调用 `_validate_active_expert_ids(...)`，非法 id 抛 `ValueError`，包含 `layer_id`、`num_logical_experts` 和 `expert_ids`。
  - 局部验证：`/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_runtime_fixed_slot_guard.py::test_runtime_rejects_out_of_range_active_expert_before_host_lookup`：`1 passed, 4 warnings`。
- 自我反思：这个护栏不是根因修复，也不会 clamp/drop/remap 非法 expert；它只把固定槽路径的错误提前并显式化，防止继续把非法 id 当成 host expert key 访问。默认 offload 关闭路径不调用 `prepare_fixed_slot_plan()`，因此不改变现有 MoE 执行逻辑。
- 根因线索更新：阅读上游 `MoERunner._forward_impl()` 后发现，上游 runner 在 `self.gate is not None` 时会先执行 `router_logits, _ = self.gate(hidden_states)`；当前 Ascend `AscendFusedMoE.forward_impl()` 直接把传入的 `router_logits` 交给 `quant_method.apply()`。Qwen3Moe 在 internal-router 路径传入的是 `router_logits=hidden_states`，hidden dim 为 2048；这可以解释为什么会出现 266 这种“小于 2048 但大于 127”的 id。下一步先用回归测试固定 Ascend runner 必须对齐上游 gate 生命周期，再做最小修复。

## 2026-06-01 internal-router gate 生命周期修正与真实 smoke 通过

- 第一轮假设：非法 expert id `[266, 362, 366, 621, 823, 952, 1465, 2003]` 形态像是从 Qwen3 hidden size 2048 维中取出的 top-k index，而不是从 128 experts 中取出的 id；因此怀疑 internal-router gate 未执行。
- 中间尝试：最初把 gate 对齐放在 `AscendFusedMoE.forward_impl()`，targeted UT 可转绿，但真实 smoke `artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_gatefix_slots8_inline_prefetch.log` 仍失败，护栏报 `fixed-slot active expert id out of range`，说明修复没有覆盖真实调用栈。
- 根因校正：真实栈是 custom op -> `layer.runner._forward_impl()` -> `AscendMoERunner.forward_impl()` -> `layer.forward_impl(...)`。因此 gate 生命周期应对齐上游 `MoERunner._forward_impl()`，修在 `AscendMoERunner._forward_impl()`，不是修在 `AscendFusedMoE.forward_impl()`。
- 按 TDD 重写回归测试 `test_ascend_moe_runner_runs_internal_gate_before_layer_forward`：
  - RED：`AscendMoERunner._forward_impl()` 未调用 `self.gate(hidden_states)`，`layer.forward_impl(...)` 收到占位 `router_logits=hidden_states`。
  - GREEN：在 `AscendMoERunner._forward_impl()` 进入 sequence-parallel context 前执行 `router_logits, _ = self.gate(hidden_states)`，再把 gate logits 传入 `forward_impl(...)`。
- 设计反思：这是恢复上游 runner 的 internal-router 语义，不是 SEW-only workaround；它解释了为什么越界 id 都小于 hidden size 2048。该修复应有利于默认 Qwen3 MoE 路径，但仍需后续更广泛既有 MoE 测试覆盖。
- 验证结果：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_fixed_slot_apply.py::test_ascend_moe_runner_runs_internal_gate_before_layer_forward`：`1 passed, 4 warnings`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_apply.py`：`17 passed, 4 warnings`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/runtime.py vllm_ascend/ops/fused_moe/fused_moe.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_apply.py`：通过。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`76 passed, 8 skipped, 4 warnings, 105 subtests passed`。
- 真实 NPU smoke：
  - 命令：`ASCEND_RT_VISIBLE_DEVICES=4 /root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/run_fixed_slot_smoke.py --output-dir artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_runnergate_slots8_inline_prefetch --inline-prompt 'Hello' --inline-max-output-tokens 1 --num-slots 8 --offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params experts --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 --kv-cache-memory-mb 512`
  - 结果：`FIXED_SLOT_SMOKE_SUMMARY ... "status": "ok"`，`completed=1`，`total_output_tokens=1`，`load_seconds=125.49`，`duration_s=1.77`。
  - 日志显示：`Loading model weights took 46.7751 GB`；`[PrefetchOffloader] Initialized 12 modules. Total GPU memory saved: 14.4955 GB, Static buffer pool: 1.2080 GB`；engine profile、KV cache 创建和 1-token generate 均完成。
- 结论：Qwen3-30B-A3B 单卡 fixed-slot sync smoke 已越过 profile dummy run、PrefetchOffloader、fixed-slot plan、slot remap、AllGather slot dispatch 和 grouped MoE backend 的最小闭环。
- 边界声明：这仍是 MVP-D correctness smoke，不是性能结果；当前 HBM 节省主要来自现有 vLLM prefetch backend，SEW fixed-slot 目前还没有释放/替换原始 full expert 参数，不能声称已实现 SEW 自身的 13.5GB HBM saving。

## 2026-06-01 MVP-D.5 correctness 对照与默认路径反向检查

- 按用户要求继续反向检查“不要影响当前系统其他功能和执行逻辑”，先审查当前 diff 边界：
  - fixed-slot 主路径仍由 `should_use_fixed_slots` 控制，默认关闭时 `w13/w2/log2phy/physical_expert_count` 保持原路径。
  - `model_runner_v1.py` 的 `get_offloader().post_init()` 和 `torch.cuda.Event = torch.npu.Event` 是对现有 prefetch offloader 生命周期/API family 的 Ascend 兼容补齐，不是 SEW-only 分支；已有 worker UT 覆盖。
  - no-offload smoke 工具现在清理全部 SEW env，而不是设置 `"0"`，避免默认路径出现 unknown-env warning。
- 按 TDD 扩展工具：
  - RED：`test_fixed_slot_smoke.py` 先失败于缺少 `configure_sew_offload_env/run_smoke`。
  - GREEN：`tools/sew_offload/run_fixed_slot_smoke.py` 支持 `--mode no_offload|trace_only|fixed_slot_sync`，并写出 `outputs.jsonl`。
  - RED：`test_compare_smoke_outputs.py` 先失败于缺少 `tools.sew_offload.compare_smoke_outputs`。
  - GREEN：新增 `compare_smoke_outputs.py`，严格比较 token id；重复 `request_id` 直接拒绝。
  - 反思修正：最初 no-offload 仅把 env 设为 `"0"`，真实 baseline 会出现 vLLM unknown-env warning；改为清理所有 `VLLM_ASCEND_MOE_OFFLOAD_*` env，并补测试覆盖 `POLICY/TRACE_MAX_RECORDS` 残留。
- 文档更新：`docs/sew-offload/04-reproduction.md` 加入独立进程 no-offload baseline、fixed-slot candidate 与 `compare_smoke_outputs.py` 的 correctness 对照命令。
- 局部验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_compare_smoke_outputs.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`：`26 passed, 4 warnings`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile tools/sew_offload/run_fixed_slot_smoke.py tools/sew_offload/compare_smoke_outputs.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_compare_smoke_outputs.py`：通过。
  - `git diff --check`：通过。
- 扩大默认路径/兼容性验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`82 passed, 8 skipped, 4 warnings, 105 subtests passed in 9.03s`。
- 真实 NPU correctness 对照：
  - no-offload clean-env baseline 命令：`ASCEND_RT_VISIBLE_DEVICES=4 /root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/run_fixed_slot_smoke.py --mode no_offload --output-dir artifacts/sew_offload/runs/no_offload_smoke_20260601_inline_1tok_cleanenv --inline-prompt 'Hello' --inline-max-output-tokens 1 --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 --kv-cache-memory-mb 512`
  - baseline 结果：状态 `ok`，`load_seconds=33.4295`，日志显示 `Loading model weights took 56.9001 GB`，输出 `{"output_token_ids":[353]}`。
  - fixed-slot candidate 命令：`ASCEND_RT_VISIBLE_DEVICES=4 /root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/run_fixed_slot_smoke.py --mode fixed_slot_sync --output-dir artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare --inline-prompt 'Hello' --inline-max-output-tokens 1 --num-slots 8 --offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params experts --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 --kv-cache-memory-mb 512`
  - candidate 结果：状态 `ok`，`load_seconds=137.3632`，日志显示 `Loading model weights took 46.7751 GB`、`PrefetchOffloader` static buffer pool `1.2080 GB`，输出 `{"output_token_ids":[353]}`。
  - strict compare 命令：`/root/miniconda3/envs/vllm-hust-dev/bin/python tools/sew_offload/compare_smoke_outputs.py --baseline artifacts/sew_offload/runs/no_offload_smoke_20260601_inline_1tok_cleanenv/outputs.jsonl --candidate artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare/outputs.jsonl --output artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare/correctness_compare.json`
  - compare 结果：`{"status":"ok","matched":1,"mismatched":0,"missing":0,"extra":0}`。
- 自我反思：
  - 该对照只证明 1-token 最小输出一致性，不是性能结果。
  - 不能声称 SEW 已节省 HBM；candidate 的 HBM 降低仍主要来自现有 vLLM prefetch backend。
  - 下一步更合理的是多 prompt/更长 output smoke，然后审查如何安全释放/替换原始 full expert 参数。

## 2026-06-01 MVP-D.5 多 prompt smoke 容量守卫反查

- 继续推进 2 prompt × 8 token correctness smoke：
  - prompt 文件：`artifacts/sew_offload/runs/fixed_slot_correctness_prompts_2prompt_8tok.jsonl`
  - no-offload baseline：`artifacts/sew_offload/runs/no_offload_smoke_20260601_2prompt_8tok/summary.json`
  - baseline 结果：状态 `ok`，`completed=2`，`total_output_tokens=16`，`load_seconds=33.6329`；`outputs.jsonl` 两条输出分别为 8 token。
  - fixed-slot candidate：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2prompt_8tok/summary.json`
  - candidate 结果：状态 `failed`，外层为 `EngineDeadError`。
- 根因定位：
  - 真实底层症状为 `RuntimeError: active expert working set size 46 exceeds num_slots=8`。
  - 这发生在第二个较长 prompt 的 prefill 阶段；单个 prefill 内多个 token 的 top-k expert 并集超过了 8 个 fixed slots。
- 设计反思：
  - 这是预期的 fail-closed 行为，不是 correctness mismatch。
  - 当前 MVP-D 的 `num_slots=8` 表示“同一层同一次 grouped MoE 调用可承载的 active expert working set 上限”，不能用 clamp/drop/remap active experts 来掩盖容量不足，否则会改变 router/top-k 语义。
  - 直接把 `num_slots` 提到 64 并不一定是正确下一步；当前实现是每层一个 slot bank，`64 slots × 48 layers` 会引入大量额外 HBM 副本，可能把 correctness 原型推向 OOM。更合理的下一步是先跑 slot-budget-compatible 的多请求/更长 decode smoke，再单独设计 full expert 参数释放/替换或全局 slot 生命周期。

## 2026-06-01 MVP-D.5 2 short prompt × 8 token correctness 通过

- 为避免把长 prefill working-set 压力与 decode correctness 混在一起，新增更保守 prompt 文件：
  - `artifacts/sew_offload/runs/fixed_slot_correctness_prompts_2short_8tok.jsonl`
  - 内容为 `Hello` 与 `Hi` 两条短 prompt，每条 `max_output_tokens=8`。
- no-offload baseline：
  - 命令使用 `--mode no_offload`，不传 native offload kwargs，并清理 SEW env。
  - artifact：`artifacts/sew_offload/runs/no_offload_smoke_20260601_2short_8tok/summary.json`
  - 结果：状态 `ok`，`completed=2`，`total_output_tokens=16`，`load_seconds=33.9988`，日志显示 `Loading model weights took 56.9001 GB`。
- fixed-slot candidate：
  - 命令使用 `--mode fixed_slot_sync --num-slots 8`，组合现有 `prefetch` offloader。
  - artifact：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/summary.json`
  - 结果：状态 `ok`，`completed=2`，`total_output_tokens=16`，`load_seconds=137.6875`，日志显示 `Loading model weights took 46.7751 GB` 和 `PrefetchOffloader` static buffer pool `1.2080 GB`。
- strict token-id compare：
  - artifact：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/correctness_compare.json`
  - 结果：`status=ok`，`matched=2`，`mismatched=0`，`missing=0`，`extra=0`。
- 自我反思：
  - 这把 correctness 证据从 1-token 单请求推进到 2 请求 × 8-token decode，但仍是窄 smoke，不是性能结论。
  - candidate 的吞吐显著低于 baseline，主要来自同步 load、host store clone、每层 slot bank 和现有 prefetch 生命周期成本；这再次说明 MVP-D 是正确性原型，不能用于论文性能数据。
  - candidate 日志中 vLLM 会报告 `VLLM_ASCEND_MOE_OFFLOAD_*` unknown-env warning；baseline clean-env 不受影响。后续可考虑把 SEW env 加入 vLLM allowed env 列表以降低日志噪音，但这不是当前 correctness 阻塞。
- 本轮验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_compare_smoke_outputs.py`：`11 passed, 4 warnings in 0.32s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`84 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.78s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile tools/sew_offload/run_fixed_slot_smoke.py tools/sew_offload/compare_smoke_outputs.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_compare_smoke_outputs.py vllm_ascend/moe_offload/runtime.py vllm_ascend/ops/fused_moe/fused_moe.py vllm_ascend/worker/model_runner_v1.py`：通过。
  - `git diff --check`：通过。

## 2026-06-01 MVP-D.6 fixed-slot memory ledger

- 按当前计划进入“释放/替换原始 expert 参数前的生命周期审查”，但没有直接释放参数；原因是这会改变模型参数所有权、vLLM weight loader/offloader 引用关系和默认路径行为。
- 按 TDD 增加只读账本：
  - RED：`test_runtime_reports_fixed_slot_memory_ledger_without_releasing_original_weights` 初始失败，`MoeOffloadRuntime` 缺少 `memory_ledger()`。
  - GREEN：新增 `MoeOffloadMemoryLedger`、`MoeOffloadRuntime.memory_ledger()`、`HostExpertStore.total_bytes`、`ExpertSlotBank.total_bytes`。
  - 账本记录 `original_expert_weight_bytes`、`host_store_bytes`、`slot_bank_bytes`，并显式报告 `original_expert_weights_retained`。
- 新增离线估算工具：
  - `tools/sew_offload/estimate_fixed_slot_memory.py`
  - UT：`tests/ut/moe_offload/test_estimate_fixed_slot_memory.py`
  - 工具不加载模型、不触碰 NPU，只按 `num_layers × num_experts × expert_bytes` 和 `num_layers × num_slots × expert_bytes` 做账本估算。
- 实测估算 artifact：
  - `artifacts/sew_offload/runs/fixed_slot_memory_estimate_qwen3_30b_slots8.json`
  - `artifacts/sew_offload/runs/fixed_slot_memory_estimate_qwen3_30b_slots64.json`
  - `artifacts/sew_offload/runs/fixed_slot_memory_estimate_qwen3_30b_slots8_released_original.json`
- 估算结果：
  - Qwen3-30B-A3B 默认参数下，`num_slots=8` slot bank 约 `5.64 GB`，当前原型 total managed 约 `186.03 GB`。
  - `num_slots=64` slot bank 约 `45.10 GB`，当前原型 total managed 约 `225.49 GB`。
  - 假设释放原始 expert 参数后，`num_slots=8` 的 host store + slot bank 仍约 `95.83 GB`。
- 自我反思：
  - 账本确认了上一轮不盲目跑 `num_slots=64` 的判断是正确的；当前 per-layer slot bank 设计下，大 slot budget 会非常昂贵。
  - 当前修改是 introspection，不改变推理路径，不释放参数，不声明 HBM saving。
  - 下一步应设计参数所有权转移：先证明 `HostExpertStore` 已完整持有 post-processed layout，再决定如何把原 layer expert 参数替换为非执行占位或受控 offloaded representation，并保持默认 offload 关闭路径不受影响。
- 本轮验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_host_store.py tests/ut/moe_offload/test_slot_bank.py tests/ut/moe_offload/test_estimate_fixed_slot_memory.py`：`16 passed, 4 warnings in 0.15s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`89 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.88s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/runtime.py vllm_ascend/moe_offload/host_store.py vllm_ascend/moe_offload/slot_bank.py vllm_ascend/moe_offload/__init__.py tools/sew_offload/estimate_fixed_slot_memory.py tests/ut/moe_offload/test_estimate_fixed_slot_memory.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_host_store.py tests/ut/moe_offload/test_slot_bank.py`：通过。
  - `git diff --check`：通过。
  - 对本轮新增未跟踪文件 `tools/sew_offload/estimate_fixed_slot_memory.py` 与 `tests/ut/moe_offload/test_estimate_fixed_slot_memory.py` 额外运行 `git diff --check --no-index /dev/null ...`：通过。

## 2026-06-01 MVP-D.7 original expert release readiness guard

- 继续按照“先证明安全，再释放参数”的顺序推进，没有直接释放或替换 `layer.w13_weight/layer.w2_weight`。
- 按 TDD 新增 release readiness guard：
  - RED：`test_runtime_release_readiness_rejects_current_correctness_prototype` 和 `test_runtime_release_readiness_accepts_explicit_safe_preconditions` 初始失败，`MoeOffloadRuntime` 缺少 `plan_original_weight_release(...)`。
  - GREEN：新增 `MoeExpertReleasePlan` 和 `MoeOffloadRuntime.plan_original_weight_release(...)`。
  - 追加缺层 guard：`test_runtime_release_readiness_rejects_missing_registered_layers`，确保目标层没有全部注册时不会误报 ready。
- 当前 guard 语义：
  - 默认会因 `default_path_not_preserved`、`host_store_not_marked_complete`、`original_expert_weights_still_retained` fail closed。
  - 只有调用者显式证明默认路径已保留、host store 已完整，并在 planning 阶段允许 retained original weights 时，才返回 `ready=True`。
  - 即使上述条件满足，若 `expected_layer_ids` 中存在未注册层，仍返回 `layers_not_registered:[...]`。
- 自我反思：
  - 这不是释放实现，只是 release plan/readiness gate；它的价值是把后续危险操作的前置条件代码化。
  - `host_store_is_complete` 目前仍是人工布尔值，不够强；下一步应让 runtime 自己按 expected layer/expert count 和 layout signature 验证 host store completeness。
  - 默认路径仍不调用该接口，因此当前系统其他功能和执行逻辑不应受影响。
- 本轮验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`：`9 passed, 4 warnings in 0.05s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`92 passed, 8 skipped, 4 warnings, 105 subtests passed in 9.11s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/runtime.py vllm_ascend/moe_offload/__init__.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`：通过。
  - `git diff --check`：通过。
  - 对本开发线新增未跟踪文件 `tools/sew_offload/estimate_fixed_slot_memory.py` 与 `tests/ut/moe_offload/test_estimate_fixed_slot_memory.py` 额外运行 `git diff --check --no-index /dev/null ...`：通过。

## 2026-06-01 MVP-D.8 host store completeness self-check

- 继续沿着“释放前先证明”的路线推进，没有释放或替换任何 `layer.w13_weight/layer.w2_weight`，也没有改变默认推理路径。
- 按 TDD 将 `host_store_is_complete` 从人工布尔条件升级为 runtime 自检：
  - RED：新增 `validate_complete_layers(...)` 相关测试，初始失败于 `HostExpertStore` 没有 completeness API，`plan_original_weight_release(...)` 仍强制要求人工 `host_store_is_complete`。
  - GREEN：新增 `HostExpertLayerSignature`、`HostStoreCompletenessReport`、`HostExpertStore.validate_complete_layers(...)`，并让 `plan_original_weight_release(...)` 默认调用自检。
  - 追加 RED/GREEN：补上 stride mismatch、空 expert layer、非 CPU host bundle 的 fail-closed 测试。
- 当前自检语义：
  - 检查 expected layer 是否都已注册。
  - 检查每个注册 layer 的 expected expert id 是否都有 host bundle。
  - 检查 host bundle 的 shape、dtype、stride 与注册时 post-processed expert layout 一致。
  - 检查 host bundle 位于 CPU，避免把 device tensor 误当作 host store。
  - `host_store_is_complete=False` 仍可作为向后兼容的额外保守 blocker；省略该参数时默认走 runtime 自检。
- 自我反思：
  - 这一步只强化 readiness guard，不接入 `prepare_fixed_slot_plan()`，因此不会影响当前 fixed-slot execution 或默认 no-offload 路径。
  - stride/device 纳入自检是必要的，因为 Ascend grouped MoE 的 backend-ready layout 不能只看 shape/dtype。
  - 通过该 guard 仍不代表可以直接声称 HBM saving；它只是后续参数所有权转移的入口条件。
- 本轮已运行：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_host_store.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`：`20 passed, 4 warnings in 0.07s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/worker/test_model_runner_v1.py tests/ut/moe_offload tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather tests/ut/test_envs.py`：`99 passed, 8 skipped, 4 warnings, 105 subtests passed in 8.84s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/host_store.py vllm_ascend/moe_offload/runtime.py vllm_ascend/moe_offload/__init__.py tools/sew_offload/estimate_fixed_slot_memory.py tests/ut/moe_offload/test_host_store.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_estimate_fixed_slot_memory.py`：通过。
  - `git diff --check`：通过。
  - 对未跟踪 SEW 文件运行 `git diff --check --no-index /dev/null ...`：通过。

## 2026-06-01 MVP-D.9 分层驻留与 partial release（代码）

- 按 `sew_offload_差距与_mvp-d.9` 计划执行 **d9_first**；同步更新 `task_plan.md`（阶段 12）、`findings.md`（差距表与反思）。
- 新增 `vllm_ascend/moe_offload/tiered_residency.py`、`expert_weight_release.py`。
- `config.py` / `envs.py`：`VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS`、`VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS`（默认关）。
- `runtime.py`：`is_resident_layer`、`should_use_fixed_slot_plan_for_layer`、`release_original_expert_weights_if_ready`；ledger 排除已 release 层字节。
- `fused_moe.py`：仅 non-resident 层 register / prepare_fixed_slot；post-load 可选 release。
- `estimate_fixed_slot_memory.py`：`compare_slot_budget_models`（per-layer vs global + 13.5GB budget 布尔）。
- 新增 UT：`tests/ut/moe_offload/test_tiered_residency_d9.py`。
- 验证：`pytest -q tests/ut/moe_offload/test_tiered_residency_d9.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_estimate_fixed_slot_memory.py` → 17 passed；`pytest -q tests/ut/moe_offload tests/ut/worker/test_model_runner_v1.py` → 92 passed, 1 skipped。
- **自我反思**：resident 层不 register 避免重复 host clone；global slot pool 仅估算未实现；release 后仍需 NPU 证明 vLLM 驻留权重 GB 下降且 token-id 不变。
- **未完成（计划 todo）**：真实 `collect_moe_trace` JSONL；`release=1` NPU smoke + ledger/HBM 对比。

## 2026-06-01 MVP-D.9 实现与验证（运行结果）

### 单元 / 集成测试（已跑）

- `pytest tests/ut/moe_offload tests/ut/worker/test_model_runner_v1.py tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_token_dispatcher.py::TestTokenDispatcherWithAllGather`：**102 passed**, 8 skipped。
- `pytest tests/ut/moe_offload/test_tiered_residency_d9.py -v`：**5 passed**（resident 层、release ledger、release 默认关、容量模型）。
- `py_compile` + `estimate_fixed_slot_memory.py --num-slots 8`：通过。

### 真实 NPU（NPU 4，`ASCEND_RT_VISIBLE_DEVICES=4`）

| 用例 | 结果 | 证据 |
| --- | --- | --- |
| `--mode no_offload`（D.9 代码树，清理 SEW env） | **ok** | `artifacts/sew_offload/runs/d9_verify_20260601/no_offload/summary.json`，`load_seconds≈39.6`，`output_token_ids=[353]` |
| `--mode fixed_slot_sync` release=0（本次会话） | **未完成** | 权重加载后日志停在 `Loading weights took 136s`；EngineCore **高 CPU 长时间无新日志**（疑似 `process_weights` 中 48 层 host clone + slot bank 初始化）；已 kill，避免占满 NPU |
| 历史 fixed-slot vs **本次** no_offload strict compare | **ok** | `compare_hist.json`：`matched=1`，token id 一致（说明默认 no-offload 路径仍与 D.5 fixed-slot 对照一致） |
| `release=1` NPU smoke | **未跑** | 需先解决/度量 post-load 耗时或缩小层数后再测 HBM |

### 反思（验证驱动）

1. **D.9 未改 release=0 的推理语义**：UT + 本次 no_offload token 与历史 fixed-slot 对照一致。
2. **不能声称 release=1 已在 NPU 验证**：partial release 仅 UT 覆盖。
3. **fixed-slot 全模型 post-load 成本仍是瓶颈**：与 D 阶段相同，D.9 的 register（host store）在 48 层上可能极慢；下一步应测「resident 层跳过 register」是否缩短加载，或 lazy register 单层。
4. **timeout 600s 可能仍不够**：若需完整 fixed-slot smoke，建议单独跑并记录 `process_weights` 分段耗时。

## 2026-06-01 GitHub research 同步审查

- 按用户要求检查“当前 Git 下这些文件是否需要上传，避免 GitHub 中已有文件重复保留”。
- 当前本地 `research` 工作区相对 `origin/research` 显示 `[ahead 1, behind 5]`，导致已经在远端的文件在本地旧分支上表现为 untracked 或 deleted-like diff；不能直接按本地 `git status` 判断是否需要新增提交。
- 已从 `origin/research` 创建干净临时 worktree，并把当前候选文件覆盖进去做内容级合并演练：
  - `docs/sew-offload/08-ascend-moe-offload-architecture.md`
  - `docs/sew-offload/09-next-steps-after-mvp-a.md`
  - `tests/ut/moe_offload/*`
  - `tools/sew_offload/{collect_moe_trace.py,compare_smoke_outputs.py,estimate_fixed_slot_memory.py,run_fixed_slot_smoke.py,simulate_expert_slots.py}`
  - `vllm_ascend/moe_offload/*`
  - 相关 `docs/tests/vllm_ascend/ops` 修改
- 审查结论：上述 SEW-Offload 文档、工具、测试和 runtime 文件在 `origin/research` 已存在，当前内容与远端一致，不需要重复上传，也不应创建同名副本。
- 唯一真实差异是 `vllm_ascend/worker/model_runner_v1.py` 中 `_torch_cuda_wrapper()` 的 3 行旧内容：
  - 本地旧内容会把 `_EventPlaceholder.record/synchronize` 从可接收 `*args, **kwargs` 改回无参数 lambda。
  - 本地旧注释会覆盖远端 `Keep CUDA-shaped APIs routed to NPU implementations after init.`。
- 该差异属于旧工作区对远端修复的回退，不应提交到 `research`。
- 同步策略：本轮不上传任何代码文件；只把本次审查记录同步到 `.planning/sew_offload/progress.md`，保留规划文件作为项目整体状态来源。

## 2026-06-01 PrefetchOffloader 与分层驻留问题澄清

- 用户提醒 `.planning` 是项目整体规划方案，后续必须同步更新；本轮已把设计澄清写入 `findings.md` 和 `task_plan.md`。
- 核对代码后确认：`PrefetchOffloader` 是 `vllm-hust/vllm/model_executor/offloader/prefetch.py` 的通用参数 offloader，不是 `vllm_ascend` 的 MoE 专用模块。
- 结论：它可以借鉴 CPU storage、静态 buffer pool、copy stream/event、post-init sync 等机制，但不能直接作为 SEW-Offload 核心接口，因为它按 layer/module forward 顺序调度，不理解 MoE routing、active expert working set、logical-to-physical slot remap、per-expert token count 和 Ascend grouped MoE layout 契约。
- 反向设计审查发现：若把 fixed-slot MVP 理解成“所有专家都卸载到 CPU”，会走向另一个极端。最终设计应是分层驻留：NPU 中可保留若干完整层或热点专家，同时维护 CPU-backed fixed-slot cache 处理剩余专家 miss。
- 已更新计划：后续参数所有权转移从“释放全部原始 expert 参数”修正为“partial release + pinned/full-resident experts + CPU-backed cache slots”的分层驻留方案。

## 2026-06-02 D.9 复盘与固定 token capacity 反思

- 查看最近提交：`ac6e3922 feat(moe-offload): MVP D.9 implement tiered residency and partial release for expert weights`，当前 `research` 与 `origin/research` 对齐，工作区在复盘前干净。
- D.9 做了四件事：tiered residency env/config、resident 层跳过 fixed-slot path、opt-in partial release 原始 expert 参数、per-layer vs global slot capacity 估算。
- 进度判断：UT 与默认路径验证已完成；release=1 的真实 NPU smoke、HBM/ledger 对照、真实 trace JSONL 仍待办。
- 暴露问题：全模型 fixed-slot post-load/register 仍重；per-layer slot bank 成本随层数线性放大；release 只替换 Parameter storage，需 NPU 证明 vLLM 没有其它引用保留 HBM。
- 复核用户关于常规 MoE 的分析后确认：当前非 offload AllGather 路径是 dropless dynamic count，`npu_moe_init_routing` 返回每个 expert 的实际 token count，`group_list_type=1` 传给 `npu_grouped_matmul`。
- 设计修正：SEW-Offload 主线应固定 expert weight slot 和权重地址，而不是固定每个 expert 的 token 容量。`expert_capacity/drop_pad_mode` 只保留为后续可选实验分支，不能作为默认路径，也不能引入 drop token。

## 2026-06-02 按新顺序执行：打点 + 小范围 release=1 smoke

- 先更新 `task_plan.md`：把下一步顺序明确为“分段耗时/ledger 打点 -> 小范围 release=1 NPU smoke -> no-offload 对照 -> trace/sweep -> 决定 per-layer/global pool -> MVP-E”。
- TDD 新增 runtime profiling：
  - RED：`test_runtime_profiles_register_and_release_with_ledger_snapshots` 初始失败于缺少 `profiling_summary()`。
  - GREEN：新增 `MoeOffloadProfileEvent`、`MoeOffloadMemoryLedger.to_jsonable()`、register/release 事件记录。
  - 追加跨进程 RED/GREEN：`VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` 写 JSONL，解决 EngineCore 子进程 runtime 无法被父进程 summary 直接读取的问题。
- 更新 `tools/sew_offload/run_fixed_slot_smoke.py`：
  - 支持 `--resident-layer-ids` 和 `--release-original-expert-weights`。
  - 清理/设置 D.9 env，避免 no-offload baseline 被旧 env 污染。
  - summary 写入 `moe_offload_profile_jsonl_events`。
- 本轮验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_tiered_residency_d9.py tests/ut/moe_offload/test_config.py`：`28 passed, 4 warnings in 0.37s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/envs.py vllm_ascend/moe_offload/runtime.py tools/sew_offload/run_fixed_slot_smoke.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_smoke.py`：通过。
  - `git diff --check`：通过。
- 真实 NPU smoke：
  - candidate：`artifacts/sew_offload/runs/d9_release1_small_profile_20260602/summary.json`，NPU 6，layer 0 non-resident，layers 1-47 resident，release=1，状态 `ok`，1 output token。
  - profile JSONL：`register_layer_for_fixed_slots` layer 0 约 `2.034s`；release 前 original expert bytes `1207959552`，release 后 original expert bytes `0`；slot bank bytes `75497472`。
  - baseline：`artifacts/sew_offload/runs/d9_no_offload_profile_20260602/summary.json`，no-offload 状态 `ok`，1 output token，reported weight `56.9001 GB`。
  - candidate reported weight `42.3454 GB`；该数字主要受 native prefetch offloader 影响，不能单独归因于 SEW layer 0 release。
  - strict compare：`artifacts/sew_offload/runs/d9_release1_small_profile_20260602/correctness_compare.json`，`status=ok`、`matched=1`。
- 自我反思：
  - 小范围 release=1 机制正确，但不是性能结论。
  - 单层 register 约 2s，说明全 48 层 per-layer register/host clone 会非常重；下一步应先 trace/sweep，别贸然扩大 release 层数。
  - 真实 artifact 必须看 `moe_offload_profile_jsonl_events`，父进程 `moe_offload_profile` 为空是 V1 多进程架构导致的预期现象。

## 2026-06-02 继续执行：真实 trace + slot/residency sweep

- 按计划继续推进 trace/sweep 阶段，并边执行边反查设计正确性。
- TDD 补跨进程 trace JSONL：
  - RED：`test_runtime_appends_trace_jsonl_when_trace_path_is_set` 失败于 trace 文件不存在。
  - GREEN：新增 `VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH`，`trace_routing()` 在 trace-only 且显式设置路径时追加 JSONL。
  - 工具层 RED/GREEN：`collect_moe_trace.py` 设置 `TRACE_PATH`，优先统计真实 JSONL 行数，避免父进程空 collector 覆盖子进程 trace。
- 补 smoke env 污染护栏：
  - RED：`run_fixed_slot_smoke.py` 的 no-offload 清理未包含 `VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH`。
  - GREEN：把 `TRACE_PATH` 纳入 `SEW_OFFLOAD_ENV_VARS`，trace_only mode 写 `moe_offload_trace.jsonl`。
- 真实 NPU trace：
  - 命令使用 NPU 6、Qwen3-30B-A3B、1 条 synthetic `short_chat`、trace-only。
  - artifact：`artifacts/sew_offload/traces/d9_trace_short_20260602/trace.jsonl`
  - 结果：`TRACE_SUMMARY status=ok`、`num_trace_records=6192`，48 层，每层 129 records。
- 初步统计：
  - `active_experts_min=8`、`p50=8`、`p90=8`、`max=128`。
  - prefill 阶段可让单层 active expert 并集接近/达到 128；decode 大量记录是 top-8。
- slot/global sweep：
  - 输出：`artifacts/sew_offload/traces/d9_trace_short_20260602/sweep_summary.json`
  - per-layer fixed-slot：`num_slots=8/16/32/64/96` 下所有 48 层都至少一次超过 budget；`num_slots=128` 才不会因 active count 超限。
  - global LRU：slots=8/32/128 hit=0；slots=512 hit=21115/miss=37267；slots=1024 hit=33294/miss=25088。
  - sticky-layer LRU 与普通 LRU 在 512/1024 上近似，没有解决核心问题。
- 本轮验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_trace_export.py tests/ut/moe_offload/test_collect_moe_trace.py tests/ut/moe_offload/test_runtime_trace_only.py tests/ut/moe_offload/test_config.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py`：`34 passed, 4 warnings in 0.40s`。
  - `py_compile` 覆盖 `envs.py`、runtime、trace collector、collect/run smoke 工具与相关测试：通过。
  - `git diff --check`：通过。
- 自我反思：
  - trace 结果反证了“直接扩大 per-layer num_slots”路线：prefill 会迫使 `num_slots=128`，而 per-layer slot bank 成本随层数线性增长。
  - trace 也反证了“简单 global LRU 就够”的路线：小/中 slot 数几乎没有 reuse，需要 layer/window-aware 策略。
  - 下一步应先做 prefill/resident-aware 设计，而不是直接进入 MVP-E async transfer。

## 2026-06-02 分层策略分析与离线验证

- 按用户要求先分析设计分层策略，并进行验证；本轮没有改在线 MoE runtime，只新增离线 analyzer 和 CLI。
- TDD：
  - RED：`test_layered_strategy.py` 初始失败于缺少 `vllm_ascend.moe_offload.layered_strategy`。
  - GREEN：新增 `LayeredStrategyAnalyzer`，把高 fan-out 记录路由到 `full_weight_path`，低 fan-out 记录路由到 `slot_cache_path`。
  - 设计反查后追加 RED/GREEN：新增 `cache_scope=global|per_layer`。global 小池仍无 hit；per-layer decode slots 能保留层内复用。
- 新增文件：
  - `vllm_ascend/moe_offload/layered_strategy.py`
  - `tools/sew_offload/analyze_layered_strategy.py`
  - `tests/ut/moe_offload/test_layered_strategy.py`
- 真实 trace 验证：
  - 输入：`artifacts/sew_offload/traces/d9_trace_short_20260602/trace.jsonl`
  - 输出汇总：`artifacts/sew_offload/traces/d9_trace_short_20260602/layered_strategy_sweep_summary.json`
  - global scope：slots=8/16/32/64 hit rate 均为 0，说明简单全局小池不是可行主线。
  - per-layer scope：
    - slots=8：hit rate 约 31.4%，H2D 约 457.8GiB。
    - slots=16：hit rate 约 59.7%，H2D 约 268.7GiB。
    - slots=32：hit rate 约 78.9%，H2D 约 141.0GiB。
    - slots=64：hit rate 约 93.9%，H2D 约 40.7GiB。
  - 所有配置均有 `full_weight_records=95`，涉及 48 层，说明 prefill/high fan-out 仍需 full-weight/resident 处理。
- 验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_layered_strategy.py tests/ut/moe_offload/test_slot_simulator.py tests/ut/moe_offload/test_collect_moe_trace.py`：`10 passed, 4 warnings in 16.93s`。
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m py_compile vllm_ascend/moe_offload/layered_strategy.py tools/sew_offload/analyze_layered_strategy.py tests/ut/moe_offload/test_layered_strategy.py`：通过。
  - `git diff --check`：通过。
- 自我反思：
  - 分层策略不能简化为“prefill 全常驻，decode 全 offload”；真实实现还要算 HBM：per-layer slots=64 效果好但 HBM 约 42GiB，slots=32 更保守。
  - 当前离线验证支持下一步 runtime 原型：高 fan-out 使用原始/常驻权重，低 fan-out 使用 fixed-slot cache；默认仍关闭，不改 router/top-k/token count。

## 2026-06-02 源码复核 Static Expert Window 假设

- 按用户要求重新从代码而不是设计假设出发，复核 `vllm_ascend/ops/fused_moe` 与 `csrc`。
- 读到的主链路：
  - `fused_moe.py`：`select_experts(...) -> moe_comm_method.fused_experts(...)`，当前 fixed-slot hook 在进入 fused experts 前替换权重和设置 `log2phy/physical_expert_count`。
  - `moe_comm_method.py`：默认 staged boundary 已经存在，顺序是 `token_dispatch -> _apply_mlp -> token_combine`。
  - `token_dispatcher.py` AllGather：`npu_moe_init_routing(... expert_tokens_num_type=1, expert_tokens_num_flag=True ...)`，返回 `expert_tokens`，设置 `group_list_type=1`。
  - `moe_mlp.py`：`npu_grouped_matmul(... group_list=group_list, group_list_type=group_list_type)`，直接消费动态 count。
  - `csrc/torch_binding.cpp` / `moe_init_routing_custom_torch_adpt.h`：默认 `drop_pad_mode=0`，只有显式 `drop_pad_mode=1` 才走 `[expert_num, expert_capacity, h]`。
- 结论：非 offload 当前没有默认固定每 expert token capacity；`Static Expert Window` 如果继续使用，必须重新解释为 expert weight residency/window，不能再指 token capacity。
- 切入点修正：
  - 短期保持 `apply()` 前置 hook 只做权重准备、slot remap 和 fail-closed correctness。
  - 下一步真正要做的 runtime 原型应进入 `MoECommMethod.fused_experts()` 的 dispatch/MLP 边界，在 dispatch 输出 dynamic group_list 后做 resident/staged/miss expert phase split。
  - phase split 必须生成完整 permuted output buffer 后复用现有 combine，避免改 router/top-k/token count 语义。
- 自我反思：
  - 之前把 “window” 说成 token capacity 容易误导，源码证明这不是当前系统主线。
  - Python phase split 适合先验证语义，但如果只停留在 Python 多次 grouped matmul，会增加 kernel launch 和小 batch 开销；真正的 Ascend 特性利用应以 fused dispatch/GMM/combine 或 staging-aware custom op 为长期目标。

## 2026-06-02 下一步规划

- 按用户要求规划下一步，并同步 `.planning/sew_offload/task_plan.md`、`findings.md`、`progress.md`。
- 阶段推进：
  - D.9 收口：分层驻留、partial release、真实 trace、离线策略和源码复核已经给出方向。
  - D.10 新目标：实现默认关闭的 dynamic-count layered runtime path selector。
  - D.11 后续：进入 dispatch 后 phase split 语义原型。
  - MVP-E 后移：等 D.10/D.11 的语义边界闭环后，再做 async transfer/overlap metrics。
- D.10 的核心执行顺序：
  1. 定义 runtime decision contract：`full_weight_path`、`slot_cache_path`、`fail_closed`。
  2. 增加 env/config：分层 runtime 开关、fanout threshold、decision trace/profile。
  3. 加 full-weight readiness guard：高 fan-out 只能在 full expert 权重仍可用时走原始/常驻路径。
  4. 接低 fan-out slot-cache path：复用现有 fixed-slot sync，不改 dynamic count。
  5. 加 observability：记录 layer、active expert count、path、reason、release/resident 状态。
  6. 跑 UT 与小范围 NPU smoke，对比 no-offload token id。
- 自我反思：
  - 这个顺序刻意不直接做 async，因为还没有在线 path selector 时，async 只会把搬运和语义问题混在一起。
  - D.10 也不做 phase split，避免一次把 path 选择、buffer 回填、combine 等价性和 transfer overlap 全搅在一起。
  - 如果 D.10 发现 high fan-out full-weight 与 release 策略冲突，优先收紧 release guard，而不是放宽 slot budget 或引入 token drop。

## 2026-06-02 用户提供已成功 Ascend 预取原型

- 用户补充曾经成功实施过 Ascend expert cache plugin：
  - patch Worker/NPUWorker `load_model`；
  - load 后扫描 `FusedMoE` 的 `w13_weight/w2_weight`；
  - expert 权重复制到 CPU DRAM；
  - 可选把原 NPU expert tensor 替换为空 tensor；
  - 每层分配 NPU `pool_w13/pool_w2`；
  - patch `MoECommMethod.fused_experts`，运行时根据 `topk_ids` 做 cache hit/miss、CPU->NPU load、LRU/priority eviction、logical-to-slot remap，再调用原 fused experts。
- 本轮本机搜索：
  - 未在 `/root` 找到 `adapters/vllm_moeinf_official_plugin.py` 或 `adapters/ascend_expert_cache.py`。
  - 已检查当前内置模块，发现 `HostExpertStore`、`ExpertSlotBank`、`prepare_fixed_slot_plan(...)` 与用户 plugin 机制高度对应。
- 规划调整：
  - D.10 优先级调整为“把用户已成功 plugin 路线内生化到 `vllm-ascend-hust`”，尤其是把真正执行 hook 放到 `MoECommMethod.fused_experts` 边界。
  - phase split 和 async transfer 继续后移；先复刻已验证的 cache/remap 路线，确保默认关闭、可测试、可观测、fail closed。

## 2026-06-02 MVP-D.10 dynamic-count layered runtime MVP

- 按 TDD 完成在线 path selector MVP：
  - RED：新增配置、runtime decision、fused MoE hook 测试，初始失败于缺少 `MoeOffloadDecisionPath` / `decide_layered_path`。
  - GREEN：新增 `MoeOffloadDecisionPath`、`MoeOffloadPathDecision`、`MoeOffloadRuntime.decide_layered_path(...)`。
  - GREEN：新增 env/config `VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME`（默认关）与 `VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD`（默认 0，回退 `num_slots`）。
  - GREEN：在 `AscendUnquantizedFusedMoEMethod.apply()` 构造 fused input 前接入 decision：low fan-out 走现有 fixed-slot sync；high fan-out 在原始 full expert 权重可用时走 full-weight；high fan-out 且 full weights 已 release 时 fail closed。
  - GREEN：`run_fixed_slot_smoke.py` 输出 TTFT/TPOT，并把 profile JSONL 的 `layered_path_decision` 写入 summary。
- 真实 NPU smoke（NPU 6，Qwen3-30B-A3B，1 条 synthetic short_chat，layer 0 non-resident + layers 1..47 resident，num_slots=8，fanout_threshold=8，native prefetch expert offload 组合）：
  - 1-token candidate：`artifacts/sew_offload/runs/d10_layered_runtime_1tok_20260602/summary.json`，status ok，reported weight `43.4704 GB`，throughput `1.327 tok/s`，TTFT `751.58 ms`，TPOT `0 ms`（1 token 无间隔）。
  - 1-token strict compare：`artifacts/sew_offload/runs/d10_layered_runtime_1tok_20260602/correctness_compare.json`，status ok，matched=1。
  - 8-token candidate：`artifacts/sew_offload/runs/d10_layered_runtime_8tok_20260602/summary.json`，status ok，throughput `1.675 tok/s`，TTFT `868.72 ms`，TPOT `558.14 ms`，load `64.83s`。
  - 8-token no-offload baseline：`artifacts/sew_offload/runs/d10_no_offload_8tok_20260602/summary.json`，status ok，throughput `6.337 tok/s`，TTFT `445.15 ms`，TPOT `116.71 ms`，load `32.30s`，reported weight `56.9001 GB`。
  - 8-token strict compare：`artifacts/sew_offload/runs/d10_layered_runtime_8tok_20260602/correctness_compare.json`，status ok，matched=1。
  - candidate decision profile：1 次 `full_weight_path`（prefill high fan-out）+ 8 次 `slot_cache_path`（decode low fan-out）；`register_layer_for_fixed_slots` layer 0 约 `1.86s`。
- 验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/moe_offload/test_config.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_layered_strategy.py tests/ut/moe_offload/test_trace_export.py tests/ut/moe_offload/test_collect_moe_trace.py`：`52 passed, 4 warnings in 8.89s`。
  - `py_compile` 覆盖 D.10 runtime、fused MoE 和工具：通过。
  - `git diff --check`：通过。
- 自我反思：
  - D.10 MVP 已跑通 correctness 和指标输出，但当前仍是同步 slot cache + native prefetch 组合，性能显著慢于 no-offload；这不能写成性能收益，只能写成 offloading 通路与 observability 闭环。
  - 先保留 high fan-out full-weight path 是正确的：真实 prefill active expert count 约 81，若强塞进 8 slots 会改变语义或 fail；full-weight readiness guard 防止 release 后静默错误。
  - 当前 hook 仍在 `apply()` 中构造 fused input 前，尚未完全内生到 `MoECommMethod.fused_experts` 的 dispatch/MLP 边界；下一步应继续下沉 hook，再进入 D.11 phase split。

## 2026-06-02 D.10 fused_experts 边界下沉执行

- 按用户“继续执行下一步操作”的要求，已把 D.10 path selector 从 `AscendUnquantizedFusedMoEMethod.apply()` 前置准备进一步下沉到 `MoECommMethod.fused_experts()` 边界。
- 代码路径调整：
  - `MoEOffloadParams` 进入 fused MoE stage contract，`build_fused_experts_input(...)` 可携带 offload metadata。
  - `AscendUnquantizedFusedMoEMethod.apply()` 不再直接准备 slot 权重；它只负责在需要时注册 layer，并把 offload metadata 传入 fused input。
  - `MoECommMethod.fused_experts()` 在 token dispatch 前调用 `_maybe_apply_moe_offload_plan(...)`：从 `topk_ids` 得到 active experts，调用 runtime decision；slot path 准备 fixed-slot plan 并替换 `w1/w2/log2phy/physical_expert_count`；full path 保留原权重；fail-closed 在 dispatch 前抛错。
- 新增/更新测试：
  - `tests/ut/ops/test_moe_comm_method.py` 覆盖 fused boundary slot path、full path、fail-closed。
  - `tests/ut/moe_offload/test_fixed_slot_apply.py` 覆盖 `apply()` 只传 metadata、不提前 remap。
- fresh 验证：
  - `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/ops/test_moe_runtime_args.py tests/ut/ops/test_moe_comm_method.py tests/ut/moe_offload/test_config.py tests/ut/moe_offload/test_runtime_fixed_slot_guard.py tests/ut/moe_offload/test_fixed_slot_apply.py tests/ut/moe_offload/test_fixed_slot_smoke.py tests/ut/moe_offload/test_layered_strategy.py tests/ut/moe_offload/test_trace_export.py tests/ut/moe_offload/test_collect_moe_trace.py`：`67 passed, 4 warnings, 27 subtests passed in 8.93s`。
  - `py_compile` 覆盖 fused MoE contracts/runtime/offload/tools：通过。
  - `git diff --check`：通过。
- post-downsink 真实 NPU smoke 状态：
  - 第一次：`artifacts/sew_offload/runs/d10_fused_boundary_layered_1tok_20260602/summary.json`，vLLM startup memory gate 失败，free `45.94/60.96 GiB`，required `54.86 GiB` at util `0.9`。
  - 第二次：`artifacts/sew_offload/runs/d10_fused_boundary_layered_1tok_retry_20260602/summary.json`，util `0.7` 仍失败，free `17.41/60.96 GiB`，required `42.67 GiB`。
  - 第三次：`artifacts/sew_offload/runs/d10_fused_boundary_layered_1tok_retry2_20260602/summary.json`，util `0.7` 仍失败，free `17.28/60.96 GiB`，required `42.67 GiB`。
  - `npu-smi info` 随后显示 NPU 6 上 PID `3272964` 占用约 `44067 MB`，但 `ps -p 3272964` 无对应 Linux 进程；等待 15s 后账本仍未回收。未擅自 kill/reset 未知设备上下文。
  - 之后 `3272964` 账本消失，但 NPU 6 又显示 PID `3354289` 占用约 `15055 MB` 且 AICore 活跃；`ps -p 3354289` 同样无对应 Linux 进程。当前仍不视为干净 smoke 资源窗口。
- 当前可报告的真实推理证据仍是下沉前 D.10 path selector smoke：
  - 1-token candidate 与 no-offload strict compare 均 `status=ok`。
  - 8-token candidate 与 no-offload strict compare 均 `status=ok`。
  - 8-token candidate throughput `1.675 tok/s`、TTFT `868.72 ms`、TPOT `558.14 ms`；no-offload baseline throughput `6.337 tok/s`、TTFT `445.15 ms`、TPOT `116.71 ms`。
- 自我反思：
  - 下沉设计是正确方向：active experts 的来源更靠近真实 fused boundary，后续 D.11 能自然进入 `token_dispatch_output -> _apply_mlp` 的 phase split。
  - 但不能把 UT 通过等同于真实 Ascend E2E 通过；post-downsink smoke 必须等 NPU 资源恢复后补跑。
  - 当前失败发生在 model load 前的 memory gate，不是 slot remap、dynamic count、grouped matmul 或 decision contract 的 runtime failure。
  - 继续扩大功能前应先取得 post-downsink 1-token strict compare；否则 D.11 phase split 会把资源问题和语义问题混在一起。

## 2026-06-02 post-downsink smoke 补跑尝试与 D.11 规划

- 按用户要求尝试补跑 `post-downsink d10_fused_boundary_layered_1tok` smoke，并准备 strict compare。
- baseline artifact 已确认：
  - `artifacts/sew_offload/runs/d10_no_offload_1tok_20260602/summary.json` status `ok`。
  - baseline `outputs.jsonl` token id 为 `[1096]`，可作为 strict compare 基线。
- NPU 6 资源复查：
  - 首次 `npu-smi info`：NPU 6 HBM `47640/65536 MB`，AICore `49%`，PID `3438992` 占用约 `44073 MB`。
  - `ps -p 3438992 -o ...` 无对应 Linux 进程。
  - 等待 30s 后再次 `npu-smi info`：NPU 6 HBM `47786/65536 MB`，AICore `51%`，同 PID `3438992` 仍占用约 `44073 MB`。
  - 当前 free HBM 约 17GiB，低于此前 util `0.7` 启动门禁所需约 `42.67GiB`；也不足以可靠加载 Qwen3-30B-A3B offload candidate。
- 本次未启动新的 smoke 命令，原因：
  - 资源状态明确不满足 vLLM startup memory gate，重复运行只会制造新的 failed artifact。
  - 占用 PID 不在 Linux 进程表中，且 AICore 活跃；未获得用户授权前不做 NPU reset 或 kill。
- D.11 规划已写入 `task_plan.md` 阶段 14：
  - 目标是 dispatch 后 phase split 语义原型，不做 async transfer。
  - 先定义 phase plan 与 expert slice contract，再做 group_list slicing、phase planner、partial MLP、full-buffer 回填、equivalence UT、窄路径集成和 observability。
  - D.11 的 NPU smoke 仍以前置 D.10 post-downsink 1-token strict compare 为建议门禁。
- 自我反思：
  - 当前不能把“未补跑”解释为工程失败；这是硬件资源不可控导致的验证阻塞。
  - 继续规划 D.11 是有价值的，但实现前最好先取得 D.10 下沉后的 1-token strict compare，避免未来出现错误时无法判断来自下沉 remap 还是 phase split。

## 2026-06-02 D.10 post-downsink 真实 NPU smoke 补跑通过

- NPU 6 资源窗口恢复（HBM 约 3620MB、AICore 0%、无业务 PID），按计划补跑下沉版（fused_experts 边界内生化）真实 smoke。
- 配置：NPU 6，Qwen3-30B-A3B，inline `Hello`，layer 0 non-resident + layers 1..47 resident，`--num-slots 8`，`--fanout-threshold 8`，`--release-original-expert-weights`，native prefetch expert offload（group_size 4 / num_in_group 1 / prefetch_step 1 / params experts），`max_model_len=512`、`max_num_seqs=1`、`kv_cache_memory_mb=512`。
- 1-token：
  - baseline `artifacts/sew_offload/runs/d10_postdownsink_no_offload_1tok_20260602`：status ok，reported weight `56.9001 GB`，token id `[353]`。
  - candidate `artifacts/sew_offload/runs/d10_postdownsink_fused_boundary_layered_1tok_retry_20260602`：status ok，reported weight `42.3454 GB`，throughput `1.189 tok/s`，TTFT `839.34 ms`，token id `[353]`，path decision `slot_cache_path`/`low_fanout_slot_cache_ready`，`release_original_expert_weights` 后 layer 0 original expert bytes 从 `1207959552` 归零。
  - strict compare `correctness_compare.json`：status ok，matched=1。
  - 第一次启动曾因 NPU 6 残留幽灵设备上下文（PID 不在 Linux 进程表、AICore 活跃）偶发 engine core init 失败；未擅自 reset/kill，约 30s 上下文自行释放后原样重跑即成功。
- 8-token：
  - baseline `artifacts/sew_offload/runs/d10_postdownsink_no_offload_8tok_20260602`：status ok，throughput `5.512 tok/s`，TTFT `582.88 ms`，TPOT `124.01 ms`，token ids `[353,91957,9,0,358,2776,501,311]`。
  - candidate `artifacts/sew_offload/runs/d10_postdownsink_fused_boundary_layered_8tok_20260602`：status ok，throughput `1.475 tok/s`，TTFT `1012.95 ms`，TPOT `630.06 ms`，token ids `[353,91957,9,0,358,2776,501,311]`，全程 9 次 `slot_cache_path`/`low_fanout_slot_cache_ready`（短 prompt prefill fan-out 亦低于阈值 8）。
  - strict compare `correctness_compare.json`：status ok，matched=1。
- 结论：
  - post-downsink NPU smoke 解除阻塞：下沉到 `MoECommMethod.fused_experts()` 后的 path decision、slot plan、`log2phy/physical_expert_count` remap 与 dynamic count/grouped matmul 在真实 NPU 上保持 token-id correctness。
  - 同步 slot path 性能仍显著慢于 no-offload（8-token candidate `1.475 tok/s` vs baseline `5.512 tok/s`），只作为 offloading 通路与 observability 闭环，不写成性能收益。
  - D.10 correctness 门禁已过，可进入 MVP-D.11 dispatch 后 phase split 语义原型；不再把硬件资源问题和 phase split 语义问题混在一起。

## 2026-06-05 MVP-D.11 NPU Semantic Smoke

- 检查 NPU 资源：NPU 2/4/5 空闲（~61.8GB free HBM），选择 NPU 2。
- 确认 `ASCEND_RT_VISIBLE_DEVICES` 是 Ascend NPU 的正确设备隔离变量（`CUDA_VISIBLE_DEVICES` 无效）。
- 创建 `tools/sew_offload/run_phase_split_smoke.py`：D.11 独立 smoke runner，支持 `--mode no_offload|phase_split`。
- 运行 1-token smoke：
  - baseline（no_offload）：token `[26288]`，throughput 2.29 tok/s ✅
  - candidate（phase_split）：token `[26288]`，throughput 2.16 tok/s ✅
  - strict compare：`status=ok, matched=1`
- 运行 8-token smoke：
  - baseline：`[26288,102064,104949,9370,104034,20074,89161,102021]`，5.60 tok/s ✅
  - candidate：完全一致，4.71 tok/s ✅
  - strict compare：`status=ok, matched=1`
- Profile JSONL：96 events（1-token）、432 events（8-token），均为 all-hit single phase。
- 阶段 13/14（MVP-D.11）**全部完成**。

## 下一步

- 进入 MVP-E：async transfer + overlap metrics。
- D.12 staging-aware fused/custom op 或 window-aware global pool。
- 新增 `vllm_ascend/moe_offload/phase_split.py`：
  - `MoEPhase` / `MoEPhasePlan` contract dataclasses。
  - `compute_expert_token_slices()`：group_list type 0/1 → per-expert `[start, end)`。
  - `plan_hit_miss_phases()`：基于 slot_readiness 字典切 hit/miss phases。
  - `_extract_phase_tokens()`、`_build_phase_group_list()`、`_slice_expert_weights()`：子 MLP 输入构建。
  - `_scatter_phase_output()`：回填到完整 output buffer。
  - `execute_phased_mlp()`：顶层编排器，单 phase fast-path。
  - `PhaseSplitProfileEvent` + JSONL writer。
- 修改 `vllm_ascend/ops/fused_moe/moe_comm_method.py`：
  - 新增 `_maybe_plan_phase_split()`：窄路径 gate、phase plan 构建、profile 写 JSONL。
  - `fused_experts()` 中在 `build_mlp_compute_input` 后分支到 phased 或单次 MLP。
- 新增 env `VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT`（默认 0）。
- 新增 `MoeOffloadConfig.phase_split_enabled`。
- 更新 `vllm_ascend/moe_offload/__init__.py` 导出新 API。
- 新增 UT `tests/ut/moe_offload/test_phase_split.py`：31 个测试全部通过。
  - 覆盖：type 0/1 slicing、hit/miss/all-hit/all-miss 等价性、空 expert、group_list 构建、权重切片、scatter、profile JSONL、空 phase fallback、乱序/缺失 slice fail-closed。
- 回归：现有 moe_offload UT 133 passed（2 预存在 import error + 1 预存在 env monkeypatch 未计入）。
- 语法检查：所有新增/修改文件 `py_compile` 通过。
- 更新 task_plan.md / findings.md / progress.md。

## 下一步

- NPU 资源可用后跑 D.11 NPU semantic smoke：1-token 和 8-token strict compare。
- 进入 MVP-E：async transfer + overlap metrics。
- D.11 复盘门禁：若 Python 多 phase 开销过高，记录为语义脚手架并保留 single-phase fallback。

## 2026-06-15 方向重定与 e2e 证伪 + ACLGraph 可行性分析

### custom GMM kernel e2e 证伪
- 新增 in-situ harness `tools/moe_gmm/e2e_gmm1_insitu.py`：env 门控（`VLLM_ASCEND_GMM1_PROBE_BACKEND/PATH`），只计 GMM1 累计耗时、单后端单进程、带 token-id 对照。
- 在 `vllm_ascend/ops/fused_moe/moe_mlp.py` 的 `unquant_apply_mlp` 加 env 门控探针（默认关，零行为变化）。
- baseline（torch_npu，NPU4）：6192 次 GMM1，2191.39ms，per-call 0.32ms，成功。
- custom 后端（NPU4/6）：**真实 decode 单 token 阶段 AICore 异常崩溃（error 507015）**；prefill/warmup 能过，decode 必崩。
- 结论：custom GMM1 在真实 decode 形态下不正确，microbench 的 ~4% 在 19-32% 噪声内且不收敛。custom GMM kernel 路线已被 e2e 证伪。

### 真实 profile 算子排名（决定方向）
- GroupedMatmul 46%、MatMulV2 12%，均 MTE2-bound 92-94%、Cube 仅 12% → 撞 HBM 带宽墙。
- skill 矩阵交叉验证：prefetch（无安全窗口）、multi-stream（共享带宽）、superkernel（需 A3 硬件）在 memory-bound 下收益被带宽锁死。能移动天花板的只有减字节：量化（减权重）或 fusion（减激活往返）。

### 用户定向：CCF-A 系统论文，重评 offload 主线
- 量化在 30B 上基本消灭 offload 动机（权重减半→轻松装单卡），offload 真实战场是 122B 等量化后仍装不下的大模型。
- 用户给出 track 2 题眼：offload 被迫 `--enforce-eager`，因为 graph capture 无法录制 offload scheduler 的 runtime CPU 决策路径。

### ACLGraph 可行性分析（driven by model-infer-graph-mode skill）
- 落盘 `docs/superpowers/specs/2026-06-15-aclgraph-offload-decision-hoisting-feasibility.md`。
- 捕获边界（代码确认）：piecewise，`splitting_ops` 只含 attention → 整个 MoE 在捕获图段内；`static_all_moe_layers` 列全 48 层。
- 阻塞点：`moe_comm_method.py:327` 的 `torch.unique(topk_ids.detach().cpu()).tolist()` + `decide_layered_path` + `prepare_fixed_slot_plan` — 同步/数据依赖/Python 控制流/条件 H2D 四重违反。capturing 防护只在观测路径（runtime.py:196,222），执行路径裸奔。
- 先例：`acl_graph.py` 的 `update_attn_params` 已为 attention 做控制面/数据面解耦（CPU 算 metadata→固定 buffer→event 排序）；无 MoE 版本。
- 难点：MoE routing 在 forward 内 gate 产出，决策既数据依赖又产生于捕获区内部，不能像 attn 那样简单上提。
- 推荐 Option 2：在 routing 点切分 MoE（复刻 attention piecewise），grouped_matmul+combine 保持捕获+固定 slot，仅 staging 走 eager。这是论文核心贡献：图模式兼容的 MoE offload via 控制面/数据面解耦。

## 下一步（2026-06-15）

- 决定性 2×2 baseline（NPU）：no-offload+图 / no-offload+eager / offload+eager / offload+图-attempt，量化 enforce_eager 的损失，作为论文 motivation 表。
- Option 2 原型：注册 MoE split boundary，验证 grouped_matmul 段在固定 slot 下可捕获。
- async staging（MVP-E）：load stream/event，量化 exposed-stall 下降。

## 2026-06-15 决定性 2×2 实测：ACLGraph×offload 冲突铁证

- 工具 `run_fixed_slot_smoke.py`，artifacts `benchmarks/results/aclgraph_2x2_20260615/`。
- **Cell 1 no-offload + ACLGraph**：✅ 捕获成功（"Graph capturing finished"，PIECEWISE，48 层），status ok，TTFT 298ms（1-tok）。8-tok 在 64GB 边缘 OOM（56.9GB 常驻）。证明 ACLGraph 在无 offload 时正常。
- **Cell 4 offload + ACLGraph（--no-enforce-eager）**：❌ **在 `capture_model()` 阶段硬失败**。根因逐字：
  - `Not allow to synchronize captured-stream, stream_id=1749` (107027)
  - `rtMemcpy ... the current capture mode does not support this operation` (107030)
  - `synchronized memcpy failed, kind = 2`（device↔host），发生于 `model_runner_v1.py:3893`。
  - `kind=2` 即 `_maybe_apply_moe_offload_plan` 里 `topk_ids.detach().cpu()` 的 D2H 同步拷贝。
- 结论：offload+图模式不是软 graph-break/重编译，而是 runtime **硬禁止** captured-stream 上的 synchronize/synchronized memcpy。**enforce_eager 是当前 offload 架构的硬约束，不是配置选择**——这是论文 track 2 motivation 的铁证。
- 证据已写入 `docs/superpowers/specs/2026-06-15-aclgraph-offload-decision-hoisting-feasibility.md` §9。
- 待测（定量 motivation 表）：no-offload+eager 与 offload+eager 的 TPOT 差（量化 launch 收益损失）；受 no-offload+ACLGraph 多 token OOM 限制，需结合量化或用 122B 多卡取干净 decode 数。

## 下一步（2026-06-15 b）

- 跑 cell 2/3（no-offload+eager、offload+eager）TPOT，补全 motivation 定量表（注意 ACLGraph 上界受 OOM 限制）。
- 进入 Option 2 原型：在 routing 点切分 MoE，把 D2H 决策+staging 移到 eager split，grouped_matmul 在固定 slot 上保持捕获。

## 2026-06-15 c：A 定量表完成 + B Option 2 原语落地（research 分支）

### A — 2×2 定量 motivation 表（NPU 4/5/7）
- Cell 1 no-offload+ACLGraph：✅ capture 成功，TTFT **298ms**（1-tok；8-tok 在 56.9GB 边缘 OOM）。
- Cell 2 no-offload+eager：✅ TTFT **797ms**，TPOT 200ms（8-tok）。
- Cell 3 offload+eager：✅ 唯一可跑，TTFT 36s/TPOT 5.4s*（*未设 resident-layer-ids，48 层全同步 slot load 的最差情况，已标 caveat）。
- Cell 4 offload+ACLGraph：❌ capture 硬失败（107027/107030，synchronized memcpy kind=2 forbidden）。
- 关键 delta：**no-offload 下 ACLGraph TTFT 比 eager 快 2.67×**（298 vs 797），即 offload 被迫 eager 丢掉的 launch 收益。写入可行性文档 §9.1。

### B — Option 2 graph-compatible offload 原语（默认关，零行为变化）
- `config.py`：新增 `graph_compatible_offload`（默认 False）。
- `envs.py`：新增 `VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE`（默认 0）。
- `runtime.py`：
  - 注册时分配**持久 per-layer log2phy buffer**（固定地址，`_log2phy_buffers`），尺寸=num_logical_experts。
  - `stage_fixed_slot_plan(...)`：eager 预备阶段（决策+H2D+in-place 写持久 buffer），capture 时拒绝（fail-closed）。
  - `capture_safe_slot_weights(...)`：capture 路径，指向固定 slot tensors + 固定 log2phy buffer，**零 host sync / 零条件 H2D**；未注册层返回 None。
  - `log2phy_buffer(layer_id)` 访问器。
- `moe_comm_method.py`：
  - `_maybe_apply_moe_offload_plan` 加 capture guard：`graph_compatible_offload and _is_current_graph_capturing()` 时走 capture-safe 路径，绕过被禁的 `torch.unique(...).cpu()`。
  - 提取 `_with_prepared_slot_weights(...)` helper，capture 路径与 eager 路径共用。
- 测试：新增 `tests/ut/moe_offload/test_graph_compatible_offload.py`（6 passed）——证明持久 buffer 地址稳定、in-place 更新、capture-safe 零同步、capture 时拒绝 stage。
- 修复 `test_moe_comm_method.py` 3 个 blanket-MagicMock 污染（2 个预存在 + 1 个我引入），显式设 `graph_compatible_offload=False`/`phase_split_enabled=False`；现 7 passed。
- 回归：B 相关 47 passed；moe_offload+ops+envs 全套 195 passed（test_envs 的 PROFILE_PATH 偶发失败是跨测试 os.environ 污染，非本次改动，隔离运行 106 subtests 通过）。

### B 的边界（诚实声明）
- 当前是**原语 + capture guard**，CPU 单测验证。**未做** model_runner 集成（capture 前/replay 前调用 stage_fixed_slot_plan 的生命周期钩子），因此**未在真实 NPU 上验证 offload+ACLGraph 能 capture 通过**。这是下一步。
- 完整 Option 2 还需：注册 MoE split boundary（custom op），把 stage 钩到 replay 前（类比 update_attn_params），固定 shape group_list（pad-to-capacity 防重编译）。

## 下一步（2026-06-15 c）
- B 的 NPU 验证：在 model_runner 加 stage_fixed_slot_plan 的 replay-前钩子，开 `VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=1` 跑 offload+ACLGraph，验证 capture 通过 + token-id 与 eager offload 一致。
- 若 capture 通过：补 async staging（MVP-E），量化 exposed-stall。

## 2026-06-16 — 实验 A：B 的 capture-pass 在真实 NPU 实证（决定性）
归档分支 `feature/moe-offload-runtime`（commit a167b355）。完整性核对：moe_offload 全套 183 passed/1 skip、ops 30 passed+35 subtests、9 个核心文件 AST 干净、B 原语方法体完整、Feature-2 GMM 探针已干净剔除。

### 环境与坑
- NPU 共享机争用对抗性极强：一张卡空窗 <60s，30B 加载需 60-235s，普通启动屡屡在内存检查点（启动后~30-45s）被抢 → `Free memory 4.x/60.96 GiB` OOM。
- `npu-smi info`（全局滚动表）行号映射不可靠；权威读数用 `npu-smi info -t usages -i <d> -c 0 | grep "HBM Usage Rate"`。
- 离线 `LLM()` + 仅设 `VLLM_ASCEND_MOE_OFFLOAD_GB` **不激活 offload**：autoconfig 的 `os.environ.setdefault` 在父进程，spawn 子进程未继承全部门控。修复：harness 直接显式设 `VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1/NUM_SLOTS=8/LAYERED_RUNTIME=1/FANOUT_THRESHOLD=8/MAX_PHASES=1`（真实 env 变量，spawn 继承）。
- 工具：`tools/sew_offload/run_graph_compat_capture_probe.py`（单配置 probe，token-id 输出）+ `tools/sew_offload/race_launch.sh`（抢卡重试：检测 <10% 即发，OOM 重试，非-OOM 崩溃停下待查）。

### 决定性一对照实验（NPU 5，Qwen3-30B-A3B，offload 14GB→8 slots / 128 experts，ACLGraph PIECEWISE，唯一变量 = graph_compatible flag）
- **控制组 flag=0**：`capture_model`（model_runner_v1.py:3893）内**崩溃**——`107027 synchronize stream failed` + `107030 synchronized memcpy failed, kind=2`（`copy_between_host_and_device_opapi / aclrtMemcpy`）。即 `torch.unique(topk_ids.detach().cpu())` 的 D2H 在捕获流被禁。日志 `.planning/sew_offload/logs/C_control_offloadON_flag0.log`。
- **实验组 flag=1**：capture **通过**——`Capturing CUDA graphs (PIECEWISE): 100%` + `Graph capturing finished in 10 secs, took 0.03 GiB`，**无 107027/107030**。capture-safe 路径绕过 D2H。日志 `.planning/sew_offload/logs/D_test_offloadON_flag1.log`。
- **结论**：仅翻转 flag，offload+ACLGraph 的 capture 从硬崩溃→通过。**论文 track 2 核心论点首次在真实硬件证实**（此前为代码推断 + CPU 单测）。

### Milestone 2 边界（实验组 generate 阶段精确暴露）
- 实验组 capture 通过、LOAD_OK 后，generate 报 `RuntimeError: Expected all tensors to be on the same device, but got weight is on cpu ... wrapper__npu_grouped_matmul`。
- 根因：offload 把专家权重放 host，capture-safe 路径让捕获图指向固定 slot，但**无 staging 钩子把专家装进 NPU slot / 写 log2phy**，replay 时 grouped_matmul 读到 host 权重。这正是文档已声明"未做"的 Milestone 2。
- 即 **Milestone 1（capture 通过）= ✓ 真实 NPU 证实；Milestone 2（replay 正确性）= 需 model_runner staging 钩子**。下一步：在 replay 前为每个注册 MoE 层调 `stage_fixed_slot_plan`（类比 update_attn_params），再验证 token-id 一致（严格一致仅在全驻留 num_slots≥128 退化配置成立；部分驻留需完整 Option 2 的 split boundary）。

## 2026-06-16 (续) — M2 启动即触发根因修正

- 读码确认 generate 崩溃（D-run, flag=1）根因：offline 探针**未触发 autoconfig→PrefetchOffloader**
  （C/D 日志无 autoconfig 行、无 PrefetchOffloader init、GB 被当作 Unknown env），
  导致非驻留层 w13 停在 CPU，高 fanout prefill 走 FULL_WEIGHT_PATH 读 CPU 权重而崩。
  **非** M2 staging 边界。详见 findings.md「2026-06-16 — M2 根因诊断」。
- 确认 PrefetchOffloader（设备驻留，graph-compatible by design）与 SEW fixed-slot（控制面）
  并存且作用于同一批非驻留层；验证命令(--enforce-eager)跑通正是靠 PrefetchOffloader 让 w13 设备驻留。
- 确认 staging hook 仍是 M2 真缺口（持久 log2phy 仅由 stage_fixed_slot_plan 写入），
  但它落在 model_runner 主路径 —— 触及项目规则「model_runner 变化需严格架构审查」。
- 因此在改 model_runner 前先与用户对齐：下一步实验路径选择（见对话）。

## 2026-06-16（续3）— Path A: SEW-only 捕获验证跑通 + 后续边界确认

### 已完成
- 诊断 E-run 107025 / F-run 107024 根因：均来自 PrefetchOffloader copy_stream（数据面），
  其 graph-capture 逻辑 gate 在未被 NPU 别名化的 `torch.cuda.is_current_stream_capturing()`（捕获期恒 False），
  copy_stream 对 NPU ACLGraph 不可捕获，wrapper 级无法修。详见 findings.md。
- acl_graph.py 补回 stock vLLM 漏移的 `get_offloader().sync_prev_onload()`（capture 前/内、replay 前三处）；
  NoopOffloader 下惰性，对非 offload 主路径零影响。
- 用户决策：走 Path A（SEW-only 捕获验证，论文主线）。
- 新增 `tools/sew_offload/race_launch_sew_only.sh`（不设 GB→NoopOffloader，直设 SEW env；
  FANOUT_THRESHOLD=128 强制 eager profile 走 SLOT_CACHE，规避读 CPU 权重崩溃）。
- **G-run（SEW-only, flag=1）capture 通过 + generate 跑通**：LOAD_OK / Capturing PIECEWISE 100% /
  GENERATE_OK / OUTPUT_TOKENS [3555,525,279,1376,6813,315,1741,4119]。107024/25/27/30 全消失。
  独立 from_env 验证 SEW 真激活（enabled/num_slots=128/graph_compatible/46层驻留/层2,3非驻留）。
  **论文主线（capture 维度）坐实：SEW 控制面 primitives 在 ACLGraph 下可捕获，无需 PrefetchOffloader/enforce_eager。**

### 进行中
- token-id parity 对照：`race_launch_baseline.sh`（无 offload 全驻留 + ACLGraph，扫描全卡抢空闲）
  取 ground-truth token。当前 8 卡全忙（86-93%），race 在后台轮询等卡。

### 已确认的下一边界（M2 真缺口）
- 源码链路确认：`log2phy[topk_ids]` gather（moe_comm_method.py:149/529）被录进捕获图固定地址，
  replay 重读持久 buffer 当前内容。持久 buffer 仅由 `stage_fixed_slot_plan` 写（需 model_runner staging hook，未装）→
  capture 时恒 -1 → replay 时层 2,3 路由到 slot[-1] → 该 2 层算错，但 46 驻留层正确 → 输出仍连贯。
- 故 **capture-pass ≠ token-correct**；token 正确性需 staging hook 写对持久 buffer。
  staging hook 落 model_runner 主路径（受严格架构审查约束）→ 实装前需与用户对齐。

## 2026-06-16（续4）— 当前代码 no-offload baseline 复现 + eager-SEW 对照

### 已完成
- **当前代码 no-offload baseline（BASE_nooffload_aclgraph，NPU5）跑通**：
  `OUTPUT_TOKENS [3555, 525, 279, 22146, 323, 63625, 315, 1667]`
  与早前 carried-over A baseline **逐 token 完全一致** → 证明本次 acl_graph.py 改动在 NoopOffloader 下确属惰性，不影响默认路径。A 可作合法 baseline。
- 三方对比定稿（前 3 token 同，第 4 起分叉）：
  ```
  BASE/A (no offload):    [3555, 525, 279, 22146, 323, 63625, 315, 1667]
  G (captured SEW flag1): [3555, 525, 279, 1376,  6813, 315,   1741, 4119]
  ```
- 厘清 RESIDENT_LAYER_IDS 语义：列出的是**驻留**层；`should_use_fixed_slot_plan_for_layer = should_use_fixed_slots AND NOT is_resident_layer`。
  G 的 resident=46 层（0–47 去掉 2,3）→ 仅层 2,3 offload。

### 进行中
- **eager-SEW 对照（H_sew_eager_flag1）**：--enforce-eager + 同一组 SEW env + 同 resident CSV。
  后台 race PID 775169 轮询 NPU5（40 tries）等空闲卡。
  - 判定逻辑：eager 路径每步 `prepare_fixed_slot_plan` 写 fresh log2phy（正确）。num_slots=128≥128 expert → 无淘汰。
    - 若 **H == BASE** → slot 机制本身 token 正确 → captured 路径唯一缺口 = staging hook 未写持久 log2phy buffer。
    - 这将把"capture-pass ≠ token-correct"的根因**精确隔离**到一个缺失的 model_runner staging hook（M2 真边界）。

### 阻塞
- 8 卡被其他容器占满（86-94%）。baseline 靠 NPU5 瞬时空窗抢到；eager 对照在后台轮询等下一个空窗。
- 不可 kill 其他容器 NPU 进程；不可动 port 8016。

## 2026-06-16（续5）— launcher 修复 + hook 设计稿 + eager 对照推翻诊断

### 已完成
1. **修复 race launcher fall-through 缺陷**（`race_launch_sew_eager.sh` + `race_launch_sew_only.sh`）：
   等不到 `<15%` 空窗时 `continue` 跳过本 try，不再硬启动 OOM 空转。新增 `FREE_THRESHOLD=15`。
   实测生效：H 在 NPU5 "at 5% (<15%) — launching" → "WON"，无 OOM 噪声。
2. **起草 hook 设计稿** `docs/sew-offload/12-eager-staging-hook-design.md`（含 `-1` buffer 代码事实、
   控制面/数据面环形依赖、Regime A/B 分层、接线点 (a)/(b)、不变量 checklist、验证计划 V0-V4）。
3. **eager-SEW 对照 H 跑通**（决定性 oracle）：**H == G ≠ BASE**，推翻"`-1` buffer 是分叉成因"。

### 关键转折
- staging hook **必要但不充分**：落地后只会 captured == eager == 仍错。
- **已冻结 hook 实现**（设计稿顶部标 FROZEN）。
- 真正缺陷：fixed-slot staging/remap 本身使 offload 层 MoE 输出偏离全驻留 baseline（eager 也错）。

### 下一步（根因化 eager-SEW 分叉，findings.md 末尾列了 4 条候选）
- 优先级 1：抓 BASE 首个分叉 step 的 top-2 logit 间距 → 判数值 vs 逻辑。
- 优先级 2：layers 2,3 设 resident 跑 SEW 应 == BASE（确认非 offload 层无回归）+ 单层二分。
- 优先级 3/4：核对 slot 权重逐元素无损 + remap 数学。

### 工具状态
- 修复版 launcher 已可靠抢瞬时空窗（无空转）。
- eager 对照 H 已得结果，PID 已自然退出。

## 续6 — 数值 tiebreak 实测闭环（2026-06-16）

- 顺序链（PID 823911）跑完：eager-SEW + baseline 双 logprob 输出已取。
- pos=3 完美镜像：BASE 22146(r1,-1.44021)/1376(r2,-1.56521) gap0.125；EAGER 1376(r1,-1.52263)/22146(r2,-1.52263) gap0.000。
- pos=2 一致到 ~0.0016 nat，margin 0.625 不翻。
- 判定：数值近简并 greedy tiebreak，非逻辑 bug。SEW 与全驻留数值等价（~0.08 nat）。详见 findings.md 同日条目。
- 张力待查：G(captured)==H(eager) 与 "-1 buffer 应出乱码" 矛盾，下一阶段独立排查 captured 路径实际是否读 stale buffer。
- TODO 状态：本节关闭数值/逻辑判别；hook 设计稿仍 FROZEN，待 captured 张力厘清后再决定 reframe/unfreeze。

## 续7 — NPU 探针实证：captured 真读 -1 buffer（2026-06-16）

- 加 env 门控探针 `SEW_OFFLOAD_PROBE` 于 `_maybe_apply_moe_offload_plan` 两分支（默认惰性，只动 offload 算子文件）。
- NPU5 graph 模式跑 G 同配置。探针序列：warmup=EAGER → capture=CAPTURE_SAFE(wire -1 buf) → prefill=EAGER → decode=零行(纯 replay)。
- 决定性：decode 零探针行 ⟹ captured 图真读 -1 buffer 并 mis-route layers{2,3}。
- 输出 pos6 才偏离 BASE（原 G 在 pos3）= 同配置不同翻转位 = run-to-run 非确定性。
- **修正**：-1 buffer 不是无害；2 层污染被 greedy 近简并掩蔽。eager-SEW 才数值等价；captured-SEW 需 staging hook。
  FROZEN hook 稿前提被坐实，应 unfreeze。详见 findings.md 同日续条目。
- 探针代码保留（env 门控），后续 scaling 实验复用。下一步：offload 层数 scaling（验掩蔽 vs 正确）+ unfreeze hook 稿。

## 续8 — offload-层数 scaling 实验完成（2026-06-16）

- 脚本 `tools/sew_offload/run_offload_scaling.sh`：NPU5 顺序链 6 配置（BASE + captured N=1/2/4/6 + eager-SEW N=4 对照），--logprobs 20。
- 结果：captured 发散随非驻留层数单调增大 pos7→pos7→**pos2**→退化重复；eager_N4 仅 pos3 近简并翻转。
- 决定性 A/B（pos2，margin 0.625 nat 的决定性位）：BASE/eager_N4 选 279 一致到 0.002 nat；cap_N4 翻成 862（279 被贬 ~0.6 nat，~1 nat 重排）。
- ⟹ captured `-1` 是真 mis-route（非数值噪声）；缺陷专属捕获路径（同 N eager 正确）；staging hook 必须，应 unfreeze。详见 findings.md 续2。
- 全部跑通无 OOM（N=6 也通过），无泄漏进程，NPU5 已释放。探针代码 env 门控保留。
- 下一步：unfreeze docs/sew-offload/12 staging hook 设计稿，作为 M2 收口实现方向。

## 续9（2026-06-16）：V1 Regime A staging hook 落地（CPU 侧完成，NPU 待资源）

- 决策：挂点选 **SEW 自有 fused_moe.py 注册点之后**，而非原设计稿推荐的 model_runner
  `capture_model` 前置 pass —— 完全不碰 model_runner，规避架构评审门槛。
- runtime.py 新增薄封装 `stage_full_residency_slot_plan(layer_id) -> bool`：
  四道门控（fixed_slots∧graph_compat、非resident、已注册、非捕获期）后从 buffer.numel()
  取 n、对全专家调既有 `stage_fixed_slot_plan`。捕获期安全 no-op（返回 False 不抛错）。
- fused_moe.py 两处接线（load-time 主路径 + lazy-forward 兜底），均在 register 之后、
  release 之前。从 host_store 独立 CPU 副本 staging，顺序安全。
- 单测：test_graph_compatible_offload.py +5 例（fill / off / capturing / unregistered /
  fail-closed），11 例全绿；邻近 fixed-slot + moe_comm_method 套件 50 例无回归。
- docs/sew-offload/12 §5 重写为"已落地"、§6 验证表标 V0/V3 ✅，V1/V2/V4 ⬜待 NPU。
- 下一步（待 NPU 资源）：V1 captured-SEW Regime A 实测，验证 pos2 决定性位回到 BASE
  （不再 279→862）；model_runner 一行未动，无需架构评审。

## 续10（2026-06-16）：V1 NPU 实测决定性通过(after-hook)

- NPU5 顺序重跑 scaling 全 6 配置(hook 已在源码,graph_compat=1∧num_slots=128 自动触发)。
- before-hook 日志已备份 .planning/sew_offload/logs/before_hook/(论文对照左半张)。
- 结果:cap_N1/N2/N4/N6 **逐 token 完全等于 BASE**,before-hook 的单调发散全部塌平。
- 决定性位 pos2:cap_N4 BEFORE chosen=862(mis-route) → AFTER chosen=279(修复)。
- 强于判据:after cap_N4 pos2 logprob 逐位精确等于 BASE 到 1e-5(279:-0.68338 vs -0.68338)
  → captured 路径与全驻留**数值等价**,非近似。
- eager_N4_control pos2 仍 279 → 印证缺陷专属捕获路径,hook 补的就是缺失接线。
- NPU5 已释放,无泄漏进程。⟹ V1 ✅ 通过。docs/sew-offload/12 §6 V1 标 ✅。
- 论文 before/after 对照图(before_hook/ vs SCALE_*.log)数据齐备。

## 续11（2026-06-16）：V2 ledger 实测通过 + Regime B 难点勘验

- V2(NPU5,cap_N4 配置 + SEW_OFFLOAD_LEDGER=1):每个 offload 层 {2,3,4,5}
  **log2phy_staged=128/128**(无 -1 残留)→ hook 确实把全专家映射写进持久 buffer。
  OUTPUT_TOKENS 再次 == BASE。⟹ V2 ✅,Regime A 验证表(V0/V1/V2/V3/V4 除 V4)收口。
- HBM 账本(Qwen3-30B-A3B,num_slots=128):每层每副本 1.125 GiB;slot_bank ==
  host_store == original_weight(逐层相等)。⟹ Regime A 同层在 NPU 同时有
  original + slot 两副本(未 release 时 2.25 GiB/层),**不省 HBM 反增**;全 48 层
  offload 仅 slot_bank 即 54 GiB > 64GB 卡。**这是 Regime B(淘汰式真 offload)的动机数据。**
- Regime B router 暴露勘验(读代码 qwen3_moe.py:226 SparseMoeBlock.forward + fused_moe
  apply:235 select_experts):gate(ReplicatedLinear)+top-k 廉价,但**第 N 层 router 输入
  = 第 N-1 层 MoE 输出(在捕获图内)**。⟹ "replay 前一次性预跑全层 router"不可行
  (拿不到深层输入),除非 eager 跑完整模型(=最慢的两遍 forward)。这正是 Regime B
  核心难点,非工程小补丁,印证论文贡献定位。
- 下一步候选(Regime B):①逐层 capture 段切分(每 MoE 层单独 capturable,层间 eager
  做 router 预跑+stage) ②上一步 routing 预测下一步(时间局部性,fail-closed 回退 eager)。
  需进一步读 acl_graph 切分粒度 + phase_split.py 现有原型再定。

## 续12（2026-06-16）：Regime B 路径① 阶段1落地(custom op 骨架,纯CPU)

- 新建 vllm_ascend/ops/fused_moe/moe_offload_stage_op.py:注册 vllm::moe_offload_stage
  custom op(direct_register_custom_op,dispatch_key=PrivateUse1)。
  - 签名:moe_offload_stage(topk_ids, layer_id, num_logical_experts) -> Tensor。
  - 返回 topk_ids.clone() 强制下游 grouped MLP 数据依赖(fx 不重排/不消除)。
  - impl:capturing→clone(no-op);未注册/非fixed-slot层→clone(透传);否则 D2H
    unique(topk_ids)→stage_fixed_slot_plan 写持久log2phy→clone。
  - fake_impl 返回 empty_like(topk_ids)。
- 单测 +5(test_graph_compatible_offload.py):registered / pass_through_unregistered /
  noop_during_capture / stages_active_set(active{1,2,3}→log2phy>=0,地址稳定) /
  fail_closed_when_active_exceeds_slots。全文件 16/16 通过(0.17s)。
- 边界:仅落地 op+fake+CPU单测。**未接线 fused_moe.apply、未加入 splitting_ops**。
  零运行时影响、完全可逆。
- 阶段2(待用户拍板,动编译路径+耗NPU):①在 fused_moe.apply 图兼容分支把 select_experts
  输出经此 op 再喂 fused_experts;②platform.py:436 旁 splitting_ops.extend
  (["vllm::moe_offload_stage"]);③NPU5 捕获验证 R3——piece 是否在此 op 处断开、
  eager 段 D2H/H2D 是否合法、token 是否==BASE。

## 续13（2026-06-17）：Regime B 路径① 阶段2 接线 + R3 NPU 首测

接线(SEW自有文件,默认关闭):
- envs.py:+VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM(默认0)。
- config.py:+offload_stage_seam 字段 + from_env。
- fused_moe.py:import op(注册);load-time & lazy 两处 stage_full_residency 钩子在
  seam on 时跳过;apply 在 offload_enabled && seam 时 topk_ids=torch.ops.vllm.moe_offload_stage(...)。
- platform.py:436 旁 seam on 时 splitting_ops.extend(["vllm::moe_offload_stage"])。

NPU1 实测(Qwen3-30B-A3B,128 experts top_k=8,nonres={2,3,4,5}):
- **R3-a slots=128**:LOAD_OK+GENERATE_OK,**capture 成功**(config dump 确认
  splitting_ops 含 vllm::moe_offload_stage;"Capturing ... PIECEWISE" + "Replaying aclgraph")。
  → **FX 切分机制 R3 成立**:自定义 op 被splitter接受、能作为捕获 piece 边界。
  但 **tokens 错**:[3555,525,279,1887,11,279,279,374] vs BASE[...279,22146,323,63625,315,1667],
  pos=3 起发散 + 重复("the the")= 与 -1 mis-route 同签名。→ eager staging 未达捕获 gather。
- **R3-b slots=16**:fail-closed "active expert working set size 51 exceeds num_slots=16"。
  prefill 512 token 的 expert 并集=51 ≫ decode top_k=8 ≫ 16 slot。证实 fixed-slot Regime B
  需 eager-prefill 回退或淘汰(=R4),guard 正确未静默corrupt。

数据流定位:捕获 piece B 的 gather 经 capture_safe_slot_weights 读**持久 log2phy buffer
固定地址**(moe_comm_method.py:342)。正确性要求 eager seam op 在 piece A replay 与 piece B
replay 之间把决策写入该 buffer。tokens 错 → 假设:seam op 未真正 eager 跑(疑被捕获进 piece
成 no-op clone)。已加 SEW_SEAM_PROBE 探针重跑 R3-a 取证(capturing vs eager_staged 计数)。

## 续14 — Option B 三段拆分设计稿 (评审门控)
- 执行用户 "执行B"。先彻底读 moe_forward 调用链 (代码事实):
  - AscendFusedMoE.forward(fused_moe.py:610) -> runner.forward -> MoERunner.forward(core:531)
    -> _forward_entry=torch.ops.vllm.moe_forward (不透明 op) -> _forward_impl -> AscendMoERunner.forward_impl -> layer.forward_impl(fused_moe.py:615)
  - 决定性事实1: AscendMoERunner(MoERunner) override forward_impl/_forward_impl, 但未 override forward/_select_forward
    => 三段编排可全落 vllm-ascend (override AscendMoERunner.forward), 不碰 vllm core moe_runner.py。
  - 决定性事实2: select_experts 在 quant_method.apply 内 (fused_moe.py:690), 即不透明 op 内 => 拆分最难子任务=把 router 提到顶层。
  - 脚手架已就位: MoEFusedExpertsInput.topk_weights/topk_ids (moe_stage_contracts.py:68) 已预期传入预算 topk。
- 产出 docs/sew-offload/13-moe-forward-split-design.md (状态: 设计评审中, 未实现)。
  拓扑: vllm::moe_router(captured) | vllm::moe_offload_stage(splitting/eager) | vllm::moe_mlp(captured)。
  全程 default-off (VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=0 走 super().forward() 回退)。
- 5 个待评审决策: ①覆写子类 forward 不碰 core; ②select_experts 外提 B1(topk 注入短路,推荐)/B2/B3; 
  ③stage 置于 prepare 前; ④首版与 multistream_gate 互斥; ⑤首版仅 _shared_experts is None。
- 验证矩阵 V-A..V-E; V-D(decode 每步 eager staging probe≠R3 的0行)/V-E(graph dump 顶层节点) 为成败判据。
- 等用户评审确认后再进 P1。未改任何 vllm core / AscendMoERunner。

## 续15 — P1: moe_router op 落地 (评审通过后)
- 6 决策拍板: ②=B1(topk 注入短路); ⑥=首版限单卡 identity-prepare; ①③④⑤=默认。
  - 决策⑥ 依据(代码实测): 标准路径 select_experts 吃 post-prepare logits;AllGather prepare
    多卡时改写 logits(prepare_finalize.py:354/395-399)。单卡 TP=DP=PCP=EP=1 时 prepare 对
    logits 恒等(dp_size>1 False;ep all_gather 恒等)=> router-before-prepare 逐位等价。
    匹配已验证服务命令。多卡(prepare 折进 router op)列后续。
- 新增 vllm_ascend/ops/fused_moe/moe_router_op.py: 注册 vllm::moe_router(不透明, PrivateUse1)。
  忠实包装 apply 路径 select_experts 调用(fused_moe.py:244-257), custom_routing_function 钉死 None。
  fake impl 返回 (num_tokens,top_k) topk_weights(logits.dtype)+topk_ids(int32)。
- 新增 4 UT(test_graph_compatible_offload.py): registered / forwards_every_arg_1to1(关键字集逐项核)/
  bit_equivalent_to_direct_select_native(native 路径逐位等价=V-B CPU 半)/ fake_shapes_and_dtypes。
- 全量 20/20 绿(16 旧 + 4 新)。未接线 runner.forward(P2)。未碰 vllm core / AscendMoERunner。

## 续16 — P2a/P2b: moe_router_indirect + moe_mlp + B1 注入 (主路径前置件)
- 决策: 三段 op 全用 layer_name 间接查表(_USE_LAYERNAME=False, torch 2.9)。索引安全关键:
  _encode_layer_name 的 "from_forward_context" 哨兵会自增 moe_layer_index;真实 layer 名查表不增。
  => P2c seam 入口须解析真实 layer 名一次, 传给三个 op(均不增), 替换单个 moe_forward 索引一致。
- P2a: moe_router_op.py 新增 vllm::moe_router_indirect(解析 layer, 读 apply 路径同源标量
  top_k/use_grouped_topk/.../e_score_correction_bias, 用 get_moe_num_logical_experts 同法算
  num_logical_experts, 委托显式标量核)。custom_routing_function!=None fail-closed。
- P2b: moe_seam_inject.py(layer_id->topk 注入注册表, moe_mlp 设/清, finally 清)。
  moe_mlp_op.py 新增 vllm::moe_mlp(解析 layer, 注入 topk, 调与 _moe_forward 同款 runner._forward_impl,
  finally 清; tuple 返回=带共享专家 fail-closed)。
  fused_moe.py apply select-site(244-257 后)加 B1 短路: has_injected_topk 则消费注入 topk, 否则原
  select_experts。注册表空(seam 关)=字节不变。imports: moe_router_op/moe_mlp_op/moe_seam_inject。
- UT: 29/29 绿(23 + 6 新: mlp registered/注册表默认空/set-peek-clear/设清环绕 forward/异常仍清/
  带共享专家 fail-closed)。fused_moe 导入 OK, 三 op 全注册。
- 未碰 vllm core / AscendMoERunner.forward(_select_forward)。P2c 接线待做。

## 续17 — P2c: AscendMoERunner._select_forward 接线三段 seam (主路径, 默认关)
- 接线点: 覆写 AscendMoERunner._select_forward。config 守卫过 -> 返回 _seam_forward_entry,
  否则 super()._select_forward()(原 opaque moe_forward 不变)。
- _seam_config_guards_pass(__init__ 期可查): offload_stage_seam 关 / _shared_experts!=None(决策⑤) /
  runner.gate!=None(否则 _forward_impl 内重算 logits 致 stale topk) / dp|ep|tp|pcp>1(决策⑥单卡 identity-prepare)
  任一 -> False。__init__ 顺序已核(moe_config@217/gate@222/_shared_experts@226/layer_name@241 均在 _select_forward@243 前)。
- _seam_forward_entry: 首调 _resolve_seam_per_layer_guards 解析真实 layer 一次并缓存 _seam_active;
  失败 -> 永久回退 torch.ops.vllm.moe_forward(与 base 同)。通过 -> moe_router_indirect -> moe_offload_stage
  (splitting/eager, 返回 topk_ids clone 建数据依赖防 DCE/重排) -> moe_mlp, 三 op 均传真实 self.layer_name。
- _resolve_seam_per_layer_guards 逐层 fail-closed: custom_routing_function!=None / multistream_overlap_gate /
  enable_npugraph_ex_static_kernel(moe_layer_index 读者, 须关保索引安全) / zero-expert;
  通过则缓存 _seam_layer_id + _seam_num_logical_experts(get_moe_num_logical_experts 同法)。
- 新增 13 UT: select_forward seam 关用 base / seam 开单卡选 seam / 带 shared|gate 回退 /
  4x 多卡(dp|ep|tp|pcp)回退 / per-layer 守卫 OK 缓存 layer_id+n / 3x fail-closed(custom_routing/
  multistream/static_kernel) / seam 入口守卫失败永久回退且 3 次全走 moe_forward(_seam_active=False)。
- 全量 42/42 绿(29 + 13)。未碰 vllm core。NPU 验证(P2d V-C/V-D/V-E)待做。

## 续18 — P2d: NPU4 实测 P2c 三段 seam (Regime A V-C/V-E 决定性通过)
- 三个根因→修复(全代码事实坐实):
  1. gate 守卫使 seam 全模型死代码(run1 mis-route pos3): 移除 _seam_config_guards_pass 的
     `if self.gate is not None: return False`; moe_router_indirect 补 `runner.gate(hidden_states)`
     (Qwen3 internal-router 传 placeholder logits)。
  2. Regime A 漏 log2phy 全量 staging(run3 MTE 越界): fused_moe.py:192,362 原 seam-gated 跳过
     load-time stage_full_residency_slot_plan + per-step seam 全覆写砸回 -1。改 regime-gated:
     新增 runtime.is_static_residency_regime(n)=(num_slots>=n); Regime A 无条件 load-time 全量 staging
     + seam op 在 Regime A 是 no-op(EAGER_PASSTHROUGH reason=regime_a); Regime B 才 per-step staging。
  3. 【主根因】splitting op 返回 clone 破坏固定地址契约(run4 log2phy 128/128 仍崩):
     moe_offload_stage 原 return topk_ids.clone() → 每次 eager replay 落不同地址,捕获 moe_mlp 读冻结
     capture-期地址 = stale → MTE DDR out-of-range 507011。改 mutate-in-place+return None
     (mutates_args=["topk_ids"], 同 unified_attention_with_output 契约),同一 topk_ids 直穿 moe_mlp。
     call site 不再 reassign。
- NPU4 实测(r3a_slots128, nonres={2,3,4,5}):
  - V-C: tokens==BASE 且 pos0-7 全 top-20 logprob 逐位等于 BASE 到 1e-5 (真数值等价)。
  - load-time fill: offload 层 {2,3,4,5} 全 log2phy_staged=128/128。
  - V-E: 首次 Replaying aclgraph 后 336 EAGER_PASSTHROUGH = 48 层×7 步, 证 seam 每层每步 eager 跑
    于捕获 router/mlp piece 间 (R3 此处为 0)。
- UT 45/45 绿(+ is_static_residency_regime predicate + Regime-A no-op; stage op 改 None-return 契约,
  4 个 stage UT 同步更新)。脚本 tools/sew_offload/run_p2d_seam_validation.sh (DEV=4)。
- 未碰 vllm core / scheduler / model_runner 主路径。NPU4 释放无泄漏。
- 下一步: Regime B (V-D 真 per-step staging) 需 R4 eviction (prefill active 并集 > num_slots
  触发 working-set 守卫), 独立下游工作。

## 续19 — P2d V-D: 真 Regime B per-step staging 达成 + Tier-1 假设证伪 (NPU4)
- 续18 "V-D 需 R4 eviction" 表述被代码事实纠正。两处误判:
  1. LRU 淘汰早已存在 (slot_bank.py:122 _lru_evictable_slot); 缺的从来不是淘汰。
  2. 真根因: 三路 seam 的 moe_offload_stage 无条件调 stage_fixed_slot_plan (fused_moe.py:599),
     从不查 decide_layered_path → prefill ~51 active 直撞 runtime.py:538 working-set 守卫。
- Tier-1 修法 (seam 接 decide_layered_path, prefill 高 fanout 走 FULL_WEIGHT_PATH eager 原始权重)
  在架构上对 offload 层【不可行】: 子代理坐实 fused_moe.py:107-118,153-157 —— 非驻留层 w13/w2 在
  load 时被搬到 CPU, 只 gate 在非驻留、与 release_original_expert_weights 标志无关。offload 层 NPU 上
  只有 num_slots 份 slot 权重、无完整副本。decide_layered_path 的 FULL_WEIGHT_PATH 用 "未释放" 当
  "NPU 可用" 是潜伏 bug; 我的改动首次点燃 → slots=16 run 崩 "weight is on cpu" (grouped matmul
  吃到 CPU w13)。已回退该改动。
- 正确框架: offload 层任何计算(prefill+decode)只能过 slot bank, 单次 forward 最多 num_slots 专家。
  Regime B 分两子区:
  - B1 (每次 forward 的 active 并 ≤ num_slots < n): prefill(并~51)与 decode(≤8)各自 stage 进 slots,
    全程 NPU, 无需 FULL_WEIGHT —— 纯配置即可 (num_slots ≥ 单次最大 fanout 且 < n)。
  - B2 (num_slots ≪ prefill 并集): prefill 必须分波流式 (每波 stage ≤num_slots + partial grouped
    matmul + 累加)。phase_split.py 现成 execute_phased_mlp/plan_hit_miss_phases 已接 live 路径, 但
    现仅 hit/miss 二相、非容量分波, 需新 capacity-bounded planner。
- V-D NPU 实测 (B1, NPU4, num_slots=96<n=128, nonres={2,3,4,5}):
  - ledger 算得 slot_bank 持 96 专家、host_store 持 128 → 32 专家永不驻留 (真 Regime B)。
  - staging 序列 (SEW_SEAM_PROBE, EAGER_STAGED 共 40): 2 轮 warmup(n_active=8) → 真 prefill 每层并集
    n_active=51/45/43/39 (≤96 全 staged、n_mapped==n_active、跑 NPU slots) → 7 解码步×4 层=28 次
    per-step staging(n_active=8) = V-D 核心。
  - 数值等价: tokens==BASE [3555,525,279,22146,323,63625,315,1667]; pos0-7 全 top-20 logprob 逐位
    等于 BASE 到 1e-5 (pos0 3555:-0.92324、pos3 22146:-1.44021、pos5 63625:-0.08265 两侧一字不差)。
- UT 47/47 绿 (stage op 回干净无条件 staging + 清晰 fail-closed; 新增 B1 fit→stage / B2 exceed→
  fail-closed 两例)。脚本新增 run_seam b1_slots96 96。NPU4 跑后 "No process in device" 无泄漏。
- 未碰 vllm core / scheduler / model_runner 主路径。
- 下一步: B2 容量分波 prefill (真 HBM 节省故事) + V4 resident 全路径回归。

## 续20 — 下一步设计源码核对（只读分析）
- 已对照源码复核 V-D 后的设计边界：
  - `slot_bank.py` 已有 READY slot LRU eviction；B2 的缺口不是淘汰，而是单次 active 并集大于 `num_slots` 时缺少 capacity-bounded wave execution。
  - `runtime.py` 的 `prepare_fixed_slot_plan`/`stage_fixed_slot_plan` 仍以“本次调用 active experts 必须一次装入 slot bank”为不变量；B1 正是满足该不变量，B2 必须改为多波。
  - `moe_offload_stage_op.py` 当前正确契约是 side-effect-only、`returns None`、`mutates_args=["topk_ids"]`；顶层 docstring 与 `moe_mlp_op.py` 仍残留“clone”旧叙述，后续应随设计文档一并修正。
  - offload 层 processed `w13/w2` 在 load 后会被搬到 CPU；因此 fixed-slot 非驻留层没有安全的 FULL_WEIGHT_PATH，`decide_layered_path` 里“未 release 即 full weights available”的旧判定应先收口为 device-resident 判定或对 non-resident offload 层禁用。
  - `phase_split.py` 已有 post-dispatch token slicing / phase group_list / scatter 等价性底座，但现有 planner 是 hit/miss 二相，不是容量分波；B2 应复用执行骨架，不应直接复用 planner。
- 初步建议顺序：先做 V4 resident/full-path 回归 + FULL_WEIGHT_PATH guard 清债，再启动 B2 prefill capacity waves。B2-MVP 只覆盖 eager prefill active 并集 > slots；decode 仍走已验证 B1 seam。 

## 续21 — P0: B1 vs 全驻留延迟实测 + decode 瓶颈归因 (NPU4, Qwen3-30B-A3B)
背景: 续19 只证 B1 数值正确, 无延迟数据。P0 用差分法补测 (T(1)≈TTFT, TPOT=(T(N)-T(1))/(N-1),
MAXTOK=32 REPS=5 取中位数), seam 内 env-gated synchronize 包夹打 STAGE_MS。不碰主路径。
- 探针/脚本 (全 env-gated, 默认关, 可逆):
  - moe_offload_stage_op.py: SEW_SEAM_PROBE 分支加 torch.npu.synchronize 包夹 + STAGE_MS。
  - run_graph_compat_capture_probe.py: --latency/--latency-repeats 差分 TTFT/TPOT/DECODE_TPS。
  - tools/sew_offload/run_p0_latency.sh: base(offload OFF) + b1_slots96。UT 47/47 绿。
- 宏观实测 (base 全驻留 vs B1 slots=96, nonres={2,3,4,5}):
  - TTFT: 190ms → 319ms (+129ms, +68%)
  - TPOT: 44.5ms → 73.7ms (+29.2ms, +66%)
  - decode 吞吐: 22.5 → 13.6 tok/s (−40%)
- 微观归因 (decode staging 双峰分布, 876 次调用):
  - 92% (806) <2ms = 命中(slot 已驻留, 仅 bookkeeping + 诊断 synchronize 开销)。
  - 8% (67) ≥10ms = 真 miss 同步 H2D, sum=6062ms, 单次均~90ms。
  - mean=7.94ms; 4 offload 层 × 7.94 = 31.8ms ≈ 实测 TPOT delta 29.2ms。
    ⇒ decode 变慢【几乎全部】由 seam 同步 per-step staging 解释, 即 T_overlap=0、miss 的 H2D
    全额砸在关键路径上 (实锤了之前"offload 现在一定比全驻留慢"的判断)。
  - prefill-class (n_active>8) n=48 mean=246ms, max=3601ms(首调用冷启动离群), 解释 TTFT +129ms。
- 方向数据支撑 (硬数):
  - P3 热专家钉死: 砍 8% miss 率即直接抹掉 ~6062ms 大头, 高杠杆已被数据坐实。
  - P1 overlap: 单 miss ~90ms H2D 若与计算重叠即移出关键路径; 命中下限 ~4×1ms=4ms/步,
    ⇒ TPOT 惩罚 29ms 里约 25ms 是可被 overlap/pinning 回收的 miss-H2D。
  - 注意: 本测为 ASYNC_LOAD=0 同步路径 = overlap 能回收的【上界基线】(worst case)。
- NPU4 跑后 "No running processes" 无泄漏。未碰 router/主路径。

## 续22 — P0 修正: 正确基线是 eager 单算子 offload (非全驻留), CPU offload 大小固定
续21 用全驻留作基线方法论有误(把 offload 代价与图捕获收益混在一起)。正确对照: **同 CPU offload
footprint 下 eager 单算子 vs B1 图捕获**, 隔离图捕获本身贡献。NPU4 跑 eager-singleop(slots=96,
nonres={2,3,4,5}, enforce_eager, STAGE_SEAM=0, GRAPH_COMPATIBLE=0)。
- HBM footprint 三跑实测(model_runner_v1 "Loading model weights took"): base 全驻留 56.90GB /
  B1 55.78GB / eager-singleop 55.78GB ⟹ **B1 与 eager 完全同 footprint**, 唯一变量=图捕获。
- slots=96 显存拆解(每专家 9MiB bf16=3×2048×768×2B): offload 4 层 slot_bank=3.375GiB +
  resident 44 层全权重 49.5GiB = 52.875GiB(全驻留 54GiB, 省 1.125GiB=4层×32逐出×9MiB);
  host_store on CPU=4.5GiB。实测 55.78 vs 56.90 差 1.125GB 与账本逐字吻合, 印证 offload 层 release NPU 原件。
- 三跑延迟(MAXTOK=32 REPS=5):
  | 配置 | TTFT | TPOT | decode tok/s | 图捕获 |
  | base 全驻留      | 190ms | 44.5ms  | 22.5 | 是 |
  | B1 offload       | 319ms | 73.7ms  | 13.6 | 是 |
  | eager 单算子 off | 233ms | 201.3ms | 4.97 | 否 |
- **核心结论(用户要的对照)**: 同 HBM footprint, B1 图捕获 decode 比 eager 单算子 offload **快 2.73×**
  (TPOT 201.3→73.7ms; tok/s 4.97→13.6)。这是"图兼容 offload"的真实贡献。
- 机理: eager TPOT 201 ≫ base 44.5 = 48 层算子逐个 dispatch 的 launch overhead; 图捕获塌掉它。
  B1=图捕获 launch 已塌(44.5) + 29ms 同步 staging tax(续21 归因)。
- eager run 真实性: 920 条 SEW_PROBE branch=EAGER(n_active=8)、0 CAPTURE_SAFE ⟹ 真单算子 offload 非退化。
- TTFT B1(319)>eager(233): B1 prefill 含 seam staging + 首次 captured-graph replay 建立; decode 决定稳态吞吐。
- 探针 env-gated 默认关; 未碰 router/主路径。NPU4 跑后 shutdown complete 无泄漏。

## 续23 — P0 干净复测 (去掉 synchronize 混淆) + prefill/decode 非对称归因
续22 的 B1 跑带 SEW_SEAM_PROBE=1 → seam 内 torch.npu.synchronize 包夹只拖慢 B1、eager 没带 =
混淆变量。新脚本 run_p0_clean_compare.sh 两跑均【无探针】, 唯一变量=enforce_eager(图捕获开关)。
逐项核对两跑 non-default args 完全一致: prompt="Briefly explain mixture-of-experts models.",
max_num_seqs=1(并发1), max_tokens=32, max_model_len=512, kv=256MiB, bf16; footprint 均 55.78GB。
- 干净三量 (slots=96, nonres={2,3,4,5}):
  | 配置 | TTFT(prefill) | TPOT(单步decode) | decode tok/s |
  | eager 单算子 off | 242.8ms | 212.3ms | 4.71 |
  | B1 captured off  | 308.0ms | 69.1ms  | 14.5 |
  | 比值 | B1 +27%(劣) | **B1 快 3.07×** | 3.07× |
- 混淆量化: B1 probed→clean: TPOT 73.7→69.1(−4.6ms/6%), TTFT 319→308(−11ms)。去混淆后 B1 更优
  (decode 倍数 2.73→3.07×)。eager probed/clean 一致(201/212 在 run jitter 内)= eager 本无混淆。
- 差分法语义澄清: T1(max_tokens=1)=一次 prefill forward(首 token 出自 prefill 末位 logits、无独立
  decode forward)=TTFT; TPOT=(TN-T1)/(N-1)=一次 decode forward。⟹ TTFT 量 prefill、TPOT 量单步 decode。
- **用户问题答 (为何 decode 降、prefill 升)**:两阶段在 ACLGraph 下命运不同——
  1. decode 固定形状(b=1,s=1)【被捕获】: 捕获塌掉 48 层逐算子 launch 开销, eager 212→captured 69ms,
     省的 143ms 是纯 launch 消除 = 图兼容 offload 核心价值。
  2. prefill 变长【不被捕获】: 两配置 prefill 都跑 eager。B1 prefill 还多过 seam(unique().cpu() D2H +
     同步 staging ~178 专家并集 51/45/43/39), 且 warmup/上一次 decode 用 LRU 把这些 prefill 专家挤出
     96 槽 → 每次新 prefill 重搬 → +65ms vs eager prefill。⟹ prefill 无捕获补偿、只承担 staging 额外开销。
  证据: V-D probed run 实测 prefill EAGER_STAGED n_active=51/45/43/39(eager)+ decode 捕获 replay。
- 端到端(32 token): B1 ≈ 308+31×69.1=2450ms vs eager ≈ 242.8+31×212.3=6824ms ⟹ B1 端到端快 2.79×。
  输出越长 decode 占比越大 B1 越赢; 极短输出(TTFT 主导)B1 略亏。
- ⟹ P2(预测/重叠预取)价值被指明: prefill 那 65ms TTFT 退化=prefill 专家被 decode 挤掉每次重搬,
  prefill 阶段提前/重叠 staging(B2 分波+异步)可压回。NPU4 两跑后 "No process in device" 无泄漏。

## 续24 — B2 容量分波 prefill: 实现 + NPU 验证 (用户选独立路径, 分小步)
目标: 让服务命令真实工作点 (num_slots 小 ≪ prefill 并集) 能跑 offload, 不再 fail-closed。
B2 分波只在 eager prefill; decode/现有 prefill/主路径全不变; env-gated 默认关。
- 架构发现: B2 不能只在 MLP 执行器层做。offload prefill 三处"单遍全专家"假设 + fail-closed 在
  MLP 前 (prepare_fixed_slot_plan runtime.py:538)。token_dispatch 用 log2phy[topk_ids] 单遍重排,
  51 逻辑专家映射不到 8 槽。⇒ 独立 B2 prefill 路径: 每波局部 dispatch+stage+matmul+combine+累加。
- 数学 keystone (CPU, fp32): token MoE 输出=Σ_{e∈topk} gate·expert_e, 加法可结合 ⇒ 按专家分不相交波
  +每波掩码+跨波累加 == 单遍。多组测试逐位==单遍到 1e-6。
- 实现 (全 CPU 单测先行, 默认关):
  - phase_split: plan_capacity_bounded_phases (⌈N/slots⌉ 波) + WaveStager 两段式 issue/wait
    (overlap-ready, buffer_count, prefetch_depth 软件流水, 现实现串行) + build_b2_wave_routing
    (offload 路径 physical_topk_ids -1→slot0 + 该位权重置0; offload 不自动 zero 故显式). 59 例绿。
  - config/envs: VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL (默认0) + config.b2_wave_prefill。
  - runtime.should_use_b2_wave_prefill (config∧prefill∧offload固定槽∧fanout>num_slots). 7 例绿。
  - moe_comm_method: fused_experts 顶部 _maybe_run_b2_wave_prefill 早分支 (默认关零开销, gate 命中才进,
    否则 return None 走原路径)。_run_b2_wave_prefill 切波 + 每波 prepare_fixed_slot_plan(wave) stage +
    _run_b2_single_wave (log2phy remap + build_b2_wave_routing 掩码 + 现有 dispatch/matmul/combine) +
    累加 routed_out。全 moe_offload UT 262 passed/1 skipped 无回归。
- NPU5 验证 (eager 非 seam, num_slots=8 ≪ 并集51, nonres={2,3,4,5}):
  - B2 真触发: WAVE_RUN layer2 51→7波 / layer3 45→6 / layer4 43→6 / layer5 39→5
    (num_slots=8 远小于 51 = B1 必 fail-closed 的工作点, B2 正常跑)。
  - tokens 8/8 逐位==BASE; 全 8 位 top-1 一致。
  - 但 logprob【非】1e-5 位等: top 档 ~0.08-0.16 nat 漂移 (pos6 尾部 token 0.42)。
    原因(非 bug): captured-B1 跑【同一次】slot-packed matmul→逐位等; B2【重结合】单次 matmul 为
    7 个 bf16 求和波, bf16 加法非结合 ⇒ 必然漂移。CPU keystone 已证 fp32 算法精确 1e-6; NPU 漂移
    纯 bf16 reduction order, 量级 ~0.5-1%, top-1 margin~1nat 远大于漂移 ⇒ 无 token 翻转。
  - 结论: B2 正确性 = "fp32 精确 + NPU top-1 全保持 + 漂移与 bf16 重结合一致", 是 wave-streaming
    把单次 reduction 拆多波的固有代价, 非缺陷。NPU5 无泄漏。
- 下一步(可选): fp32/强制单波进一步隔离漂移; B2-with-seam 图捕获集成; autoconfig 接线; README。
