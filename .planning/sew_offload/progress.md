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