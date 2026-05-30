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
