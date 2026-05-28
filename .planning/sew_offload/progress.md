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
