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
