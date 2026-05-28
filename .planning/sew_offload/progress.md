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
