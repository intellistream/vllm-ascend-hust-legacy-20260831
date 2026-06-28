# ASC-DOCS 同步任务清单

> **仓库**: vllm-ascend | **专题包**: ASC-DOCS — 文档
> **PR 总数**: 93 | **已合并**: 54 | **待处理**: 39
> **策略分布**: pick: 85 | pick-later: 7 | ignore: 1

---

## 前置准备

| # | 任务 | 状态 |
|---|------|------|
| 1 | 确认 upstream remote 已配置并 fetch 最新 | ✅ |
| 2 | 确认 main 分支处于最新状态 | ✅ |
| 3 | 创建同步分支 | ✅ |

---

## 第 1 批：Bug 修复（5 PRs） 🔴 已完成

| PR | 标题 | 状态 | 处理方式 |
|----|------|------|----------|
| #8574 | [BugFix] msprobe data collection support aclgraph | ✅ | 手动合并：保留双方测试方法 |
| #9303 | [BugFix] flash_attn_v3 → flash_attn_npu_v3 | ✅ | 空提交：HEAD已含 |
| #9916 | [BugFix][DOC] Del engine_id and connector_path | ✅ | --ours：HEAD已无相关参数 + git rm |
| #9962 | [BugFix] Remove legacy capture-size pruning | ✅ | --theirs：移除遗留函数 |
| #10518 | [Attention][BugFix] Enable multistream_dsv4_dsa_overlap | ✅ | --theirs + git rm |

---

## 第 2 批：pick — 冲突待处理（32 PRs） 🟡

| PR | 标题 | 状态 | 处理方式 |
|----|------|------|----------|
| #9128 | [Misc] Upgrade torch-npu to 2.10.0 | 🔍 | 🔍 待确认：CANN 8.5.1 vs 9.0.0 / torch-npu 2.9.0 vs 2.10.0 |
| #9160 | [CI] Remove quantization e2e test case | ✅ | --ours：保留量化测试条目 + 保留被删的 test_quantization.py |
| #9298 | [Doc] Fix CANN 9.0.0 release-notes URL | 🔍 | 🔍 待确认：同#9128，依赖CANN版本 |
| #9271 | [Feature][Ops] Add A5 custom operator build support | 🔍 | 🔍 A5产品名待确认 |
| #7886 | [Ops][Feature] Add support for Qwen2.5-Math-RM-72B | ✅ | --theirs：models/index.md 纯追加2个新模型条目，其余4文件自动合并 |
| #8537 | [Doc] add Mixtral-8x7B-Instruct-v0.1 model docs | ✅ | 无冲突：纯新增模型文档和测试配置 |
| #9466 | [Doc] Correct the README file and link errors | ✅ | 手动合并：保留About This Fork + 追加supported models链接；版本冲突 --ours |
| #9201 | [Refactor] migrate compilation backend torchair→npugraph | ✅ | git rm + --theirs：torchair→npugraph_ex 纯重命名，2个测试+1个已删README |
| #9449 | [CI][Nightly] Add the external_dp test framework | ✅ | 无冲突 |
| #9344 | [Doc][Feature] Add Hy3-preview model tutorials doc | ✅ | --theirs：pyproject.toml 末尾追加2个模型文件路径，其余4文件自动合并 |
| #9572 | [1/N][Feature] Support FULL_AND_PIECEWISE | 🔍 | 🔍 待确认：8个冲突含核心代码，upstream含旧版update_aclgraph_sizes被#9962移除 |
| #9668 | [Ops][Misc] Remove VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL | ✅ | 手动：ascend_config.py --theirs但multistream_dsv4_dsa_overlap保持True；test --theirs |
| #7820 | [Feature] Mooncake kvpool usage optimization | ✅ | --theirs：Mooncake 初始化优化+fabric内存+ReplicateConfig，test 纯追加 |
| #9697 | Misc: introduce additional_config for dsa_cp | ✅ | add_config.md --ours（保留#10518条目），test_basic.py --theirs（新增测试参数） |
| #9807 | [CI] Refactor test folder | ✅ | 无冲突 |
| #9881 | [Doc][Misc] Update model-code converter writing guide | 🔍 | 🔍 待确认：upstream 全文重写（587行→469行），删除5章节+新增开发者视角+本地预览 |
| #9989 | [Doc] Translated Doc files | 🔍 | 🔍 依赖 #9881：doc_writing.po 冲突需等 #9881 全文重写决策；release_notes.po 纯翻译可 --theirs |
| #10017 | [Test][CI] final part for selected test | ✅ | 无冲突 |
| #9980 | [Doc]Add InternVL3.5 documentation | ✅ | 无冲突：纯新增模型文档，index.md 自动合并 |
| #9481 | [Doc] Update MiniMax-M2.5.md change A2 to single node | 🔍 | 🔍 待确认：HUST 用多节点还是单节点部署 MiniMax-M2.5？upstream 完全重写为单节点+Eagle3 |
| #9731 | [Feature] Add Mooncake SSD offload with embedded client | ✅ | --theirs：SSD offload 纯追加，per-rank 目录 |
| #10090 | [Doc] move example from installation to quick_start | ✅ | --ours：版本表保留 CANN 8.5.1，其余内容自动合并 |
| #10059 | [CI] add slash command dispatch | ✅ | 无冲突 |
| #10139 | [Doc] Translated Doc files | 🔍 | 🔍 8个.po文件冲突（时间戳+内容），依赖英文文档同步完成 |
| #10070 | [Doc][CI] Refine doctest workflow | ✅ | conf.py --ours（CANN 8.5.1），installation.md --theirs（多一个 fallback 镜像 URL） |
| #10269 | [Doc] Translated Doc files | 🔍 | 🔍 翻译文件，同 #9989 #10139，依赖英文文档同步 |
| #10292 | [Feature] add ascendc ops store_kv_block | 🔍 | 🔍 部分完成：torch_binding_meta --theirs + llm_base_proposer git rm；4个C++/Python核心代码 aborted，需人工 |
| #10034 | [Ops][Feature] add setup of batch_invariant_ops | ✅ | --ours：build_aclnn.sh 保留HEAD旧安装方式（upstream替换为subshell新流程，不确定兼容性）；其余7文件自动合并 |
| #9882 | [CI] Rename CI variant label a2→910b | ✅ | 无冲突 |
| #10344 | [Doc] Fix explanations for batch_invariant_ops | ✅ | 无冲突：文档修正，自动合并 |
| #10571 | [Doc][Misc] Update v0.21.0rc1 release notes | ✅ | --theirs：纯新增 v0.21.0rc1 发布说明和版本记录 |
| #10533 | [Doc] Correct product name A5→Ascend 950 | 🔍 | 🔍 待确认：产品名 A5→Ascend 950，需确认HUST环境 |

---

## 第 3 批：pick-later — 待验证（7 PRs） 🟢

| PR | 标题 | 状态 | 处理方式 |
|----|------|------|----------|
| #7804 | [Feature] Verify and support Qwen3-ASR-1.7B | ✅ | 手动合并：index.md --ours（HEAD已有4条更全），json只加Qwen3-ASR-1.7B（未加Qwen3-VL-4B-Instruct） |
| #9567 | [Doc] remove --async-scheduling from configs/docs | 🔍 | 🔍 14个冲突超阈值：13 modify/delete(git rm可解) + 1 content yaml；有1个文件修改 |
| #8368 | [Doc][Test] Add testable docs codegen framework | ✅ | git rm ×4 + index.md --ours（HEAD保留ascend_benchmark_runner+doc_writing） |
| #9955 | [Test] Move more test to selected way | 🔍 | 🔍 14个冲突超阈值：310p测试文件大规模目录重构，与HEAD目录结构不兼容 |
| #10027 | [Test] Update test coverage guide | ✅ | git rm ×4 + --ours ×2（CI镜像URL、pip命令、文档表述均保留HEAD） |
| #10576 | [CI] Fix /e2e command | ✅ | git rm ×4 + --theirs ×2（CI镜像构建版本号更新为上游新版本） |

---

## 冲突处理方式说明

| 处理方式 | 适用场景 |
|----------|----------|
| **手动合并** | 双方在同一位置添加了不同内容，都需保留 |
| **空提交（skip）** | HEAD 已包含 upstream 改动 |
| **保留 HEAD（--ours）** | HUST 特有定制，不应覆盖 |
| **采用 upstream（--theirs）** | 纯新增内容，HEAD 侧无相关改动 |
| **🔍 待确认** | 涉及版本号/产品名变更，需确认 HUST 环境 |

> **原则**：每个冲突先分析再处理，优先保护 HUST 侧定制内容。分两步：Step 1 分析+方案 → Step 2 审核通过后执行。

---

## 合并冲突风险文件（需特别关注）

以下文件被 3+ PR 修改，合并顺序影响冲突数量：

- [ ] `docs/source/_templates/Model-Deployment-Tutorial-Template.md` — 已合并 #8942 #9143 #9331 #9662 #9976
- [ ] `docs/source/installation.md` — #9128 #10090 #10070
- [ ] `docs/source/quick_start.md` — #10090（已合并 #9484 #9493 #9619）
- [ ] `docs/source/user_guide/support_matrix/supported_models.md` — #7886 #9344（已合并 #8942 #9995 #9889）
- [ ] `docs/source/user_guide/configuration/additional_config.md` — #9697 #10292（已合并 #8574 ✅ #9557 #10518 ✅）
- [ ] `README.md` — #9128 #9298 #9466（已合并 #9602 #10109）
- [ ] `README.zh.md` — #9128 #9298 #9466（已合并 #9602 #9976）

---

## 推荐处理顺序

按冲突影响范围和依赖关系排序：

### Phase 1：版本/基础配置（先处理，减少后续冲突）
1. #9128 — torch-npu 2.10.0 升级 🔍 版本确认
2. #9298 — CANN 9.0.0 URL 🔍 版本确认
3. #9466 — README 链接修正
4. #10090 — 文档结构调整
5. #10070 — doctest workflow

### Phase 2：模型文档
6. #7886 — supported_models 表
7. #8537 — Mixtral 文档
8. #9344 — Hy3-preview
9. #9980 — InternVL3.5
10. #9481 — MiniMax-M2.5
11. #9881 — model-code converter guide

### Phase 3：功能/配置
12. #9572 — FULL_AND_PIECEWISE
13. #9697 — dsa_cp 配置
14. #10292 — store_kv_block
15. #10034 — batch_invariant_ops
16. #10344 — batch_invariant_ops 文档

### Phase 4：平台/重构（敏感）
17. #9201 — torchair→npugraph
18. #9668 — 移除 context_parallel
19. #9271 — A5 算子 🔍 产品名
20. #10533 — A5→Ascend 950 🔍 产品名

### Phase 5：Mooncake/KV
21. #7820 — kvpool 优化
22. #9731 — SSD offload

### Phase 6：CI/测试
23. #9160 — 移除量化 e2e
24. #9449 — external_dp 框架
25. #9807 — 重构测试目录
26. #10017 — selected test
27. #10059 — slash commands
28. #9882 — a2→910b 重命名

### Phase 7：翻译 & 发布
29. #9989 — 翻译文件
30. #10139 — 翻译文件
31. #10269 — 翻译文件
32. #10571 — v0.21.0rc1 release notes

### Phase 8：pick-later（低优先级）
33-39. #7804 #9567 #8368 #9955 #10027 #10576

---

## 最终审查清单

- [ ] 所有 pick PR 已同步
- [ ] 无损坏链接
- [ ] 单元测试通过
- [ ] CI 流水线通过
- [ ] PR 已创建并推送
