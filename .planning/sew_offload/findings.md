# SEW-Offload 发现记录

## 仓库上下文

- 当前仓库：`/root/vllm-ascend-hust`
- 当前分支：`research`
- 当前 `git status`：干净。
- `paper/` 和 `slide/` 目录存在，但当前为空。
- 项目规则文件存在：`CLAUDE.md` 要求先阅读 `AGENTS.md`。

## 工程规则发现

- 新环境变量必须定义在 `vllm_ascend/envs.py` 的 `env_variables` 字典中。
- 新功能需要 UT，位置通常在 `tests/ut/`；NPU 相关路径需要 e2e 或实际硬件验证。
- 模型相关行为优先通过 patch、继承或组合接入，不应直接添加上游模型文件。
- `model_runner` 变化需要严格架构审查，因此 SEW-Offload 初期应避免修改 model runner 主路径。
- `tensor.item()` 在 NPU hot path 会导致同步，应避免作为在线路径控制手段。

## MoE 代码边界

- Python MoE 路径集中在 `vllm_ascend/ops/fused_moe/`：
  - `fused_moe.py`
  - `token_dispatcher.py`
  - `moe_mlp.py`
  - `moe_runtime_args.py`
  - `prepare_finalize.py`
- C++/Ascend C MoE kernels 已存在：
  - `csrc/moe_grouped_matmul`
  - `csrc/moe_dispatch_normal`
  - `csrc/moe_combine_normal`
  - `csrc/moe_gating_top_k`
  - `csrc/moe_init_routing_custom`
  - `csrc/dispatch_ffn_combine*`
- 已有 routed expert capture patch：
  - `vllm_ascend/patch/worker/patch_routed_experts_capturer.py`
  - 可作为 MVP-0 routing trace 的低侵入参考。

## 研究定位

- GPU MoE offloading 的旧控制面主要是 expert cache / CPU-GPU prefetch / CUDA expert execution。
- Ascend 新控制面是 static graph replay、AIC/AIV 分工、MTE/显式搬运、固定 GM window、NUMA-aware H2D。
- SEW-Offload 的核心思想是不改 router，复用现有 grouped MoE 的 per-expert count 表示，把动态 expert 工作集映射到固定 HBM expert slots 与可选 activation capacity windows。

## 新增确认

- 参考 slide `moe_serving_report.tex` 采用中文技术叙事，结构是：问题定义、观察、重要性、已有工作不足、关键想法、系统设计、实验计划；SEW-Offload slide 应复用这个节奏。
- `vllm_ascend/ops/fused_moe/fused_moe.py` 中 `AscendFusedMoE.forward_impl()` 已经把 prepare、expert selection、fused expert execution、finalize 串起来；SEW-Offload 的低侵入点应落在 expert execution boundary，而不是 scheduler/model runner。
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py` 已有 typed runtime contracts；SEW-Offload 应新增自己的 contract/config，不污染现有 MoE contract，除非进入稳定阶段。
- `tests/ut/ops/test_fused_moe.py` 和 `tests/ut/ops/test_moe_runtime_args.py` 提供了 mock NPU op、contract builder 的测试风格，可作为 SEW-Offload UT 模板。
- 已有 `tests/e2e/multicard/2-cards/test_qwen3_moe_routing_replay.py` 用 `enable_return_routed_experts=True` 检查 Qwen3-30B-A3B 的 routed expert replay；MVP-0 可以利用这个思路验证 trace 通路。
- vLLM Ascend 本地文档确认 graph mode 通过 ACLGraph/NPUGraph capture replay，static kernel 是显式启用的固定 shape 编译优化；weight prefetch 文档确认 Ascend 已经有通过额外 pipeline 预取线性层权重、缓解 MTE 压力的实现经验。

## 关键修正：现有 grouped MoE 已经实现的部分

- `token_dispatcher.py` 已经把 token-level expert assignment 整理成 per-expert token count / cumulative count：
  - AllGather 路径通过 `npu_moe_init_routing(... expert_tokens_num_type=1 ...)` 返回 `expert_tokens`，并设置 `group_list_type = 1`。
  - MC2 路径通过 `npu_moe_distribute_dispatch` 返回 `expert_token_nums`，并设置 `group_list_type = 0`。
  - All2AllV 路径通过 `torch.histc(topk_ids, bins=num_experts)` 统计 `tokens_per_expert`。
- 因此论文不能把“从 per-token assignment 变成 per-expert count”作为 SEW-Offload 的新贡献；这是已有 Ascend MoE 后端的执行表示。
- 但这还不是 offloading-oriented static window：
  - `topk_ids` 仍是每轮 router 产生的动态 token-level 输入；
  - `expert_tokens/group_list` 的张量形状通常固定，但数值动态，GroupedMatmul 仍按每轮 `split_value` 决定每个 expert 的 `m`；
  - 权重仍按 `layer.w13_weight/layer.w2_weight` 全量常驻或现有列表传入，没有 host expert store、固定 HBM slot、slot replacement、slot prefetch；
  - 没有固定 slot 地址来服务 ACLGraph/static kernel，也没有 capacity-tiered activation window。
- 修正后的研究贡献应聚焦为：基于已有 per-expert count 的 offload-aware fixed expert-slot residency window，即固定数量 HBM expert slots、稳定权重张量地址、动态 expert-to-slot 映射、slot miss/prefetch、可选 capacity-tier graph replay。
