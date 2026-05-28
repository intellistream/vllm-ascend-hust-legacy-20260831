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
- SEW-Offload 的核心思想是不改 router，把动态 `(token, expert, gate)` 映射到固定 expert windows 和 weight slots。
