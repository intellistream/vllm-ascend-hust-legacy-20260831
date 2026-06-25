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

## 关键修正：SEW-Offload 的优化目标不是 hit rate，而是隐藏预取时间

- 新版整体设计应明确为：固定 slot 解决 Ascend 能不能稳定执行，prefetch/orchestration 解决 offloading 慢不慢，hit-first phased execution 解决 miss 已经发生时还能不能把等待藏起来。
- 固定 slot 不是最终目标，而是 Ascend graph/static-kernel 友好的执行底座：slot tensor 地址、layout、dtype、capacity 稳定，动态变化的是 `expert_id -> slot_id` 映射。
- 预取策略不应只优化 expert cache hit rate，而应优化 exposed stall：
  - `T_stall = max(0, T_load_miss - T_overlap)`。
  - 有价值的系统指标是 host-to-HBM load time 中有多少被 routing、dispatch、resident expert compute、后续 layer compute 或下一 decode step 前的空窗隐藏。
- hit-first phased execution 的核心是：当前层 active experts 中，slot hit 的 expert 先组成 grouped MLP phase 执行，同时 miss experts 异步加载；miss 到齐后再组成少量 follow-up grouped MLP phase，避免 per-expert 小 kernel。
- 这一路线仍然不修改 router、不改变 top-k expert、不 drop token，只改变 expert 权重驻留、加载、预取和 grouped execution 的相位编排。

## 2026-05-29 当前设备与模型选择发现

- 当前机器 `npu-smi info` 显示 8 张 Ascend `910B3`，每张 HBM 约 64GB；多张卡已有大进程占用。单机单卡实验应优先选择空闲卡，并用 slot budget 人工制造 HBM 不足场景。
- 当前 CANN/驱动环境：`npu-smi 25.3.rc1`，`/usr/local/Ascend/cann-8.5.1`，操作系统 openEuler 24.03 aarch64。
- 当前默认 Python 环境能 import `vllm_ascend==0.18.0.post1.dev3327+gc6b27e12`，但不能 import `torch`、`torch_npu`、`vllm`、`transformers`；真正运行实验前需要进入正确运行环境/容器。
- 本地没有发现 `Qwen3-30B-A3B` 权重；本地有 `/data/models/Qwen3.5-122B-A10B`，大小约 234G，`config.json` 显示 `qwen3_5_moe`，48 层、hidden size 3072、256 experts、top-8、bf16；这适合作为后期真实 HBM 压力测试，但不适合作为第一个工程闭环。
- 本地还有 Qwen3-32B/8B 等 dense 权重；它们适合验证 vLLM Ascend/ACLGraph/weight prefetch 基线，但不能验证 MoE expert offloading 的核心贡献。
- Hugging Face Qwen3-30B-A3B 模型卡显示：30.5B total、3.3B activated、48 层、128 experts、top-8、原生上下文 32768；它是首选论文主模型，因为规模刚好、MoE 明确、active/total 参数差异明显。
- vLLM Ascend Qwen3-30B-A3B 文档明确以 Multi-NPU 方式部署，并说明 Atlas A2 64GB tensor-parallel-size 至少 2。这说明在 64GB 单卡上研究 expert offloading 有真实动机，而不是人为构造。

## 2026-05-29 现有 vLLM weight offloading 实测前代码发现

- `/root/miniconda3/envs/vllm-hust-dev/bin/python` 是当前可用运行环境，`vllm` 与 `vllm_ascend` 均从本地 repo editable import。
- 用户给出的 `/data/Qwen3-30B-A3B` 当前不存在；实际模型路径为 `/data/shared-models/Qwen3-30B-A3B`，大小约 57G，配置为 Qwen3 MoE：48 层、128 experts、top-8、bf16。
- vLLM-Hust 现有 weight offload 有两个 backend：
  - `uva`：把参数搬到 pinned CPU memory，并尝试创建 accelerator view；配置入口是 `cpu_offload_gb` 与 `cpu_offload_params`。
  - `prefetch`：按 layer group 选择整层模块参数，CPU store + static device buffer pool + async H2D prefetch；配置入口是 `offload_group_size`、`offload_num_in_group`、`offload_prefetch_step`、`offload_params`。
- 这些现有方法是“层级/参数级”offloading，不是 MoE expert 工作集驱动的 offloading；即使 `offload_params` 选择 `experts`，调度粒度仍随 layer 顺序，而不是根据本轮 routed expert miss/hit 来编排。
- 代码层面存在潜在 Ascend 适配风险：
  - `PrefetchOffloader` 直接使用 `torch.cuda.Stream/Event/current_stream/is_current_stream_capturing`；Ascend v2 runner 在初始化时有 `torch_cuda_wrapper()` 可把部分 CUDA API 映射到 NPU API，但需要实测确认 offloader 生命周期是否完全落在 wrapper 内。
  - `UVAOffloader` 依赖 `get_accelerator_view_from_cpu_tensor()`；当前 `current_platform.device_name` 为 `npu`，`is_cuda_alike()` 为 false，而 vLLM 的 `is_uva_available()` 只检查 pin memory，可能误判 Ascend 支持 UVA。vLLM Ascend patch 只说明 Ascend 不支持 vLLM worker 的 UvaBuffer 语义，不等价于通用参数 UVA offload 已适配。

## 2026-05-29 可利用的 Ascend/NPU 差异化控制面

- ACL Graph：vLLM Ascend 文档说明 graph 通过 capture/replay 减少 host launch overhead，且 graph replay 需要输入一致性，因此用 padding/bucketing 处理动态 shape。对 SEW-Offload 的含义是：offloaded expert 不应以动态权重对象进入执行路径，而应落入固定 slot，让图看到稳定入口。
- Stream resource constraint：vLLM Ascend ACLGraph 文档说明 piecewise graph 会受 stream 数量约束，子图和 bucket 过多可能带来额外成本。对 SEW-Offload 的含义是：不能为每个 expert/miss 创建小图，应把 active experts 合并为少量 hit/miss phases。
- Weight prefetch：vLLM Ascend 已有 weight prefetch，利用 vector 计算阶段隐藏权重预取 pipeline，并通过 CMO 操作预取到 L2；MoE 默认 prefetch gate_up。对 SEW-Offload 的含义是：现有 prefetch 是 HBM/L2 层面的 cache 预热经验，我们要把 host->HBM expert slot load 也设计成 deadline-aware pipeline。
- Ascend C/MTE/Cube/UB/L1/L0 层次：官方 Ascend C 文档说明 GM、UB、L1、L0A/L0B/L0C 等存储层级、Cube 输入/输出位置、MTE 数据搬运通路和对齐/分形 layout。对 SEW-Offload 的含义是：expert slot 不只是“显存缓存”，还应尽量保持 layout、alignment、slot 地址稳定，服务后续 grouped matmul 和可能的静态 kernel。
- 当前代码已经有 `vllm_ascend/ops/weight_prefetch.py` 与 `torch_npu.npu_prefetch` 路径；`experts_selector.py` 在 MoE top-k 选择前触发 MoE gate_up prefetch；`moe_mlp.py` 在 GMM 前做 postprocess 等待。这可以作为 SEW-Offload 的编排参考，但现有机制没有 host expert store、fixed HBM expert slots、slot miss/replacement、deadline-aware host->HBM load。

## CCF-A 新硬件系统论文的问题定义模式

- 调研对象包括 Dune/OSDI 2012、Arrakis/OSDI 2014、BPFS/SOSP 2009、TPP/ASPLOS 2023、CXL-ANNS/USENIX ATC 2023、eRPC/NSDI 2019、TPU/ISCA 2017、TVM/OSDI 2018。
- 这些论文的共同写法不是“支持一种新硬件”，而是：
  1. 旧软件抽象建立在旧硬件假设上；
  2. 新硬件/设备特性让旧假设失效或不完整；
  3. 新硬件没有自动解决问题，而是暴露了新的控制面；
  4. 现有系统要么忽略该控制面，要么用旧抽象间接使用它；
  5. 论文提出新的系统抽象，把硬件特性转化为可测量收益。
- 对 SEW-Offload 的启发：研究问题不能写成“在 Ascend 上做 MoE offloading”，而应写成“HBM 受限时，如何把动态 expert 工作集映射到 Ascend 友好的静态、稳定、可预取隐藏的 expert execution windows”。
- 新主问题定义：现有 MoE offloading 方法通常把 expert 权重视为动态 cached device objects，主要通过 cache replacement 和 prefetch prediction 优化“哪些 expert 在设备上”；这个抽象在 Ascend NPU 上不完整，因为动态 expert loading 会和 stable weight addresses、fixed execution windows、graph/static-kernel reuse、explicit data movement 发生冲突，最终把 host-to-HBM loading 暴露在关键路径上或迫使系统放弃静态执行规律。
- 论文中的研究问题应写成陈述式问题定义，而不是“how to”式方案句；更合适的写法是“旧方法如何做、依赖什么假设、这个假设在 Ascend 上为什么失效、导致什么系统后果”。

## 2026-06-01 MVP-D.9 设计对照与反思

| AOE-Serve / 计划项 | 仓库状态 |
| --- | --- |
| Tiered residency（NPU 常驻层 + CPU/slot 冷层） | `TieredResidencyPolicy` + `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS`；resident 层跳过 slot plan，仍可用全量 `w13/w2` |
| Partial release 原始 expert Parameter | `RELEASE_ORIGINAL_EXPERT_WEIGHTS=1` + `release_original_expert_weights_if_ready`；占位 `empty(0)` Parameter；**仅** non-resident 且 guard ready |
| 容量模型 | `compare_slot_budget_models`：per-layer `num_layers×num_slots×expert_bytes` vs global `num_slots×expert_bytes`；当前 runtime **实现**为 per-layer bank，global 为设计参考 |
| 13.5GB offload budget | `benchmark_config.yaml` `target_offloaded_weight_gb: 13.5`；估算工具输出 `*_within_offload_budget` 布尔 |
| 默认路径 | `release=0`、resident 空集 → 与 D.5 前行为一致（UT 92 passed） |

**设计反思（执行中记录）：**

1. **不能把“全专家都 offload”当目标**：D.9 明确分层；盲目 `num_slots=64` 会使 per-layer slot bank ~45GB（见 D.6 估算），与 HBM 节省目标矛盾。
2. **Release 必须在 host store 完整且 fixed-slot 已注册后**：否则 grouped MoE 无合法 NPU 权重；fail-closed guard 保留 `allow_retained_original_weights=True` 仅用于 planning API。
3. **Resident 层不注册 slot bank 更合理**：若 resident 仍 clone host store，会重复占 CPU；当前 resident 层在 `process_weights` 不 register（仅 non-resident 注册），减少无谓 host clone——若未来 resident 也要 trace，需单独路径。
4. **Global slot pool 未实现**：simulator/估算已对比；真正实现需改 `ExpertSlotBank` 为跨层共享与 eviction，属 D.9 之后或 MVP-E 前的大改；当前不假装 global 已上线。
5. **真实 HBM 下降仍待 NPU 验证**：release 只 drop Parameter storage；vLLM loader 可能仍保留其它引用；下一步用 `Loading model weights took X GB` + `memory_ledger` + npu-smi 交叉验证。

**待办：** 真实 trace JSONL；`release=1` NPU smoke + strict compare；可选把 SEW env 加入 vLLM allowlist 减日志噪音。

## 2026-06-05 MVP-D.11 NPU Smoke 发现

- NPU 2（910B3，60.40GB 空闲 HBM）可用，成功运行 D.11 phase split smoke。
- `ASCEND_RT_VISIBLE_DEVICES`（非 `CUDA_VISIBLE_DEVICES`）是 Ascend NPU 的正确设备隔离变量。`CUDA_VISIBLE_DEVICES` 在 NPU 环境下无效，会导致自动设备选择超时。
- Phase split 在当前无 offload 配置下每层均为 all-hit single phase，走 `execute_phased_mlp()` fast-path。每层每 decode step 产生一条 `phase_split_plan` profile event。
- 1-token & 8-token strict token-id compare 全部通过（status=ok）。
- Python 层 phase plan + JSONL 写盘开销约 16%（8-token: 4.71 vs 5.60 tok/s），与 D.10 的同步 slot path 开销（~73%）相比轻微，属于语义脚手架预期范围。

- D.11 核心设计：在 `token_dispatch_output` 之后、`_apply_mlp` 之前插入 phase split。单 phase fast-path 直接委托，多 phase 走 extract→slice→build→scatter 路径。
- AllGather 路径使用 `group_list_type=1`（count 模式），`sorted_hidden_states` 按 expert 顺序排列。Phase split 需要从 group_list 计算每个 expert 的 token slice `[start, end)`，然后对子集 expert 做 token 抽取和权重切片。
- `_slice_expert_weights()` 对 `MoEWeights` 的 w1/w2/w1_bias/w2_bias 等字段统一做 `tensor[list(expert_indices)]` 索引；None 字段保留 None。
- 判断 expert readiness：当前默认所有 expert 均为 ready（hit），仅当 offload runtime 启用且 slot_bank 中该 expert 状态不是 `SlotState.READY` 时才标记为 miss。这为 MVP-E async transfer 保留了正确的 hit/miss 语义入口。
- 窄路径 gate：`VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT=1` 只在 AllGather + unquantized + no bias + no EP 下启用，否则 fail closed 抛出 RuntimeError。不覆盖 MC2/All2All/FusedMC2/quantized/bias/EP 路径。
- 等价性验证：mock MLP 使用 `x @ w1[e] @ w2[e]`（方阵权重），确保单 phase 与多 phase（hit/miss 切分）输出逐元素一致。31 个 UT 全部通过。
- 与 D.10 的关系：D.10 的 `_maybe_apply_moe_offload_plan` 在 dispatch 前做 slot plan remap；D.11 的 `_maybe_plan_phase_split` 在 dispatch 后做 phase split。两者互不冲突——D.10 改变了 `fused_experts_input` 的 weights/routing，D.11 在此基础上切分 MLP compute。
- 当前 D.11 使用同步 slot_readiness 判定；MVP-E 将改为异步判断 + stream overlap。

- 最新提交：`ac6e3922 feat(moe-offload): MVP D.9 implement tiered residency and partial release for expert weights`，当前 `research` 与 `origin/research` 对齐。
- MVP-D.9 的实际范围：
  - 新增 `TieredResidencyPolicy` 和 `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS`，使指定 MoE layer 保留完整 NPU expert 权重，并跳过 fixed-slot path。
  - 新增 `VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS`，仅 opt-in 时对 non-resident layer 在 host store/slot bank readiness guard 通过后，把原始 `w13_weight/w2_weight` 替换成 zero-element placeholder Parameter。
  - `memory_ledger()` 会把已 release layer 的 original expert bytes 排除；新增容量估算对比 per-layer slot bank 与 global slot pool。
  - `fused_moe.py` 只在 `should_use_fixed_slot_plan_for_layer(layer_id)` 为真时 register/prepare slot；resident layer 继续走原始权重路径。
- 当前进度判断：
  - Python/UT 层面已完成 tiered residency、partial release guard、ledger、容量模型。
  - 默认 no-offload 路径保持干净；D.9 release 默认关闭。
  - 真实 NPU 上 release=1 尚未完成 smoke/HBM 验证，不能声称 partial release 已真实降低 vLLM reported resident weight。
- 暴露的问题：
  - 全模型 fixed-slot register 仍可能在 post-load 阶段很慢，疑似 48 层 host clone + slot bank 初始化成本过高。
  - 当前 runtime 仍是 per-layer slot bank，`num_slots` 越大 HBM 成本按 layer 数线性放大；global slot pool 还只是估算/设计参考。
  - partial release 只替换 Parameter storage，仍需排查 vLLM loader/offloader 是否存在其它引用导致 HBM 不释放。
  - 当前 fixed-slot capacity guard 限制的是“单层单次 MoE 调用 active expert 并集 <= num_slots”，不是每个 expert 的 token 容量。
- 对用户动态 count 分析的确认：
  - 当前常规 AllGather MoE 调用 `DeviceOperator.npu_moe_init_routing(... expert_tokens_num_type=1, expert_tokens_num_flag=True ...)`，并把 `group_list_type=1` 传给 grouped matmul，属于 dropless dynamic count。
  - `csrc/torch_binding.cpp` 默认 `expert_capacity=-1, drop_pad_mode=0`；`moe_init_routing_custom_torch_adpt.h` 中 `drop_pad_mode==0` 会分配 `[num_out_tokens, h]`，不是 `[expert_num, expert_capacity, h]`。
  - 因此 SEW-Offload 不应把固定 token capacity/drop-pad 作为主路。固定 token capacity 只能作为可选实验分支，用于研究 graph/static-kernel bucket 收益，不能改变 router/top-k 语义，也不能 drop token。
- 设计修正：
  - 保留“固定 expert weight slot / 稳定权重地址”作为主线。
  - 放弃把“固定每专家 token 容量”作为默认路线；默认继续复用动态 count/group_list，slot remap 只改变 expert id 到 slot id 的映射，不改变每个 expert 本轮实际 token 数。
  - 后续若研究 activation capacity，应采用 bucket/pad-only/无 drop 的可选路径，并用 trace 证明 padding 浪费与 graph 收益的权衡。

## 2026-06-02 D.9 执行结果：分段打点与小范围 release=1 smoke

- 已新增 runtime profiling：
  - `MoeOffloadRuntime.profiling_summary()` 返回内存中事件、按事件名累计耗时和当前 `memory_ledger`。
  - `VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` 可选启用跨进程 JSONL profile；每次 `register_layer_for_fixed_slots` / `release_original_expert_weights` 都追加事件和 ledger 快照。
  - `run_fixed_slot_smoke.py` 会设置 artifact 下的 `moe_offload_profile.jsonl`，并在 `summary.json` 写入 `moe_offload_profile_jsonl_events`。
- 关键坑：vLLM V1 EngineCore 在子进程中运行，父进程 `get_moe_offload_runtime().profiling_summary()` 只能看到父进程空 runtime。因此真实 NPU artifact 必须看 `moe_offload_profile_jsonl_events`，不能只看父进程 `moe_offload_profile`。
- 小范围 release=1 smoke：
  - baseline：`artifacts/sew_offload/runs/d9_no_offload_profile_20260602/summary.json`，no-offload 成功，reported `Loading model weights took 56.9001 GB`，1 output token。
  - candidate：`artifacts/sew_offload/runs/d9_release1_small_profile_20260602/summary.json`，NPU 6，`resident_layer_ids=1..47`，仅 layer 0 non-resident + release=1，成功，reported `Loading model weights took 42.3454 GB`，1 output token。
  - strict token-id compare：`artifacts/sew_offload/runs/d9_release1_small_profile_20260602/correctness_compare.json`，`status=ok`、`matched=1`。
- candidate profile JSONL：
  - `register_layer_for_fixed_slots` layer 0 用时约 `2.034s`。
  - release 前 ledger：`host_experts=128`、`host_store_bytes=1207959552`、`original_expert_weight_bytes=1207959552`、`slot_bank_bytes=75497472`。
  - `release_original_expert_weights` layer 0 用时约 `0.0004s`。
  - release 后 ledger：`original_expert_weight_bytes=0`、`host_store_bytes=1207959552`、`slot_bank_bytes=75497472`、`total_managed_bytes=1283457024`。
- 设计反思：
  - 小范围 release=1 证明 partial release 机制、slot-backed execution 和 token-id correctness 在真实 NPU 上可以闭环。
  - reported weight 降低主要仍来自组合使用的 native `PrefetchOffloader`，不是 layer 0 release 单独贡献；不能把该数值解释为 SEW full expert offload 性能收益。
  - 单层 register 约 2s，推测全 48 层 register 会非常重；这解释了此前全模型 fixed-slot post-load 卡住/过慢现象。
  - 下一步应先采真实 trace，再做 resident layer / num_slots / global pool sweep；不应直接扩大到 48 层 release。

## 2026-05-29 现有 offloading baseline 实测结论

- artifact：`/root/vllm-ascend-hust/artifacts/sew_offload/existing_offload_20260529T143705Z`
- 为了让当前本地 `vllm-hust/main` 与 `vllm-ascend-hust/research` 协同运行，做了最小 API 兼容补丁：
  - `vllm_ascend/ops/fused_moe/fused_moe.py` 兼容当前 `MoERunner`、缺失的 `SharedFusedMoE` 模块、`routed_input_transform` 字段位置和缺失的 `reduce_results`。
  - `vllm_ascend/worker/model_runner_v1.py` 补齐 `torch.cuda.is_current_stream_capturing -> torch.npu.is_current_stream_capturing` 映射，用于验证现有 prefetch 后端能否越过 CUDA API 阻塞。
- no-offload 基线成功：
  - case：`baseline_no_offload_after_reduce_results_patch`
  - `LOAD_OK seconds=49.062`
  - `GENERATE_OK seconds=17.158`
  - checkpoint size `56.87 GiB`，safetensors 加载 `17.25 seconds`
  - vLLM 日志：`Loading model weights took 56.9001 GB`
  - `npu_monitor.txt` 解析峰值：NPU 4 HBM `63886 MB`
  - 结论：Qwen3-30B-A3B 在单张 64GB 910B3 上可以极限跑通，但在仅 512 token、0.5GB KV cache 条件下已经接近 HBM 上限。
- UVA expert offload 失败：
  - case：`uva_experts_cpu8gb`
  - 失败点：`ValueError: get_accelerator_view_from_cpu_tensor is currently not supported in: npu`
  - 结论：当前 vLLM UVA offload 抽象依赖 CPU pinned memory 的 accelerator view，Ascend NPU 不支持，不能作为 SEW-Offload 基线底座。
- prefetch expert offload 原始路径失败：
  - case：`prefetch_experts_group4_num1_step1`
  - 模型权重驻留降为 `43.4001 GB`，HBM 峰值约 `50697 MB`
  - 失败点：`torch.cuda.is_current_stream_capturing()` 在当前 `torch 2.9.0+cpu + torch_npu` 下触发 dummy CUDA base class 错误。
  - 结论：现有 prefetch 后端确实能释放约 13.5GB 模型权重 HBM，但实现仍带 CUDA graph/stream 假设。
- prefetch expert offload 在补齐 CUDA-to-NPU wrapper 后继续失败：
  - case：`prefetch_experts_group4_num1_step1_after_cuda_wrapper_patch`
  - 模型权重驻留仍为 `43.4001 GB`，HBM 峰值约 `50696 MB`
  - 新失败点：`RuntimeError: Expected all tensors to be on the same device, but got weight is on cpu ... wrapper__npu_grouped_matmul`
  - 结论：更深层问题不是简单 CUDA API wrapper，而是现有 layer/parameter prefetch 无法保证 Ascend MoE grouped matmul 看到 post-processed、layout-compatible、NPU-resident expert weights。
- 研究含义：
  - 现有 offload 证明了 HBM 压力和 offload 收益真实存在。
  - 但现有 UVA/prefetch 不是 MoE expert-working-set-aware，也不是 Ascend slot/layout-aware。
  - SEW-Offload 应在 `AscendUnquantizedFusedMoEMethod.apply()` 到 `moe_comm_method.fused_experts()` 的边界做 fixed expert slots，而不是仅包装 decoder layer forward。
  - 下一步应先做 trace-and-measure 与同步 slot miss load，真实测量 host-to-HBM expert copy 时间和 exposed stall。

## 2026-05-29 最小 Benchmark 协议

- 根据用户反馈，benchmark 规范已收缩为当前阶段需要的最小可复用定义，不再提前规划完整 benchmark framework。
- 当前 benchmark 只固定五件事：模型、数据集、workload buckets、13.5GB offload budget、指标。
- 固定主模型为 `/data/shared-models/Qwen3-30B-A3B`，单机单卡 Ascend 910B3，TP=1，bf16，不修改 router/top-k/expert activation 语义，不 drop token/expert。
- 固定正式数据集为 `lmsys/lmsys-chat-1m`，seed `20260529`，1000 条请求；工程调试可用 synthetic smoke set，但不能作为正式 benchmark 结果。
- 固定 workload buckets：short_chat 200、medium_chat 300、long_prefill 200、decode_heavy 200、burst_mixed 100；主并发为 1/4/8。
- 固定 offload budget：target offloaded weight `13.5GB`，target resident weight `43.4GB ± 0.5GB`，依据现有实测 no-offload `56.9001GB` 与 native prefetch `43.4001GB`。
- 核心指标为 `exposed_stall_per_output_token_ms`，同时报告 TTFT、TPOT、ITL、latency p50/p90/p99、output tok/s、success rate、resident weight、peak HBM、host-to-HBM bytes/copy time、prefetch wait time。
- 持久规则已简化写入 `AGENTS.md` 和 `.planning/sew_offload/task_plan.md`：后续可比较实验先检查 `docs/sew-offload/benchmark_config.yaml` 是否匹配，但暂不引入 artifact layout、validity gates、方法分层等复杂要求。

## 2026-05-29 Native Offloading Minimal Benchmark Pilot

- 新增最小 runner：`tools/sew_offload/run_minimal_offload_benchmark.py`，读取 `docs/sew-offload/benchmark_config.yaml`，只输出 throughput、TTFT、TPOT。
- 当前机器没有本地 `lmsys/lmsys-chat-1m`，运行环境也没有 `datasets` 包；本次先用同 bucket schema 的 `synthetic_smoke` manifest 验证原生 offloading 能否进入生成。
- smoke manifest：`artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/requests.jsonl`，本次运行 1 条 `short_chat`，128 prompt tokens，128 output tokens。
- native expert prefetch：`offload_backend=prefetch`、group4/num1/step1、`offload_params=experts`，resident weight `43.4001 GB`，符合约 13.5GB offload budget，但 profile forward 在 `torch_npu.npu_grouped_matmul` 前失败：expert weight 仍在 CPU，hidden states 在 NPU。
- native all-param layer prefetch：group4/num1/step1、offload all layer params，resident weight `42.9722 GB`，也在生成前失败；普通 `wrapper_NPU__matmul` 收到 CPU tensor，并伴随 Ascend vector core / MTE DDR address out-of-range 异常。
- no-offload sanity：resident weight `56.9001 GB`，1 条 short_chat 成功，128 output tokens，output throughput `7.6207 tok/s`，TTFT `465.78 ms`，TPOT `128.59 ms`。这只证明 runner 能产出三指标，不是 offloading 结果。
- 关键结论：当前 vLLM 原生 offloading 对 Qwen3-30B-A3B 单卡 Ascend MoE 还不能产生可报告性能；真实优化起点是 fixed NPU expert slots、layout-stable post-processed weight buffers、MoE execution boundary 集成、native torch.npu transfer/synchronization，而不是先调 prefetch policy。

## 2026-05-31 再核实：当前确实不支持 Ascend MoE expert offloading 服务

- 代码层面确认 `vllm-hust` 的 weight offloading 只有 `auto`、`uva`、`prefetch` 三类，配置入口是 `cpu_offload_gb/cpu_offload_params` 与 `offload_group_size/offload_num_in_group/offload_prefetch_step/offload_params`；它们是通用参数/层级 offload，不读取当前 MoE 层 `topk_ids`、per-expert token counts、slot hit/miss 或 expert deadline。

## 2026-06-15 Expert 搬运分解与流水线发现

- 详细文档已落在 `docs/sew-offload/11-expert-transfer-breakdown-and-pipeline.md`。
- 当前 miss expert load 的真实行为已经代码复核确认：
  - `MoECommMethod.fused_experts()` 先执行 `_maybe_apply_moe_offload_plan()`，然后才 token dispatch，再 MLP compute。
  - `MoeOffloadRuntime.prepare_fixed_slot_plan()` 对 `unique_active_experts` 逐个处理 miss expert。
  - `TransferEngine.load_sync()` 对每个 miss expert 做两次同步 copy：`w13.copy_` + `w2.copy_`。
- 因此当前路径是：
  - **先搬后算**
  - **逐 expert 同步搬**
  - **没有 transfer/compute overlap**
  - **不是 batch 搬运**
- 当前真实路径更接近 **no-pin**，不是 pinned/UVA：
  - `HostExpertStore.register_layer()` 使用 `detach().cpu().clone()`。
  - `patch_v2/patch_uva.py` 明确说明 Ascend NPU 不支持真正 UVA；当前 wrapper 是 CPU tensor + NPU mirror。
  - 本轮 `--pin-memory` 只能当 PyTorch CPU allocator flag 的控制实验，不能解读为 Ascend 官方 pinned DMA path。
- no-pin 当前真实路径的两层分解：
  - **size sweep 拟合**：单 expert 约 `1.034 ms`，其中 size-dependent payload 项约 `0.571 ms`，fixed+residual 约 `0.464 ms`。
  - **CANN timeline**：`two_tensor_current` 的 `record_function` 窗口约 `0.898 ms/expert`，其中 `aclrtMemcpy` span 约 `0.778 ms`，`aclrtSynchronizeStream` 约 `0.030 ms`，host other 约 `0.089 ms`。
- 上述两组分解不矛盾：
  - fit 分解切的是 `size-dependent payload` vs `fixed/residual`
  - profiler 分解切的是 `record window` 内部的 `aclrtMemcpy/sync/host-other`
  - 这说明 `aclrtMemcpy` runtime span 内部本身包含了相当多 size-insensitive 成分，不等于纯 PCIe wire payload time。
- 三种 copy 模式对照（no-pin）：
  - `single_contiguous_expert`：`0.770 ms/expert` window，`0.699 ms` memcpy。
  - `two_tensor_current`：`0.898 ms/expert` window，`0.778 ms` memcpy。
  - `batched_contiguous_experts`（8 experts/batch）：折算 `0.478 ms/expert` window，`0.452 ms/expert` memcpy。
- 直接含义：
  - 把一个 expert 拆成两次 copy 确实有固定开销。
  - 多个 miss expert 打成 batch 后，链路利用率和每 expert 时间都显著改善。
  - 后续优化重点不应只盯 PCIe 峰值，而应同时减少 copy 次数、做 batch、并把 miss 搬运藏到 hit compute 下面。
- 已冻结的 MVP-E 优先级：
  1. `load_sync -> load_async`
  2. D.11 phase split 真正变成 hit-first overlap
  3. miss experts 按 tensor 类型或 packed layout 做 batched copy
  4. 新增 overlap 指标：`miss_transfer_ms`、`hit_phase_compute_ms`、`exposed_stall_ms`
- 官方文档交叉验证：vLLM `OffloadConfig` 文档同样描述 `auto/uva/prefetch` 三类 weight offload；vLLM Ascend KV Cache CPU Offload 文档是针对 KV cache block 的 `NPUOffloadingSpec`；vLLM Ascend Weight Prefetch 文档是将已在设备侧的权重预取到 L2/cache，不是 host-to-HBM expert weight offload。
- `PrefetchOffloader` 通过 module forward hook 插入 `wait_prefetch/start_prefetch`，按静态层组和参数名选择 offload 对象；内部使用 `torch.cuda.Stream/Event/current_stream/is_current_stream_capturing/stream`。即使被 Ascend wrapper 部分兼容，它仍不是 Ascend-native MoE expert working-set runtime。
- `UVAOffloader` 依赖 `get_accelerator_view_from_cpu_tensor(cpu_data)`；本地 Qwen3-30B-A3B 实测明确失败：`ValueError: get_accelerator_view_from_cpu_tensor is currently not supported in: npu`。
- `vllm-ascend-hust` 具备 MoE routing、per-expert token grouping、`npu_grouped_matmul`、EP/EPLB、ACLGraph/NPUGraph、HBM/cache weight prefetch、KV offload 等能力，但没有 host expert store、固定 HBM expert slots、expert miss/replacement、deadline-aware host->HBM expert load、hit-first phased grouped MoE execution。
- Ascend weight prefetch 文档和 `vllm_ascend/ops/weight_prefetch.py` 说明现有 prefetch 是把已在设备侧/HBM 的权重通过 CMO/npu_prefetch 预热到 cache，利用 MoE gating/RMSNorm/SwiGLU 等 vector 窗口隐藏访问，不是 CPU/host 到 HBM 的 expert offloading。
- 当前实测证据仍成立：expert-only native prefetch 可把 resident weight 从 `56.9001 GB` 降到 `43.4001 GB`，但 profile forward 在 `wrapper__npu_grouped_matmul` 前失败，原因是 expert weight 仍在 CPU 而 hidden states 在 `npu:0`。all-param prefetch 降到 `42.9722 GB`，但 dense matmul 也出现 CPU/NPU tensor mixing 和 MTE DDR out-of-range 异常。
- 结论：问题真实存在。不是简单参数没开，而是缺少 Ascend-specific MoE expert offloading 抽象。下一步应实现 `trace -> simulator -> fixed NPU expert slots + sync miss load -> async prefetch -> hit-first phases`，而不是继续调 `offload_group_size` 或补零散 CUDA wrapper。

## 2026-05-31 总体架构决策

- 新设计文档落盘：`docs/sew-offload/08-ascend-moe-offload-architecture.md`。
- 系统边界继续选择 `AscendUnquantizedFusedMoEMethod.apply()` 中 `select_experts(...) -> build_fused_experts_input(...) -> moe_comm_method.fused_experts(...)` 之间；此处能看到 `topk_ids/topk_weights/layer_id/expert_map/weights/backend metadata`，是最低侵入的 expert execution boundary。
- 控制面模块：`TraceCollector`、`Deadline-Aware PrefetchPlanner`、`CostModel`、`Replacement/Residency Policy`、`Hit-First PhaseScheduler`。
- 数据面模块：`HostExpertStore`、`ExpertSlotBank`、`TransferEngine(torch.npu streams/events)`、`Layout/Postprocess Validator`、可选 `npu_prefetch` cache warmup。
- 关键不变量：不改 router/top-k/gate weights；每个 active expert 计算前必须映射到 ready slot；slot 地址稳定；slot layout/dtype/stride/shape 与 Ascend grouped MoE backend 契约一致；phase split 输出必须等价于 single phase。

## 2026-05-31 MVP-A 实现发现

- MVP-A 选择最小可验证 trace-only 范围：在 `select_experts(...)` 后记录 `topk_ids` 派生出的 active experts 和 per-expert token counts，并把原始 `topk_ids/topk_weights` 对象原样返回。
- `TraceCollector.record()` 会 detach `topk_ids`，必要时搬到 CPU 后统计；该同步只在 `VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1` 且 `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1` 时发生，默认关闭路径不统计。
- 全局 runtime 使用懒初始化，读取集中定义在 `vllm_ascend/envs.py` 的 `VLLM_ASCEND_MOE_OFFLOAD_*` 变量；`trace_max_records` 用 bounded deque 控制内存。
- MVP-A 不含 host expert store、slot bank、transfer engine、replacement、phase scheduler 或任何权重/执行路径变更，因此符合“无偏移”要求。
- 现有 `tests/ut/test_envs.py` 的类型推断漏掉 `float(...)` env handlers，导致已有 `VLLM_ASCEND_UTILITY_*` float 环境变量测试失败；已将测试输入生成逻辑扩展到 float 类型。

## 2026-05-31 下一步规划判断

- 下一步最优顺序是先做 trace export 和 offline simulator，而不是直接做 fixed slot 或 async prefetch。
- 原因：fixed slot 是第一个改变执行路径的阶段，必须先有可复现 trace 和 slot/policy 证据，避免在真实 NPU copy 与 grouped MoE backend 上盲调。
- MVP-B 只改变观测与 artifact，不改变推理；MVP-C 只做离线策略评估；MVP-D 才进入同步 fixed slot correctness。
- fixed slot correctness 的 hard gate 是：`npu_grouped_matmul` 前不能出现 CPU expert weight，slot tensor 地址稳定，输出与 no-offload 容差内一致。
- async transfer 和 hit-first phases 必须等 fixed slot correctness 成立后再做，否则会把 correctness bug 和 overlap scheduling bug 混在一起。

## 2026-05-31 MVP-B 实现发现

- JSONL trace schema 复用 `TraceRecord.to_jsonable()`，因此字段稳定为 `layer_id/step_id/mode/num_tokens/top_k/num_experts/active_experts/expert_token_counts`。
- `TraceCollector.write_jsonl(path)` 返回记录数；空 collector 会写出空文件，这使默认关闭路径也能明确产出 0 条记录。
- `collect_moe_trace.py` 在运行 vLLM 前显式设置 `VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1` 和 `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1`，并 `reset_moe_offload_runtime()`，避免进程内旧 runtime 缓存错误配置。
- CLI 的 synthetic smoke manifest 生成不依赖 tokenizer，避免 prepare-only 阶段加载大模型；真实 trace collection 才实例化 `vllm.LLM`。
- 当前 prepare-only smoke 已验证 manifest 生成链路；真实 Qwen3-30B-A3B trace collection 仍需 NPU 空闲窗口。

## 2026-05-31 MVP-C 实现发现

- Offline simulator 可以完全独立于 `torch_npu` 与真实 NPU 执行路径运行，适合先做 slot budget/policy sweep。
- 当前 simulator 的 hit/miss 粒度是 `(layer_id, expert_id)` whole-expert residency；这与 MVP-D 的 whole-expert fixed slot 一致。
- `phase_opportunity_count` 的首版定义是同一 trace record 内同时出现 resident hit 和 miss；它只表示 hit-first phase 有机会，不表示一定值得 split。
- 默认 expert size 估算为 `14,680,064` bytes，来自 Qwen3-30B-A3B 单个 expert 的 bf16 whole-expert 近似：`w13 + w2`。
- `sticky_layer_lru` 当前是保守启发式：incoming 同层 expert miss 时，优先驱逐其他层 resident expert；这用于捕捉 decode temporal locality，但后续需要真实 trace sweep 验证。

## 2026-05-31 MVP-D 设计反思

- Fixed slot 接入不能只把 `w13/w2` 从 `num_experts` 维替换成 `num_slots` 维，并保持原始 `topk_ids` 不变；这样 grouped MoE backend 会用 expert id 索引 slot tensor，语义会错。
- 当前代码中 `MoECommMethod.fused_experts()` 已支持 `routing.log2phy`，会在 token dispatch 前把 `topk_ids` remap；这可能是 expert-to-slot remap 的正确接入口。
- 因此 MVP-D 分成两层：
  1. 安全底座：HostExpertStore、ExpertSlotBank、LayoutValidator、TransferEngine，同步 load 可单测。
  2. 执行接入：slot-backed weights + log2phy/topk remap + backend-ready layout 验证。该层尚未接入主路径。
- Layout 校验必须区分 copy-compatible 与 backend-ready：CPU host bundle 到 NPU slot copy 时设备类型可以不同；进入 grouped MoE backend 前才必须验证 slot tensors 在 NPU 上。
- Runtime 的 fixed-slot path 当前采用 fail-closed guard：只有 enabled、non-trace、num_slots>0 时认为应进入 fixed slot，但在 log2phy remap 完成前 `prepare_weights_for_execution()` 明确抛 `NotImplementedError`，避免静默错误。

## 2026-05-31 MVP-D.2 Remap 设计反思

- `log2phy` remap 是必要条件，但不是充分条件。当前 `MoECommMethod.fused_experts()` 会把 `topk_ids` remap 到 physical id，但 `TokenDispatcherWithAllGather` 默认仍使用 `self.num_experts_local` 作为 `expert_num`，All2All/MC2 也有各自基于原 MoE config 的 expert count 逻辑。
- 因此 slot-backed weights 的安全执行契约至少包含三件事：
  1. `w13/w2` 第 0 维是 physical slot id，地址稳定。
  2. `topk_ids` 通过 `logical_to_physical` remap 到 slot id。
  3. token dispatch 生成的 `expert_tokens/group_list` 使用 physical slot count，而不是 logical expert count。
- 已新增 `physical_expert_count` 路由字段，并只在 AllGather 无 `expert_map`、无 redundant experts 的窄路径启用。这样默认路径完全不变，slot path 也不会误入 EP/EPLB 复杂语义。
- `ExpertSlotBank` 从“每 slot 一个独立 tensor”升级为 `[num_slots, ...]` backing tensors；单 slot object 只持有 view。这是为了满足 Ascend grouped MoE 对稳定连续权重张量的输入需求。
- `HostExpertStore` 显式 `.cpu().clone()`，因此它是 host expert store 的真实方向；但当前没有从原 layer 参数中释放 full expert 权重，所以只能声明为 correctness prototype，不能声称已经达到 13.5GB offload budget。
- 下一步接主路径前必须再加 hard gate：
  - 仅支持 unquantized、单卡、AllGather、无 `expert_map`、无 dynamic EPLB、无 redundant experts。
  - bias/scale 暂不支持或必须随 slot 同步搬运。
  - backend-ready 校验必须确认 `PreparedSlotWeights.w1/w2` 在 NPU 上，且 `physical_expert_count == w1.shape[0] == w2.shape[0]`。

## 2026-05-31 MVP-D.3 主路径窄接入发现

- 已把 `PreparedSlotWeights` 接入 `AscendUnquantizedFusedMoEMethod.apply()`，但只在 fixed-slot enabled 且 `MoECommType.ALLGATHER` 的窄路径启用。
- 主路径 hard gates 是正确性设计的一部分，不是临时偷懒：
  - MC2/FUSED_MC2/All2All 的 dispatch/combine metadata、expert_map、global expert num 语义更复杂，不能假设 `physical_expert_count` 足够。
  - bias 目前没有进入 `HostExpertStore`/slot bank，因此开启 bias 时必须 fail closed。
  - force load balance 会在 trace/active expert 提取后改写 `topk_ids`，当前顺序下会让 prepared slots 与实际 topk 不一致，因此必须拒绝。
  - zero-expert path 会在 grouped MoE 前改写 expert index/scale 并合并额外输出，当前 fixed-slot path 不能只按原 active experts 准备 slot，必须拒绝。
- 当前 `apply()` 内 lazy registration 主要服务生命周期安全；真实路径的主要注册点是 `process_weights_after_loading()` 后的 post-processed weights。后续释放原始 full expert 权重时，必须确认 slot bank 已经基于 post-processed layout 初始化。
- 当前 fixed-slot path 依然会同步读取 `topk_ids` 到 CPU 计算 active expert working set；这与“先正确、后 overlap”一致，但不是最终高性能路径。

## 2026-06-01 PrefetchOffloader 与分层驻留设计澄清

- `PrefetchOffloader` 位于 `vllm-hust/vllm/model_executor/offloader/prefetch.py`，是 vLLM-Hust/vLLM 侧通用参数 offloader，不是 `vllm_ascend` 的 MoE 专用模块。
- 它的核心功能是：按 decoder layer 的静态顺序和 `offload_group_size/offload_num_in_group/offload_prefetch_step/offload_params` 选择模块参数，把参数保存在 CPU storage 中，再用静态 device buffer pool 和 forward hook 在层执行前后触发 `wait_prefetch/start_prefetch`。
- `PrefetchOffloader` 可复用的思想包括 CPU 参数副本、静态 device buffer pool、异步 copy stream/event、post-init 后同步 processed weights、编译/graph cache hash 参与配置。
- 但它不能直接作为 SEW-Offload 的核心接口，原因是抽象层级不同：它以“模块/层”为调度单位，不读取当前 MoE routing 的 `topk_ids`、active expert working set、per-expert token count、slot hit/miss 或 expert deadline。
- 它还假设参数名不变、模块 forward 顺序静态、参数对象可被替换成静态 buffer；而 Ascend grouped MoE 需要在 `topk_ids -> group_list -> npu_grouped_matmul` 之间保持 logical expert 到 physical slot 的一致映射，并保证 weight layout、device、slot count 与 backend 契约匹配。
- 本地实测也说明简单复用不够：native prefetch expert offload 能降低 resident weight，但在 Ascend MoE `npu_grouped_matmul` 前出现 CPU/NPU weight mixing；UVA 后端在 NPU 上不支持 accelerator view。
- 设计修正：不能把 SEW-Offload 描述成“所有专家参数都放 CPU，NPU 只保留当前 active experts”。更合理的目标是 tiered residency，即 NPU 显存中保留一部分完整层/热点专家，再用固定 expert slots 作为 CPU expert 的补充 cache。
- 分层驻留应至少包含两类 NPU expert residency：
  1. pinned/full-resident experts：常驻 NPU，不参与 slot eviction，适合显存足够时保留若干完整 MoE 层或长期热点 expert。
  2. cache/slot experts：来自 CPU host store，进入固定 slot bank，按 policy 替换和预取。
- 因此后续参数所有权转移方案应从“释放全部原始 expert 参数”升级为“按 policy 释放部分 expert/layer 的原始参数”：保留的专家仍由原始 NPU 参数拥有，卸载的专家由 HostExpertStore + ExpertSlotBank 拥有。
- fixed-slot MVP-D 的 `num_slots` 只是 CPU-backed cache 容量，不应被理解为 NPU 中可驻留 expert 的全部容量；真实系统还需要 `resident_layers/resident_experts` 这类预算/策略。

## 2026-06-01 MVP-D.4 兼容性反查与 backend-ready 门禁

- 反向检查默认路径后补充回归护栏：`VLLM_ASCEND_MOE_OFFLOAD_ENABLED` 未开启时，`AscendUnquantizedFusedMoEMethod.apply()` 仍把原始 `layer.w13_weight/layer.w2_weight` 传给 backend，`routing.log2phy` 与 `routing.physical_expert_count` 均保持 `None`。
- `build_fused_experts_input(...)` 的默认 routing 仍是 logical expert space；新增 `physical_expert_count` 不应改变未显式传参的既有调用。
- `TokenDispatcherWithAllGather` 默认仍使用 `self.num_experts_local` 生成 `expert_num/active_expert_range`；只有 fixed-slot 显式传入 `physical_expert_count` 时才切到 slot space。
- `PreparedSlotWeights.validate_backend_ready(expected_device_type=...)` 已成为 fixed-slot backend 前置门禁：检查 `physical_expert_count > 0`、`w1/w2` 第 0 维等于 physical expert count，并复用 `LayoutValidator.validate_backend_ready` 检查设备类型。
- `apply()` 中调用该门禁时使用 `x.device.type` 作为期望设备；这样 CPU mock UT 仍可运行，真实 NPU 推理时 CPU slot tensor 会在 Python 边界被清晰拒绝，而不是让 `npu_grouped_matmul` 抛混乱的 CPU/NPU 混用错误。
- 新增 `tools/sew_offload/run_fixed_slot_smoke.py`，只用于 MVP-D correctness smoke：通过 env 启用 fixed-slot sync path，生成 `summary.json` artifact；它不是正式 benchmark runner，也不能用于论文性能数据。
- 自我反思：当前仍未释放原始 full expert 参数，不能声称已经实现 HBM saving；当前的系统价值是验证 slot remap、slot residency 和 backend-ready 契约是否成立。真实节省 HBM 需要后续在 post-load 后释放/替换原 expert 参数，并确认 vLLM weight loader 生命周期安全。
- 自我反思：真实 NPU smoke 尚未执行；当前只能说 mock backend 和 Python 边界门禁通过，不能声称 `npu_grouped_matmul` 已接受 slot weights。

## 2026-06-01 MVP-D.4 profile dummy routing 反查

- 真实 NPU prefetch+fixed-slot smoke 曾越过模型加载，显示 `Loading model weights took 46.7751 GB`，但在 vLLM profile/dummy run 中失败于 `enable_force_load_balance` 与 fixed-slot 小窗口的冲突。
- 原因不是真实请求语义，而是 profile run 会把 dummy `topk_ids` 改写成均衡覆盖 logical experts 的随机集合；当 `num_slots` 很小（例如 8）时，dummy active expert working set 会超过 fixed-slot budget。
- 新设计：仅在 `enable_force_load_balance and fixed_slots` 的 profile 路径中，把 dummy routing 限制为 `[0, top_k)` 且要求 `num_slots >= top_k`；真实请求不走该分支，仍保留 router/top-k 语义和 active expert working set。
- 单测失败根因：CPU mock backend 使用 CPU hidden states，但 lazy fixed-slot registration 为 CPU/offloaded weight 选择当前 NPU slot，这是实机正确策略；因此单测应显式注册 CPU slot bank，而不是放宽实机 backend-ready 设备门禁。
- 反向兼容结论：默认路径未开启 offload 时不会进入 profile dummy slot 路径；AllGather dispatcher 仍默认使用 logical expert count，只有 fixed-slot 显式传入 `physical_expert_count` 才切到 slot count。

## 2026-06-01 PrefetchOffloader 生命周期缺口

- 修正 profile dummy routing 后，真实 NPU prefetch+fixed-slot smoke 继续推进，但在 profile run 的现有 vLLM `PrefetchOffloader` forward hook 中失败：`AssertionError: Buffer pool not assigned`。
- 栈显示失败发生在 `/root/vllm-hust/vllm/model_executor/offloader/prefetch.py::start_onload_to_static()`，说明 module forward hook 已安装并执行，但 `_ModuleOffloader.assign_buffer_slot(...)` 未被调用。
- 对比上游 `GPUModelRunner.load_model()`：加载模型、可选 cudagraph wrapper 后调用 `get_offloader().post_init()`；`PrefetchOffloader.post_init()` 正是在这里同步 processed CPU storage、分配 `StaticBufferPool`、给 module offloaders 指向 GPU/NPU static buffers，并启动初始 prefetch。
- Ascend `NPUModelRunner.load_model()` 重写了加载流程，但此前漏掉了这一步。这不是 SEW fixed-slot 自身错误，而是现有 prefetch offload 在 Ascend 重写加载路径上的生命周期缺口；SEW smoke 暴露了它。
- 最小修正：在 Ascend `load_model()` 的可选 ACLGraph wrapper 之后补齐 `get_offloader().post_init()`，与上游顺序一致。`NoopOffloader.post_init()` 为空，因此默认不开 weight offload 时不改变执行逻辑。

## 2026-06-01 CUDA-to-NPU wrapper Event 兼容缺口

- 补齐 `PrefetchOffloader.post_init()` 后，真实 smoke 进一步推进，日志显示 `[PrefetchOffloader] Initialized 12 modules ... Static buffer pool: 1.2080 GB`，说明 buffer pool 生命周期问题已解。
- 新失败发生在 `PrefetchOffloader.post_init()` 的初始 prefetch：`torch.cuda.current_stream().record_event(fork_event)` 调到 `torch_npu.npu.Stream.record_event`，后者执行 `event.record(self)`，但 `_torch_cuda_wrapper()` 退出后把 `torch.cuda.Event` 改成了 `_EventPlaceholder`，其 `record` lambda 不接受 stream 参数。
- 这属于 CUDA-to-NPU wrapper 的 API family 不一致：`torch.cuda.current_stream/default_stream/stream` 在 finally 中保持为 NPU 实现，但 `torch.cuda.Event` 却变回 placeholder，导致 NPU stream 与 placeholder event 混用。
- 最小修正：wrapper 退出后也保持 `torch.cuda.Event = torch.npu.Event`，与其它 stream API 一致。targeted UT 已证明修正前失败、修正后通过。

## 2026-06-01 internal-router gate 与 fixed-slot smoke 结论

- `ExpertKey(layer_id=0, expert_id=266)` 的根因不是 slot mapping 自身，而是 Ascend MoE runner 漏掉上游 `MoERunner._forward_impl()` 中的 internal-router gate 生命周期：Qwen3Moe internal-router 场景传入 `router_logits=hidden_states`，如果不先执行 gate，`select_experts()` 会在 hidden size 2048 维上取 top-k，因此产生 266、362、2003 这类小于 2048 但大于 127 的非法 expert id。
- 正确修复位置是 `AscendMoERunner._forward_impl()`，不是 `AscendFusedMoE.forward_impl()`；真实调用栈由 custom op 进入 `layer.runner._forward_impl()`，再委托到 `layer.forward_impl(...)`。
- fixed-slot 越界 id 护栏仍有价值：它不是根因修复，但能把错误从 host-store `KeyError` 提前为带 `layer_id/num_logical_experts/expert_ids` 的 `ValueError`，避免后续类似问题以模糊查表错误出现。
- 真实 Qwen3-30B-A3B NPU smoke 已通过：
  - artifact：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_runnergate_slots8_inline_prefetch/summary.json`
  - 状态：`ok`
  - 请求：1
  - 输出 token：1
  - `load_seconds=125.49`
  - `duration_s=1.77`
  - 日志显示 `Loading model weights took 46.7751 GB`，`PrefetchOffloader` 初始化 12 modules，static buffer pool `1.2080 GB`。
- 设计边界：这只证明 MVP-D fixed-slot sync correctness 的最小闭环已经通过真实硬件，覆盖 profile dummy run、slot load、slot remap、AllGather physical count 和 grouped MoE backend。它不是性能结果，也不能证明 SEW 自身已节省 HBM，因为当前 full expert 参数尚未从原模型结构释放，HBM 降低主要来自组合使用的现有 vLLM `prefetch` backend。

## 2026-06-01 MVP-D.5 correctness 对照与默认路径反查

- correctness 对照应以独立进程分别运行 baseline 和 candidate，而不是在同一 Python 进程中构造两个 `LLM` 实例；原因是 vLLM/NPU runtime、offloader 全局状态和 HBM 占用会互相污染。
- `run_fixed_slot_smoke.py` 已扩展为三种单次运行模式：
  - `no_offload`：清理所有 `VLLM_ASCEND_MOE_OFFLOAD_*` 环境变量，且不向 `LLM(...)` 传入 native weight offload kwargs；这比把 env 设成 `"0"` 更贴近真实默认路径，也避免 vLLM unknown-env warning。
  - `trace_only`：启用 SEW trace-only，但 `num_slots=0`，用于后续观测对照。
  - `fixed_slot_sync`：显式启用 fixed-slot sync，继续组合现有 `prefetch` offloader 作为当前 correctness 原型的 host/offload 触发方式。
- 新增 `outputs.jsonl` artifact，记录 `request_id/output_text/output_token_ids/output_tokens`。新增 `tools/sew_offload/compare_smoke_outputs.py`，默认只接受 token id 完全一致；不做文本相似度或容忍式比较，避免 correctness 门禁漂移。
- 真实 Qwen3-30B-A3B 单卡对照结果：
  - no-offload baseline：`artifacts/sew_offload/runs/no_offload_smoke_20260601_inline_1tok_cleanenv/summary.json`，状态 `ok`，`load_seconds=33.4295`，日志显示 `Loading model weights took 56.9001 GB`，输出 token id `[353]`。
  - fixed-slot candidate：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare/summary.json`，状态 `ok`，`num_slots=8`，`load_seconds=137.3632`，日志显示 `Loading model weights took 46.7751 GB` 和 `PrefetchOffloader` static buffer pool `1.2080 GB`，输出 token id `[353]`。
  - strict compare：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_inline_1tok_compare/correctness_compare.json`，状态 `ok`，`matched=1`，`mismatched=0`。
- 设计边界继续成立：该对照证明 1-token 最小输出一致性，不是性能结论；`fixed_slot_sync` 的 HBM 降低仍主要来自现有 vLLM prefetch backend，SEW fixed-slot 尚未释放/替换原始 full expert 参数。

## 2026-06-01 fixed-slot working-set 容量边界

- 2 prompt × 8 token smoke 中，baseline no-offload 能成功完成两条请求，但 fixed-slot `num_slots=8` candidate 在第二条较长 prompt 的 prefill 阶段失败，底层症状为 `active expert working set size 46 exceeds num_slots=8`。
- 该现象说明当前 fixed-slot correctness 原型的 slot budget 语义是“单层单次 MoE execution 的 active expert 并集上限”，不是“全模型缓存容量”或“长期 resident experts 总数”。
- 因此这个失败应被归类为容量预算不足的 fail-closed 成功案例，而不是执行正确性错误：
  - 不能 drop active expert；
  - 不能 clamp expert id；
  - 不能把多个 logical expert 强行映射到同一 physical slot；
  - 不能在 grouped MoE 前静默跳过超预算 expert。
- 直接提高到 `num_slots=64` 需要谨慎：当前 slot bank 按 layer 注册，slot tensor 在每个 MoE 层各自分配；对于 Qwen3-30B-A3B 这类 48 层模型，大幅增大 per-layer slots 会显著增加 HBM 占用。后续要支持更大 prefill working set，应优先审查 post-load 后释放/替换原 full expert 参数的生命周期，或重新设计更接近全局预算的 slot residency，而不是把 per-layer fixed slots 当作免费旋钮。

## 2026-06-01 2 short prompt × 8 token correctness 结论

- 在 `num_slots=8` 保持不变的情况下，使用两个短 prompt（`Hello`、`Hi`）各生成 8 token，可以通过 no-offload baseline 与 fixed-slot sync candidate 的独立进程严格 token-id 对照。
- artifact：
  - baseline：`artifacts/sew_offload/runs/no_offload_smoke_20260601_2short_8tok/outputs.jsonl`
  - candidate：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/outputs.jsonl`
  - compare：`artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/correctness_compare.json`
- compare 结果为 `status=ok`、`matched=2`，说明当前窄路径覆盖了多请求、8-token decode 的最小 correctness。
- 该结果与较长 prompt 的 `active working set size 46 > num_slots=8` 失败并不矛盾：前者验证 slot-budget-compatible decode correctness，后者暴露 prefill active expert 并集容量边界。两者共同支持下一步先做生命周期/容量模型审查，而不是直接进入 async transfer。

## 2026-06-01 fixed-slot memory ledger 与生命周期审查

- 当前 fixed-slot correctness 原型在权重生命周期上同时存在三份 expert 权重相关状态：
  - 原始 `layer.w13_weight/layer.w2_weight` 仍保留，用于默认路径和当前模型对象所有权；
  - `HostExpertStore` 对每个 expert 做 `.detach().cpu().clone()`；
  - `ExpertSlotBank` 为每层分配 `[num_slots, ...]` 的 backing tensors，供 grouped MoE backend 使用稳定 slot 地址。
- 因此当前不能声称 SEW fixed-slot 自身已经节省 HBM。现有真实 smoke 中 candidate 的 resident weight 降低主要来自组合使用的 vLLM `prefetch` backend，而不是 SEW 释放了 full expert 参数。
- 新增只读账本接口：
  - `MoeOffloadRuntime.memory_ledger()` 返回 registered layers、host experts、original expert bytes、host store bytes、slot bank bytes 和 total managed bytes；
  - `HostExpertStore.total_bytes` 统计 CPU clone；
  - `ExpertSlotBank.total_bytes` 统计 slot backing tensors。
- 新增离线估算工具：`tools/sew_offload/estimate_fixed_slot_memory.py`，默认使用 Qwen3-30B-A3B 的 48 层、128 experts/layer、单 expert `14,680,064` bytes。
- 默认 Qwen3-30B-A3B 估算结果：
  - `num_slots=8`：原始 expert 权重约 `90.19 GB`，host store 约 `90.19 GB`，slot bank 约 `5.64 GB`，当前原型 total managed 约 `186.03 GB`。
  - `num_slots=64`：slot bank 约 `45.10 GB`，当前原型 total managed 约 `225.49 GB`。
  - 若假设释放原始 expert 参数，`num_slots=8` 的 managed bytes 仍约 `95.83 GB`（host store + slot bank），这说明释放原始参数只是第一步，host store 表示和 slot 生命周期还需要进一步设计。
- 设计结论：下一步不应直接进入 async transfer，也不应把 `num_slots=64` 当作简单 smoke 参数。必须先定义 post-load 后原始 full expert 参数的所有权转移、vLLM loader/offloader 的引用边界、slot bank 的 per-layer vs 全局容量语义，以及失败时如何 fail closed。

## 2026-06-01 original expert release readiness guard

- 新增 `MoeOffloadRuntime.plan_original_weight_release(...)`，但它只是只读 readiness guard，不执行释放、不替换 `layer.w13_weight/layer.w2_weight`，也不改变默认路径。
- 该 guard 的 blockers 设计刻意保守：
  - `default_path_not_preserved`：调用者没有证明默认 no-offload 路径仍可使用原始参数；
  - `host_store_not_marked_complete`：调用者没有证明 host store 已完整持有 post-processed expert layout；
  - `original_expert_weights_still_retained`：当前 runtime ledger 仍看到原始 expert 参数字节数，除非调用者显式允许在 planning 阶段保留；
  - `layers_not_registered:[...]`：目标 MoE 层没有全部注册到 fixed-slot runtime。
- 设计反思：当前 `host_store_is_complete` 仍是人工前置条件，这是有意保守的中间状态。下一步应把它升级成 runtime 自检，例如按 layer 记录 expected expert count、host bundle count、layout signature、dtype/stride/device，以及与 slot bank shape 的 copy-compatible 校验结果。
- 这样做的价值是把“什么时候可以释放原始 full expert 参数”从口头判断变成可测试的 release plan，同时避免在当前 correctness prototype 中贸然释放参数，影响默认执行逻辑。

## 2026-06-01 host store completeness self-check

- `HostExpertStore` 现在在 `register_layer(...)` 时记录每层 expert 数量以及单 expert `w13/w2` 的 shape、dtype、stride。它仍然把 bundle `.detach().cpu().clone()`，因此自检要求 host bundle 的 device type 为 `cpu`。
- `validate_complete_layers(expected_layer_ids)` 会返回 `HostStoreCompletenessReport`，blockers 覆盖缺层、缺 expert、shape/dtype/stride mismatch 和非 CPU bundle。
- `MoeOffloadRuntime.plan_original_weight_release(...)` 默认调用 runtime 自检，不再需要调用方传 `host_store_is_complete=True` 才能通过；但传 `host_store_is_complete=False` 仍会添加 `host_store_not_marked_complete`，作为旧调用方的保守阻断。
- 设计结论：release readiness guard 已从“人工声明完整”推进到“runtime 可测试地证明完整”。下一步真正危险的部分不是 completeness，而是参数所有权转移：如何在 post-load 后释放/替换原始 full expert 参数，同时不破坏默认路径、weight loader/offloader 引用和 fixed-slot fallback。

## 2026-06-02 真实 trace 与 slot/residency sweep 结论

- 已补跨进程 trace JSONL 出口：`VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH`。原因与 profile 相同：vLLM V1 EngineCore 在子进程运行，父进程内存 collector 不能代表真实 MoE trace。
- 真实 trace artifact：
  - `artifacts/sew_offload/traces/d9_trace_short_20260602/trace.jsonl`
  - NPU 6，Qwen3-30B-A3B，1 条 synthetic `short_chat`，trace-only，不改变推理执行。
  - `num_trace_records=6192`，48 层，每层 129 条记录。
- trace 形态：
  - `active_experts_min=8`、`p50=8`、`p90=8`、`max=128`。
  - prefill 记录可在单层触达 81 到 128 个 active experts；decode 记录常见为 1 token × top-8，即 8 个 active experts。
  - 这进一步确认：token 数是动态 count，不应固定每 expert token capacity；真正困难是 prefill 阶段 active expert working set 太大。
- per-layer fixed-slot 读法：
  - 对这条 trace，`num_slots=8/16/32/64/96` 时所有 48 层都至少一次超过 slot budget。
  - `num_slots=128` 才不会因 active expert count 超 budget 而 fail-closed，但 per-layer slot bank 的 HBM 成本会随 48 层线性放大，不适合作为默认扩大路径。
- global LRU simulator 读法（设计参考，不代表当前 runtime 已实现 global pool）：
  - slots=8/32/128：`hit_count=0`，miss 全部暴露，说明简单 LRU 被 layer-wise 访问模式冲掉。
  - slots=512：`hit_count=21115`、`miss_count=37267`、估算 host-to-HBM bytes 约 547GB。
  - slots=1024：`hit_count=33294`、`miss_count=25088`、估算 host-to-HBM bytes 约 368GB。
  - `sticky_layer_lru` 在 512/1024 上与普通 LRU 近似，没有解决核心问题。
- 设计反思：
  - 当前 per-layer slot bank 适合验证 fixed weight entry 与 remap correctness，但不适合直接承接 prefill-heavy workload。
  - 简单 global pool 不是银弹；若进入 global pool，需要 layer/window-aware policy、prefill/full-resident 策略、decode cache window 和异步 prefetch 一起设计。
  - 下一步应先定义 prefill/resident-aware 分层策略：prefill 或高 fan-out 层保留完整权重或走不同阶段，decode/低 fan-out 窗口才使用 CPU-backed slot cache。

## 2026-06-02 prefill/resident-aware 分层策略离线验证

- 新增离线策略分析器：
  - `vllm_ascend/moe_offload/layered_strategy.py`
  - `tools/sew_offload/analyze_layered_strategy.py`
  - UT：`tests/ut/moe_offload/test_layered_strategy.py`
- 策略契约：
  - `active_experts > fanout_threshold` 的 MoE 调用走 `full_weight_path`，不 drop、不 clamp、不强行 slot remap；这主要对应 prefill / 高 fan-out。
  - `active_experts <= fanout_threshold` 的 MoE 调用走 `slot_cache_path`；这主要对应 decode / 低 fan-out。
  - cache scope 分两类验证：`global` 表示全模型共享小池；`per_layer` 表示每层保留自己的 decode slot window。
- 真实 trace 离线 sweep artifact：
  - `artifacts/sew_offload/traces/d9_trace_short_20260602/layered_strategy_sweep_summary.json`
  - 输入 trace 仍是 1 条 synthetic `short_chat`，6192 records。
- 关键结果：
  - 所有 slots=8/16/32/64 配置下，`full_weight_records=95`，且涉及全部 48 层。这说明高 fan-out prefill 不能靠小 slot cache 安全承接。
  - `cache_scope=global` 时，即便先过滤高 fan-out，slots=8/16/32/64 的 slot-cache hit 仍为 0；层间执行顺序会把小池冲掉。
  - `cache_scope=per_layer` 时命中显著改善：
    - slots=8：hit rate 约 31.4%，miss 33484，估算 H2D 约 457.8GiB。
    - slots=16：hit rate 约 59.7%，miss 19655，估算 H2D 约 268.7GiB。
    - slots=32：hit rate 约 78.9%，miss 10310，估算 H2D 约 141.0GiB。
    - slots=64：hit rate 约 93.9%，miss 2979，估算 H2D 约 40.7GiB。
- 设计判断：
  - “高 fan-out full-weight + per-layer decode slot cache”是当前 trace 支持的候选路线。
  - 简单 global 小池应作为反例，不应进入 runtime 主线；若做 global pool，必须是 window-aware / layer-aware，而不是朴素 LRU。
  - slots=64 的 decode cache 效果最好，但 per-layer slot bank HBM 约 42GiB，可能抵消 offload 目标；slots=32 HBM 约 21GiB、hit rate 约 78.9%，更像下一步原型的保守点。
  - 下一步 runtime 原型应保持默认关闭，并只做分层路径选择：高 fan-out 使用原始/常驻权重，低 fan-out 使用当前 fixed-slot path；不要引入 token drop/pad。

## 2026-06-02 源码复核：非 offload MoE 没有默认 Static Expert Window

- 结论：用户判断成立。当前 `vllm-ascend-hust` 非 offloading MoE 默认是 dropless dynamic count，而不是固定“每个 expert 最多/必须处理 N 个 token”的 Static Expert Window。
- Python 主路径证据：
  - `AscendUnquantizedFusedMoEMethod.apply()` 先通过 `select_experts(...)` 得到 `topk_ids/topk_weights`，再进入 `moe_comm_method.fused_experts(...)`。
  - `MoECommMethod.fused_experts()` 的结构是 `token_dispatch(...) -> build_mlp_compute_input(...) -> _apply_mlp(...) -> token_combine(...)`。
  - AllGather dispatcher 调用 `DeviceOperator.npu_moe_init_routing(... active_num=num_tokens * top_k, expert_tokens_num_type=1, expert_tokens_num_flag=True ...)`，没有传 `expert_capacity` 或 `drop_pad_mode`，随后设置 `group_list_type = 1`，即 count mode。
  - `moe_mlp.py` 把 dispatch 返回的 `group_list/group_list_type` 直接传给两次 `torch_npu.npu_grouped_matmul(...)`，没有在 Python 层做 per-expert capacity pad/drop。
  - All2AllV 路径同样由 `torch.histc(topk_ids, ...)` 计算每个 expert 的真实 token 数，并返回 `group_list_type=1`。
  - MC2 路径返回 `expert_token_nums` 和 `group_list_type=0`，这是 dispatch 产生的真实分组信息，不是固定 token capacity。
- C++/custom op 证据：
  - `csrc/torch_binding.cpp` 中 `npu_moe_init_routing_custom` 默认 `expert_capacity=-1, drop_pad_mode=0`。
  - `moe_init_routing_custom_torch_adpt.h` 中只有 `drop_pad_mode == 1` 才分配 `[expert_num, expert_capacity, h]`；默认 `drop_pad_mode == 0` 分配 `[num_out_tokens, h]` 或 `[bs * k, h]`。
  - 因此 fixed capacity/drop-pad 是底层算子能力，不是当前非 offload Python 主路径默认语义。
- 设计修正：
  - 不应把固定 token capacity 作为默认路线，也不应用 drop/clamp/remap active experts 解决 slot 不足。
  - 真正要固定的是 expert weight 的入口、地址稳定性和驻留/搬运窗口；token 分组继续跟随现有 dynamic count。
  - 当前 fixed-slot hook 位于 `apply()` 中、`fused_experts()` 之前，适合做短期权重替换和 `log2phy/physical_expert_count` remap。
  - 真正的 Ascend NPU MoE offload 切入点应下沉到 `MoECommMethod.fused_experts()` 内部：在 `token_dispatch_output` 已经产生 `sorted_hidden_states + group_list + combine_metadata` 之后、`_apply_mlp()` 之前，对 active experts 按 resident / staged / miss 分类，分阶段执行 MLP，再把阶段输出合并回完整 permuted token buffer，最后复用现有 `token_combine()`。
  - 长期性能切入点还应关注现有 `dispatch_ffn_combine` / `dispatch_gmm_combine_decode` 一类 fused Ascend op，因为它们已经把 dispatch、GMM、combine 和 AIC/AIV/MTE 协同放进 kernel；Python 分阶段只能先保证语义正确，真正隐藏搬运开销需要后续下沉到 fused kernel 或新增 staging-aware custom op。

## 2026-06-02 下一阶段规划判断

- 下一阶段不应直接做 async transfer。原因是当前仍缺一个在线 runtime 决策层：真实请求到来时，系统必须先知道该 MoE 调用应该走 full-weight、slot-cache 还是 fail-closed。
- MVP-D.10 的目标是把离线分层策略变成在线、默认关闭、可观测的 path selector：
  - `active_experts > fanout_threshold`：高 fan-out，优先走 full-weight path；这要求该 layer 的原始/常驻 full expert 权重仍在 NPU 且未被 release。
  - `active_experts <= fanout_threshold`：低 fan-out，走当前 fixed-slot sync path；继续复用 dynamic count 的 `topk_ids -> log2phy -> group_list` 语义。
  - full-weight 不可用且 active experts 超过 slot budget：必须 fail closed，不能 drop/clamp expert，也不能把 token 静默丢给错误 slot。
- MVP-D.10 的非目标：
  - 不做 phase split；一次 MoE 调用仍只选择一个 path。
  - 不做 async overlap；所有 miss load 仍可同步。
  - 不改 router/top-k，不引入 `expert_capacity/drop_pad_mode=1`。
  - 不支持 All2All/MC2/quant/bias 复杂路径，继续按现有边界 fail closed。
- MVP-D.11 才进入 dispatch 后 phase split 语义原型：
  - 在 `token_dispatch_output` 之后切 active expert phases。
  - 先用 Python/CPU mock 或小张量测试证明 phase 输出回填到完整 permuted buffer 后，`token_combine()` 等价于 single phase。
  - 不追求性能，只追求语义正确和不变量明确。
- MVP-E 才进入 async transfer：
  - 在 D.10/D.11 已经知道 path/phase 的前提下，给 miss expert 加 load stream/event、wait time、copy time、exposed stall metrics。
  - 若 phase split overhead 大于可隐藏搬运时间，必须有 single-phase fallback。
- 长期 D.12 方向：
  - Python 分阶段只是语义脚手架。若要真正用好 Ascend AIC/AIV/MTE，应把 staging-aware 逻辑下沉到 fused dispatch/GMM/combine 或新增 custom op。
  - Global pool 不是普通 LRU，应设计成 layer/window-aware：至少知道 decode 层序、resident layers、prefill full path 和 per-layer reuse window。

## 2026-06-02 用户已验证的 Ascend expert cache plugin 经验

- 用户提供了曾经成功实施的外部 plugin 路线：`VLLM_PLUGINS=ascend,moeinf_official` 后，plugin patch vanilla vLLM Worker 或 vLLM-Ascend NPUWorker 的 `load_model`，模型加载完成后调用 `activate_ascend_expert_cache(model)` 和 `activate_vllm_moeinf_official(model)`。
- 该方案的关键机制：
  - 只扫描 MoE `FusedMoE` 层，只处理 `w13_weight` / `w2_weight`，dense、shared、attention、embedding 权重不动。
  - 把每层 expert 权重复制到 CPU DRAM：`cpu_w13 = w13.detach().to("cpu").contiguous()`，`cpu_w2 = w2.detach().to("cpu").contiguous()`。
  - 可选释放原始 NPU expert tensor：`MOE_INFINITY_EXPERT_CACHE_KEEP_ORIGINAL=0` 时把原 NPU expert tensor 替换为空 tensor，释放 HBM。
  - 每个 MoE 层在 NPU 上分配 `pool_w13/pool_w2`，shape 为 `[slots, ...]`，slot 数可固定配置，也可由 cache GB budget 计算。
  - 最关键 hook 是 patch `vllm_ascend.ops.fused_moe.moe_comm_method.MoECommMethod.fused_experts`，因为此时 router `topk_ids` 已经可见，可以知道本次真实 active experts。
  - 运行时根据 `topk_ids` 做 hit/miss、CPU->NPU slot load、LRU/priority eviction、logical expert id 到 slot id remap，再把 fused op 输入 `w1/w2` 替换为 `pool_w13/pool_w2` 后调用原始 vLLM-Ascend fused experts。
- 与当前仓库内实现的对应关系：
  - `HostExpertStore` 已覆盖 CPU expert store，但当前是按 expert clone，而用户 plugin 描述是每层 `cpu_w13/cpu_w2` contiguous tensor；后续需比较两种 CPU layout 对 H2D copy 和索引开销的影响。
  - `ExpertSlotBank` 已覆盖 per-layer NPU pool。
  - `MoeOffloadRuntime.prepare_fixed_slot_plan(...)` 已覆盖 miss/load/remap 的同步正确性原型。
  - 当前主接入更多发生在 `AscendUnquantizedFusedMoEMethod.apply()`，而用户已验证方案选择 `MoECommMethod.fused_experts`。D.10 应优先把 hook 下沉/内生化到 fused experts 边界，减少与 router 前逻辑耦合。
- 本机源码状态：当前 `/root` 下未找到 `adapters/vllm_moeinf_official_plugin.py` 或 `adapters/ascend_expert_cache.py`，因此暂以用户提供的实现步骤作为设计证据；若后续能提供源码，应做逐行迁移审查。
- 对 D.10 的影响：
  - D.10 不应从抽象 phase split 开始，而应先复刻这条已成功路线的 in-tree 版本：load 后注册 host store + per-layer pool，fused experts 边界按 active experts 做 cache/remap。
  - `KEEP_ORIGINAL=0` 对应当前 `release_original_expert_weights`，但必须配合 high fan-out full-weight readiness guard，避免 prefill 需要 full path 时原权重已释放。
  - cache GB budget 自动换算 slots 是很有价值的工程入口，后续可作为 `num_slots` 的替代配置，但第一步仍保留显式 `num_slots` 降低变量数量。

## 2026-06-02 D.10 在线分层 runtime 设计结论

- 已验证的最小正确路径：
  - `active_expert_count > fanout_threshold`：走 `full_weight_path`，前提是原始/常驻 full expert 权重仍可用。
  - `active_expert_count <= fanout_threshold`：走 `slot_cache_path`，复用 fixed-slot sync path、`log2phy` 和 `physical_expert_count`，不改 token count / group_list 语义。
  - high fan-out 且 full weights 已 release：走 `fail_closed`，不得 drop/clamp/remap active experts。
- 真实 NPU 结果说明：
  - D.10 candidate 可在 Qwen3-30B-A3B / Ascend NPU 6 上正常推理，并与 no-offload 严格 token-id 一致。
  - 8-token candidate profile 中 `full_weight_path=1`、`slot_cache_path=8`，证明 prefill/decode 的在线分层实际发生。
  - candidate reported model weight `43.4704 GB`，主要来自 native `PrefetchOffloader` 组合；SEW layer0 slot bank ledger 约 `75.5 MB` slot + `1.208 GB` host store + `1.208 GB` retained original。
  - candidate 指标：throughput `1.675 tok/s`、TTFT `868.72 ms`、TPOT `558.14 ms`；no-offload baseline：throughput `6.337 tok/s`、TTFT `445.15 ms`、TPOT `116.71 ms`。
- 设计反思：
  - 当前实现是 correctness MVP，不是性能优化结果。慢的主要原因是同步 CPU->NPU slot load、Python path decision/CPU unique、native prefetch offloader 组合以及每层/每步缺少 overlap。
  - `fanout_threshold=8` 对 1-token decode 很自然，因为 top-k=8；对 prefill 则会转 full path，避免 slot capacity 改变语义。
  - `release_original_expert_weights=True` 若扩大到需要 high fan-out 的层，会与 full path 冲突；后续 release policy 必须与 resident/full-weight path selector 联动，而不是按层盲 release。
  - `MoECommMethod.fused_experts` 下沉仍然重要：只有进入 dispatch/MLP 边界，后续 D.11 才能做 resident/miss phase split，并最终给 MVP-E async transfer 提供可重叠窗口。

## 2026-06-02 D.10 fused boundary 下沉后的设计反思

- 下沉到 `MoECommMethod.fused_experts()` 的设计判断：
  - 这是比 `AscendUnquantizedFusedMoEMethod.apply()` 更合理的 offload 边界，因为此处已经携带真实 `topk_ids`、当前 layer 权重和 routing metadata，且仍位于 token dispatch 前，可以安全替换权重和 `log2phy/physical_expert_count`。
  - `apply()` 只传 metadata 可以降低耦合：router、load-balance、zero-expert、top-k 选择仍在原路径完成，offload runtime 不需要提前理解更上层 MoE 细节。
  - 下沉后仍保持“一次 MoE 调用选择一个 path”，没有做 phase split；这符合 D.10 范围，也避免在尚未 E2E 复验时引入回填/合并等新语义风险。
- correctness 不变量复核：
  - slot path 必须在 dispatch 前完成 logical expert id 到 physical slot id 的 remap，并同步设置 `physical_expert_count`；否则 `group_list` 和 `w1/w2` 第 0 维会不一致。
  - full path 必须保留原始 full expert 权重；如果原权重已 release，则 high fan-out 只能 fail closed。
  - fail-closed 的位置应早于 token dispatch，避免部分 dispatch 后才发现缺权重导致状态难以恢复。
- 当前硬件 smoke 阻塞判断：
  - post-downsink 三次 smoke 都失败于 vLLM worker startup memory gate，尚未进入模型加载和 MoE forward。
  - NPU 6 显示约 44GB HBM 残留账本，但 Linux 进程表查不到对应 PID；这是资源/设备上下文问题，不是 MoE offload path 的反证。
  - 在用户未授权前，不应 reset NPU 或 kill 未知上下文；正确做法是记录阻塞、等待资源释放后用同一命令补跑。
- 下一步优先级：
  1. 资源恢复后优先补跑 `d10_fused_boundary_layered_1tok` post-downsink smoke，并与 `d10_no_offload_1tok_20260602/outputs.jsonl` 做 strict compare。
  2. 通过后再跑 8-token post-downsink smoke，检查 `layered_path_decision` 中是否仍是 prefill `full_weight_path` + decode `slot_cache_path`。
  3. 再进入 D.11 dispatch 后 phase split；不要在 post-downsink smoke 未闭环时直接做 async transfer。

## 2026-06-02 D.11 dispatch 后 phase split 规划发现

- D.11 的正确目标：
  - 验证“dispatch 后把 active experts 分成少量 phase 执行，并回填到完整 permuted token buffer 后，最终 combine 结果等价于 single-phase MLP”。
  - 这是语义原型，不是性能优化；Python 多 phase 很可能更慢，但能证明后续 async overlap 的 buffer contract。
- D.11 的非目标：
  - 不启动 async CPU->NPU transfer；只允许同步 ready/miss 判定。
  - 不改变 router/top-k/topk_weights。
  - 不启用 `expert_capacity/drop_pad_mode=1`，继续使用当前 dynamic count/group_list。
  - 不支持 EP/EPLB/All2All/MC2/quant/bias；这些路径继续 fail closed。
  - 不把每个 expert 拆成小 kernel；phase 数必须受控，默认最多 2 到 3 个 phase。
- 推荐 contract：
  - `MoEPhase`: phase id、logical experts、physical experts、每个 expert 在 `sorted_hidden_states` 中的 token slice、phase path/reason。
  - `MoEPhasePlan`: 原始 group_list、phase 列表、完整输出 shape、fail-closed blockers。
  - 回填不变量：每个 token row 必须被恰好写入一次；不能重复写、不能漏写、不能改变 token 顺序。
- 技术切入点：
  - phase split 发生在 `MoECommMethod.fused_experts()` 内部、`token_dispatch_output` 之后、`build_mlp_compute_input/_apply_mlp` 之前。
  - 现有 `token_combine` 应保持不变；D.11 只负责构造与 single MLP 等价的完整 `mlp_output`。
  - UT 应先用 mock MLP 或小张量线性函数验证回填等价性，避免一上来依赖 NPU grouped matmul。
- 关键风险：
  - `group_list_type=1` 是 cumulative count 还是 per-expert count 必须以现有 dispatcher/MLP contract 为准，不能凭直觉切 slice。
  - 如果 phase 内 active experts 不是连续 expert id，可能需要重建 phase-local `group_list/log2phy/physical_expert_count`，否则 grouped matmul 会读错权重行。
  - Python 多次 `_apply_mlp` 会增加 launch overhead；D.11 结论即使性能差，也可作为 D.12 fused/custom op 的语义脚手架。
- 验证门禁：
  - 单测必须证明 single phase 与 multi phase 输出一致。
  - post-downsink D.10 1-token strict compare 仍是进入 D.11 真实 NPU smoke 的建议前置门禁。

## 2026-06-02 post-downsink D.10 1-token 验证结论

- `MoECommMethod.fused_experts` 下沉后的真实 NPU 1-token smoke 已通过：
  - artifact：`artifacts/sew_offload/runs/d10_fused_boundary_layered_1tok_retry4_20260602`
  - candidate token id `[1096]`，与 no-offload baseline `[1096]` 一致。
  - strict compare `status=ok`、`matched=1`。
- 设计含义：
  - 下沉到 fused boundary 后，slot path 的权重替换、`log2phy` remap、`physical_expert_count` 和后续 token dispatch/grouped matmul/combine 至少在 1-token 真实路径上是正确的。
  - profile 同时出现 `slot_cache_path` 和 `full_weight_path`，说明 layered decision 在真实 forward 中生效，而不是只在 UT 或离线 trace 中生效。
  - 1-token 的 TPOT 为 0 是正常现象，不能用于评估连续 decode；后续需要 post-downsink 8-token smoke 才能报告更有意义的 TPOT。
- 对 D.11 的影响：
  - D.11 的前置风险降低：现在若 phase split 后出现 token mismatch，更可能是 phase 回填/切分问题，而不是 D.10 fused boundary remap 本身未闭环。
  - 但进入 D.11 前仍建议补一个 post-downsink 8-token strict compare，因为 D.11 主要服务 decode 阶段，而 decode 稳定性需要多 token 输出观察。

## 2026-06-02 post-downsink D.10 8-token 验证结论

- 8-token post-downsink smoke 已通过：
  - artifact：`artifacts/sew_offload/runs/d10_fused_boundary_layered_8tok_20260602`
  - candidate token ids `[1096, 374, 264, 5052, 54, 12, 4596, 1078]`
  - no-offload baseline token ids `[1096, 374, 264, 5052, 54, 12, 4596, 1078]`
  - strict compare `status=ok`、`matched=1`
- 指标结论：
  - candidate throughput `1.648 tok/s`、TTFT `910.87 ms`、TPOT `563.37 ms`
  - no-offload baseline throughput `6.337 tok/s`、TTFT `445.15 ms`、TPOT `116.71 ms`
  - 当前 offload path 明显慢于 no-offload，只能作为 correctness/observability MVP，不能作为性能收益结果。
- 路径结论：
  - profile 中 1 次 high fan-out `full_weight_path` 与 8 次 decode `slot_cache_path` 同时出现。
  - 这证明 D.10 的分层决策在连续 decode 中稳定触发，slot-cache path 不只是 1-token 偶然通过。
- 阶段结论：
  - D.10 可以收口：默认关闭、fail closed、full/slot path selection、slot remap、真实 NPU 1-token/8-token correctness 和三指标输出均已闭环。
  - 下一步 D.11 的目标应保持为语义原型：分批计算和回填等价性，不直接承诺性能。

## 2026-06-16 — M2 根因诊断（决定性，修正了 Experiment A 的 generate 解读）

### 两套 offload 机制并存，缺一不可
- **PrefetchOffloader（vLLM 原生，数据面/设备驻留）**：`/root/vllm-hust/vllm/model_executor/offloader/prefetch.py`。
  - docstring 明确「torch.compile + CUDA graph compatible」：用 static GPU buffer + event-based fork 把 H2D copy 并入 graph capture。
  - `_CpuParamOffloader.assign_static_buffer()` 把 `param.data` 指向 **GPU static buffer**（prefetch.py:716），forward 前用 `wait_prefetch` custom op 等待 H2D 完成。
  - 由 autoconfig 通过 `--ascend-moe-offload-gb` → `offload_backend=prefetch` + `offload_params={experts}` 装配。
- **SEW fixed-slot runtime（控制面/expert 粒度）**：`vllm_ascend/moe_offload/`。
  - `process_weights_after_loading` 对非驻留层把 `w13/w2` stage 到 **CPU**（fused_moe.py:143）。
  - `_maybe_apply_moe_offload_plan` 决定 SLOT_CACHE / FULL_WEIGHT / FAIL_CLOSED 路径。

### 两者作用于**同一批非驻留层**
- `offload_enabled=True` 仅当 `should_use_fixed_slot_plan_for_layer(layer_id)`（fused_moe.py:291,308）= `enabled ∧ ¬resident`。
- autoconfig 的 `resident_layer_ids` 之外的层既被 PrefetchOffloader offload，也被 SEW 标记非驻留。
- **验证命令（--enforce-eager）能跑通的原因**：PrefetchOffloader 在 layer forward 前把 `layer.w13_weight` 变为设备驻留，于是 SEW 的高 fanout `FULL_WEIGHT_PATH`（runtime.py:378）读到**设备** tensor → 不崩。

### Experiment A 的 D-run（flag=1）generate 崩溃 = 探针配置错误，非 M2 边界
- 证据：C/D 两个 offline `LLM()` 日志**均无** "Enabled Ascend MoE offload autoconfig" 行、**无** PrefetchOffloader init 行；`VLLM_ASCEND_MOE_OFFLOAD_GB` 被 vLLM 当作 "Unknown env"（D 日志 21 行）。
- 即：offline 探针只用显式 `_ENABLED=1` 等启用了 **SEW fixed-slot**，但 **从未触发 autoconfig→PrefetchOffloader**。
- 于是非驻留层 `w13` 停在 CPU，高 fanout prefill 走 FULL_WEIGHT_PATH 读 CPU 权重 → `weight is on cpu @ npu_grouped_matmul`（acl_graph.py:122 非捕获 bypass，eager prefill）。
- **结论**：generate 崩溃**不**证明缺 staging hook，它证明探针漏装了设备驻留机制。

### Experiment A 的 capture 结论仍然成立
- capture 崩溃源是 `torch.unique(topk_ids).cpu()`（moe_comm_method.py:347）在被捕获流上的 D2H sync（107027/107030），与 PrefetchOffloader 是否激活**无关**。
- flag=0 capture 崩、flag=1 capture 过的对照仍有效。

### staging hook 仍是 M2 真缺口（独立确认）
- eager `prepare_fixed_slot_plan` 分配**新** `logical_to_physical`（slot_mapping.py:63）；capture-safe 路径读**持久 buffer**（runtime.py:621）。
- 只有 `stage_fixed_slot_plan` 在 eager 期把映射 in-place 写入持久 buffer（runtime.py:593）。
- 捕获图 replay 读持久 buffer + 固定 slot tensor；无 pre-replay staging → replay 读到陈旧 `-1`。

### M2 正确性验证的配置难题（关键设计决策）
- 部分驻留（num_slots=8 / 128 experts）：decode 每步 active expert 集合数据依赖，replay 前无法预知 → 仅 staging hook **不足以**保证 token 正确，需要完整 Option 2 的 split boundary。
- 全驻留（num_slots≥128）可让 active set 恒为 resident 子集、log2phy 静态 → staging 一次即正确；
  但**若与 PrefetchOffloader 叠加**会双份占用 HBM（slot bank 128 experts + prefetch buffer），30B 不可行。
- **修正后的 M2 验证设计**：仅用 SEW fixed-slot 作唯一 offloader（不挂 PrefetchOffloader），
  仅对少数非驻留层（2–4 层）设 `num_slots=128`、`fanout_threshold=128`（强制 prefill 也走 SLOT_CACHE，
  避免 FULL_WEIGHT_PATH 读 CPU），其余层 resident。这样 w13 设备驻留由 slot bank 保证，
  staging hook 在 replay 前写持久 log2phy，可验证 token 与 eager offload baseline 一致。内存有界。

## 2026-06-16 — 为何 offline 探针漏装 autoconfig（结构性，决定重跑方式）

- autoconfig monkeypatch（`patch/platform/patch_moe_offload_autoconfig.py`）只在 `adapt_patch(is_global_patch=True)`→`import patch.platform` 时把
  `EngineArgs.create_engine_config` 替换为带 autoconfig 的版本。
- `adapt_patch(global)` 由 `AscendPlatform.pre_register_and_update` 调用，存在两条触发点：
  - **CLI/api_server**：`arg_utils.py:2446` 在 `_add_cli_args`（**解析参数阶段**）调 `pre_register_and_update(parser)` → patch 在 `create_engine_config` 之前装好 → autoconfig 生效。
  - **offline LLM()**：`arg_utils.py:1634` 是**原始** `create_engine_config` 的第一行才调 `pre_register_and_update()`（无 parser）。
    即「在尚未打补丁的 create_engine_config 内部」才打补丁 → 只对“下一次”生效，而 offline 只调一次 → **autoconfig 永不触发**。
- 结论：offline `LLM()` 探针无法复现 validated 命令的 autoconfig→PrefetchOffloader。**修正版实验必须**：
  - (A) 用 validated api_server CLI（保证 line 2446 装补丁）；或
  - (B) 在 offline 探针 `from vllm import LLM` **之前** 显式 `import vllm_ascend.patch.platform`（等价 monkeypatch 提前装好），并仅设
    `VLLM_ASCEND_MOE_OFFLOAD_GB=14` + `VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE={0,1}`，其余 env 交给 autoconfig setdefault，忠实复刻 validated 配置。

## 2026-06-16 — E-run（修正版 flag=1）决定性结果：107025 根因 = PrefetchOffloader 悬挂 copy_stream

修正版探针（PrefetchOffloader 正确挂载：Initialized 12 modules / saved 14.4955 GB / static buffer 1.2080 GB；权重 44.24 GB）在
**capture 阶段**崩溃，错误号与之前不同：

```
capture_end: ... AclmdlRICaptureEnd ... error code is 107025
EE9999: rtStreamEndCapture execution failed,
  reason=capture model contains a stream that was not joined to the original stream
```

栈：`acl_graph.py:157 with torch.npu.graph(...)` → `graphs.py:397 __exit__` → `capture_end()`（_dummy_run 的 piecewise capture）。

### 根因（已逐条用源码核实）

1. **107025 ≠ 107027/107030。** 107027/107030 是 SEW eager 路径 `torch.unique(topk_ids).cpu()` 的 D2H 同步 memcpy；
   flag=1 的 SEW capture-safe 路径**已成功消除该 D2H 屏障**（capture 越过了它）。107025 是“**有侧流 fork 后未 join 回主捕获流**”。
2. **悬挂流来自 PrefetchOffloader（数据面），与 SEW 无关。** `prefetch.py` 用 `copy_stream`（line 156）做 H2D overlap，
   其所有 graph-capture 分支都 gate 在 `torch.cuda.is_current_stream_capturing()`（line 256/517/543）。
3. **该符号是“真 CUDA”符号，未被 NPU 别名化。** 实测：`torch.cuda.is_current_stream_capturing` 来自
   `torch/cuda/__init__.py → _cuda_isCurrentStreamCapturing()`；且 `torch.cuda.{Stream,Event,current_stream,
   is_current_stream_capturing}` 与 `torch.npu.*` **均非同一对象**。在无 CUDA 的 NPU 机器上，
   `torch.npu.graph()` 捕获期间它**恒为 False**。
4. 后果链：`_prefetch_in_capture`（prefetch.py:517）恒 False → `join_after_forward()`（prefetch.py:306 仅 join
   `_prefetch_in_capture=True` 的层）**恒空操作** → 最后一层 prefetch 的 `copy_stream` fork 永不被 rejoin →
   `AclmdlRICaptureEnd` 报“stream not joined”。
5. **叠加放大：vllm-ascend 的 ACLGraphWrapper 漏移了上游 offloader 同步钩子。** stock vLLM `CUDAGraphWrapper`
   在 capture 前后/replay 前调用 `get_offloader().sync_prev_onload()`（cuda_graph.py:310/359）+
   `join_after_forward()`（cuda_graph.py:324）；`vllm_ascend/compilation/acl_graph.py` 的对应位置（155–167、199–213）
   **三处全缺**。model_runner 也只调了 `get_offloader().post_init()`（model_runner_v1.py:3076），未调 join。

### 结论与修复方向

- 这是 **vllm-ascend 忠实移植遗漏 + torch.cuda/npu 符号未别名化** 的双重数据面 bug，**不是 SEW 缺陷**，也独立于 M2 staging hook。
- 单纯忠实补回 `join_after_forward()` **无效**——它在 NPU 上恒空操作（依赖 `is_current_stream_capturing`）。
- 正确修复：在 ACLGraphWrapper 的 capture 块内（`self.runnable(...)` 之后、`torch.npu.graph` 退出之前）调用
  **无条件** `get_offloader().sync_prev_onload()`（prefetch.py:282 = `wait_stream`，不 gate 捕获状态），把 copy_stream
  drain 进捕获流；并在 capture 前 / replay 前各补一处 `sync_prev_onload()` 以对齐 stock。NoopOffloader 下三处均为 `pass`，
  对非 offload 主路径零影响。
- 论文价值：这印证了“数据面 device-residency offloader（CUDA-graph 兼容设计）直接搬到 Ascend ACLGraph 会在
  **流捕获状态探测 API 边界**失效”——CUDA 的 graph 兼容技巧不能直接复用于 NPU。强化 SEW 控制面/数据面解耦的论证。

## 2026-06-16 — F-run（drain fix, flag=1）：107025→107024，证实 PrefetchOffloader 在 NPU ACLGraph 下根本不可捕获

在 ACLGraphWrapper capture 块内补上无条件 `get_offloader().sync_prev_onload()`（drain copy_stream）后：

- **107025（stream not joined）消失** —— 说明“悬挂未 join”确实是 copy_stream 引起。
- 失败下沉一层为 **107024**：
  `rtStreamWaitEvent execution failed, reason=in the model capture scenario,
   the event wait task has no corresponding event record task`（log 行 472/525）。
  栈：`worker.py:740 capture_model` → `model_runner_v1.py:3893` → ACLGraphWrapper capture。

### 决定性结论

- `sync_prev_onload()` = `current_stream().wait_stream(copy_stream)`，在 NPU 上等价“在 copy_stream 上 record event，
  再让捕获流 wait 该 event”。但 **copy_stream 从未被纳入本次 capture**（其 fork 受 `torch.cuda.is_current_stream_capturing()`
  门控，NPU 恒 False），所以被 wait 的 event 在**捕获图内没有对应 record** → 107024。
- 即：**fork 悬挂会 107025，强行 drain 会 107024**。copy_stream 对 NPU ACLGraph capture 是“外来流”，
  wrapper 级 drain 无法修复——copy 工作压根不在图里。**数据面 PrefetchOffloader 在 NPU ACLGraph 下根本不可捕获**，
  根因在 **流捕获状态探测 API 边界**（CUDA `is_current_stream_capturing` 未被 NPU 别名化）。
- acl_graph.py 的 drain 改动是**忠实对齐 stock vLLM 的 parity**，且在 NoopOffloader 下为 `pass`（非 offload 主路径零影响）；
  它对 PrefetchOffloader 无法奏效，但**对 SEW-only 路径完全惰性**（get_offloader()=NoopOffloader）。保留无害。

### 两条前进路径（待定）

- **路径 A（论文主线）：SEW-only 捕获验证。** 不挂 PrefetchOffloader（autoconfig 不设 offload_backend=prefetch），
  以 SEW fixed-slot capture-safe path 作唯一权重驻留机制（固定 slot + 持久 log2phy buffer，零 host sync、零外来流）。
  预期：107027/107030（flag=1 已清）+ 107024/107025（无 copy_stream）全部消失 → capture 通过。
  需一个“可捕获性”配置：少数非驻留层（2–4 层）、num_slots≥该层激活专家全集，先证 capture+generate 正确，再谈部分驻留。
- **路径 B（baseline/消融）：让 PrefetchOffloader 的 capture 探测 NPU-aware。** patch 其
  `is_current_stream_capturing` 解析到 `torch.npu` 版本，使 fork_event 真正把 copy_stream 纳入 capture。
  若成立则数据面 offload 也能在 ACLGraph 下工作——可作 SEW 的对照，但属厂商集成修复，非论文核心贡献。

## 2026-06-16 — G-run（SEW-only, flag=1）：capture 通过 + generate 跑通（token 正确性待 parity 验证）

配置：不设 GB（→ NoopOffloader，无 copy_stream），直接设 SEW env：
`ENABLED=1 / NUM_SLOTS=128 / FANOUT_THRESHOLD=128 / LAYERED_RUNTIME=1 / MAX_PHASES=1 / GRAPH_COMPATIBLE=1`，
非驻留层 = {2,3}（46 层驻留）。脚本：`tools/sew_offload/race_launch_sew_only.sh`。

结果（log：.planning/sew_offload/logs/G_sewonly_flag1.log）：
```
VLLM_ASCEND_MOE_OFFLOAD_GB None           ← NoopOffloader（grep PrefetchOffloader 计数=0 确认）
GRAPH_COMPATIBLE 1
Capturing CUDA graphs (PIECEWISE): 100% 1/1
LOAD_OK seconds=140.053                    ← 含 capture_model，捕获通过
GENERATE_OK seconds=6.494
OUTPUT_TOKENS [3555, 525, 279, 1376, 6813, 315, 1741, 4119]
OUTPUT_TEXT  What are the key components of such models
```
**107024/107025/107027/107030 全部消失。** 无 "on cpu"/device-mismatch。
旁证：独立进程 `MoeOffloadConfig.from_env()` 用同一组 env → enabled=True/num_slots=128/graph_compatible=True/
46 层驻留/层2,3非驻留，确认 SEW 真激活（非空跑）。

### 论文主线结论（capture 维度，已坐实）
SEW 控制面 fixed-slot capture-safe primitives 在 NPU ACLGraph PIECEWISE 下**可捕获**，
无需 PrefetchOffloader、无需 --enforce-eager。这正是论文要证明的"控制面/数据面解耦使 MoE offload 可图捕获"。

### 重要保留：token 正确性尚未证明（不可凭输出连贯下结论）
- 机制分析：eager 路径 `prepare_fixed_slot_plan` 经 `slot_mapping.from_slot_bank` 生成**全新** log2phy（fresh -1 张量），
  **不写持久 buffer**；capture 路径 `capture_safe_slot_weights` 读**持久 buffer**，而持久 buffer 仅由
  `stage_fixed_slot_plan`（需 model_runner staging hook，当前未装）写入。
- 故 capture 时持久 buffer 极可能为全 -1 → replay 时层 2,3 的 log2phy 索引为 -1 → expert gather 取错 slot。
- 输出仍连贯，最可能因 48 层仅 2 层非驻留、其余 46 层全对，掩盖了 2 层的错误。
- **下一步必做 token-id parity 对照**：无 offload 全驻留 ACLGraph baseline（相同 prompt/seed）取 ground-truth；
  若 G-run tokens != baseline → 证实 replay 需 staging hook（M2 真正边界），capture-pass 与 token-correct 是两件事。

## 2026-06-16 — 初步 token parity（A vs G）：分叉确认 capture-pass ≠ token-correct

发现 `A_control_flag0_aclgraph`（08:42 跑，**早于探针装 autoconfig 补丁**）实为**无 offload 全驻留 baseline**：
`Loading model weights took 56.9001 GB`（全量权重）、无 AUTOCONFIG_PATCH_ARMED、无 PrefetchOffloader、无 register_layer
→ 其 GB=14 被当 Unknown env 忽略，offload 从未激活。恰好提供 ground-truth。

相同 prompt/seed=0/temp=0/ACLGraph PIECEWISE 下：
```
A (无 offload 全驻留):       [3555, 525, 279, 22146, 323,  63625, 315, 1667]  "What are the advantages and disadvantages of using"
G (SEW-only 层2,3 offload): [3555, 525, 279, 1376,  6813, 315,   1741, 4119]  "What are the key components of such models"
                            └ 同 3 token ┘ └──────── 从 index 3 起分叉 ────────┘
```
**前 3 token 一致、第 4 个起分叉。** 若 G 的 SEW 是空操作则应与 A 完全相同；分叉证明 SEW 确实改变了层 2,3 的计算 —
与"捕获路径读持久 log2phy buffer 恒 -1 → 层 2,3 路由到 slot[-1] → 算错"机制一致。
**初步坐实 capture-pass ≠ token-correct。**

### 注意（严谨性）：A 是"碰巧"的 baseline，仍需当前代码上的受控对照
- A 早于本次 acl_graph.py 改动（该改动在 NoopOffloader 下惰性，理论不影响 A，但需复现）。
- 真正决定性对照是 **eager-SEW**（--enforce-eager + 同一组 SEW env）：eager 路径每步跑
  `prepare_fixed_slot_plan` 生成**正确的 fresh log2phy**。预期：
  - 若 eager-SEW == A（无 offload）→ eager offload token 正确 → 唯一缺口就是捕获期 staging hook；
  - 则 G（captured SEW）≠ eager-SEW 精确隔离出"持久 buffer 未被 staging hook 写"这一个原因。
- 两个对照（当前代码无 offload baseline + eager-SEW）均需空闲卡；当前 8 卡全忙（86-93%），
  `race_launch_baseline.sh` 后台轮询等卡中。

## 2026-06-16 — 代码事实坐实 capture-pass≠token-correct 的精确根因（无需跑实验即可证）

逐函数追踪持久 log2phy buffer 的"写"与"读"，定位 G 分叉的机制根因：

### 持久 buffer 生命周期
1. **分配/初始化**（`runtime.py:309`，`register_layer_for_fixed_slots` 内）：
   `_log2phy_buffers[layer_id] = torch.full((num_logical_experts,), -1, int32, device)` —— 固定地址，初值全 -1。
2. **唯一写入者**：`stage_fixed_slot_plan`（`runtime.py:592-593`）`buf.copy_(prepared.log2phy)`。
   - **live 调用点 = 0**。`grep stage_fixed_slot_plan` 全部命中在 `tests/ut/moe_offload/`，
     加上 `moe_comm_method.py:338` 一条**注释**、`runtime.py` 内 def/docstring。生产路径无人调用。
3. **捕获期读取者**：`capture_safe_slot_weights`（`runtime.py:602`）→ 经
   `ExpertSlotMapping(logical_to_physical=buf)` → `PreparedSlotWeights.from_slot_bank`
   （`slot_mapping.py:145` `log2phy=mapping.logical_to_physical`）→ 把**持久 buf 本体**交给执行。
   - **live 调用点**：`moe_comm_method.py:341`，在
     `if graph_compatible_offload and _is_current_graph_capturing():` 分支内。
4. 捕获图内 gather：`moe_comm_method.py:149/529` `topk_ids = log2phy[topk_ids]`，
   按固定地址录进图；replay 复用 buf 当前内容。

### 结论（airtight）
> 捕获路径**读** persistent buf；唯一**写**者 `stage_fixed_slot_plan` 在生产路径**无调用点**。
> 故 offload 层（2,3）的 buf 恒为初值 **-1**，每次 replay 的 `log2phy[topk_ids]` 取到 -1
> → slot 索引错误 → 路由错误 → 输出从 token 3 起分叉（与 G 实测一致）。

这把"capture-pass ≠ token-correct"的根因**精确收敛到一个缺失的 eager pre-replay staging hook**：
需要在 model_runner 的 replay 前对每个 offload 层调用 `stage_fixed_slot_plan`（host 决策+H2D+写 buf）。
**这是 SEW 控制面原语正确性之外的"接线缺口"，不是原语逻辑 bug。** 该 hook 触及 model_runner，属架构评审门控（M2 真边界）。

> 注：eager 路径（`_is_current_graph_capturing()=False`）走 `moe_comm_method.py:346+`，
> 用 `torch.unique().cpu()` 算 active_experts + slot_mapping 生成 **fresh 正确 log2phy**，不读持久 buf。
> 故 eager-SEW 对照（H_sew_eager_flag1）预期 token 正确（== BASE）—— 待空闲卡跑通验证。

## 2026-06-16 — eager-SEW 对照（H）结果：预测被推翻，根因诊断需修正

**实测（NPU5，--enforce-eager + 同一组 SEW env + 同 resident CSV，FANOUT=128/num_slots=128/graph_compat=1）：**
```
BASE (no offload):      [3555, 525, 279, 22146, 323, 63625, 315, 1667]
G    (captured SEW):    [3555, 525, 279, 1376,  6813, 315,   1741, 4119]
H    (eager SEW):       [3555, 525, 279, 1376,  6813, 315,   1741, 4119]   ← == G，逐 token 完全一致
```
config dump 确认 H：`enforce_eager=True`、`CompilationMode.NONE`、`cudagraph_mode=NONE`、
`AUTOCONFIG_PATCH_ARMED True`、`GRAPH_COMPATIBLE 1`、SEW env 全设、`LOAD_OK`+`GENERATE_OK`。

### 关键推翻
- 先前预测 **H == BASE**（eager 路径用 `torch.unique().cpu()`+fresh log2phy，不读持久 buf，应正确）。
- 实测 **H == G ≠ BASE**：eager 路径**不碰** persistent `-1` buffer，却与 captured **同样分叉**。
- ⟹ "持久 `-1` buffer 是 G 分叉根因" 的结论**被推翻**。`-1` buffer 确是 capture 路径的真实缺陷
  （代码事实不变），但它**不是** G 观测分叉的成因——因为 eager 不读它却同样错。
- ⟹ 分叉成因是**两条 SEW slot 路径（含 eager）共有**的：fixed-slot 权重 staging / remap 本身
  对 offload 层（2,3）产生了与全驻留 baseline 不同的 MoE 输出。

### 对 staging hook 的影响：必要但不充分
- 若按原计划落地 Regime A staging hook，captured 将变成 == eager == **仍然错**。
- hook 解决的是"capture 路径不读正确映射"，但**解决不了 eager 本身就错**这个更深的缺陷。
- **结论：冻结 hook 实现。** 优先根因化 eager-SEW 为何偏离 BASE。

### 下一步候选（根因化 eager-SEW 分叉，按代价排序）
1. **判定是数值还是逻辑**：在首个分叉 decode step（生成 index 3）抓 BASE 的 top-2 logit 间距。
   - 若 BASE top-2 极接近 ⟹ slot-packed grouped matmul 的 reduction order 数值差翻转了边界 token（可能"可接受"）。
   - 若 top-1 明显领先却被 SEW 翻掉 ⟹ 逻辑错（remap 索引 / slot 权重 layout / load_sync 搬运）。
2. **隔离 remap vs 权重**：固定 layers 2,3 为 resident（RESIDENT 含全 48 层）跑 SEW，应 == BASE，
   确认非 offload 层无回归；再仅 offload 单层（如只去掉层 2）二分定位。
3. **核对 slot 权重正确性**：比对 `slot_bank.w13_slots[slot_of(e)]` 与原始 `w13_weight[e]` 是否逐元素相等
   （staging 搬运 + layout 是否无损）。
4. **核对 remap 数学**：`remap_topk_ids = logical_to_physical[topk_ids]` 是否把 logical expert 正确映射到其 slot。

## 2026-06-16 — CPU 测试：staging+remap 数学层面证明无损（排除 mis-mapping）

新增 `tests/ut/moe_offload/test_fixed_slot_staging_correctness.py`（CPU，3 个全过）：
1. `test_staged_slot_weights_are_elementwise_lossless`：`w13_slots[log2phy[e]]` 与原始
   `w13_weight[e]` **逐元素相等**（`load_sync` 的 `copy_` 无损）。w2 同。
2. `test_log2phy_is_a_permutation_when_all_experts_fit`：num_slots==num_experts 时 log2phy 是合法排列。
3. `test_remap_recovers_correct_expert_weights_via_gather`：`w13_slots[log2phy[topk_ids]]`
   == `w13_weight[topk_ids]`，即 remap+gather 精确还原全驻留布局下每 token 的专家权重。

### 结论（卡-free 推进）
- **排除"slot 装错专家 / remap 索引错 / 权重 layout 损坏"这一类逻辑 bug。** 数学层面 SEW slot 路径正确。
- 结合"前 3 token 与 BASE 精确一致"（粗逻辑错会在 prefill 第一遍就污染 token 0），
  **数值差成为领头假设**：slot-packed grouped matmul 的 reduction/accumulation order 与 resident kernel 不同。
- 仍存的另一可能：真实 `npu_grouped_matmul` 路径里 group_list / per-expert token count 是否在
  logical vs physical id 上保持一致（CPU 测试覆盖不到真实 kernel）。

### 决定性下一步（需卡）：首个分叉点 logit 间距
- 改 probe 输出 logprobs，跑 BASE + eager-SEW，量生成 index 3 的 top-2 logit 间距。
  - top-2 极接近 → 数值差翻转临界 argmax（论文可表述为"可接受数值误差"）。
  - top-1 明显领先却被翻 → 逻辑/pipeline 一致性 bug，需进 fused_moe 真实路径二分。
- 用 eager-SEW 作载体（已证 eager==captured，且 eager 加载更快 ~86s，无 capture 复杂度）。

## 2026-06-16 — 决定性结论：SEW 分叉是数值 tiebreak，非逻辑 bug（logprob 实测）

改 probe 加 `--logprobs 20`，BASE + eager-SEW 同 prompt/seed=0 跑通，比对首个分叉点 pos=3：

```
            pos=2 (chosen 279, 决定性)          pos=3 (分叉点)
BASE:       279 @ -0.68338 (r1)  862 @ -1.308   22146 @ -1.44021 (r1,选中)  1376 @ -1.56521 (r2)
EAGER-SEW:  279 @ -0.68502 (r1)  862 @ -1.310   1376  @ -1.52263 (r1,选中)  22146 @ -1.52263 (r2)
            └ 一致, |Δ|≈0.0016, margin 0.625 ┘   └ 两候选近简并; SEW 把绝对 logprob 挪 ~0.08 nat 即翻序 ┘
```

### 判定：数值差（numerical），非逻辑/correctness bug
- pos=3 的两个候选 `' advantages'(22146)` / `' key'(1376)` 在 BASE 与 SEW 中都仅相差 ~0.04–0.13 nat（近简并）。
- SEW slot-packed grouped matmul 的 reduction order 使绝对 logprob 偏移 **~0.08 nat**，恰好翻转这对近简并 token 的 argmax。
- pos=2 有 0.625 nat 决定性 margin，BASE/SEW logprob 一致到 ~0.0016 nat，**不翻**。
- "决定性位置一致、仅近简并位翻转" = 教科书级数值扰动签名。

### 结论链（本阶段闭环）
1. CPU 测试：staging+remap 数学无损（排除 mis-mapping 逻辑 bug）。
2. logprob：eager-SEW 与全驻留 baseline 数值等价（差 ~0.08 nat），分叉是 greedy tiebreak 假象。
⟹ **SEW fixed-slot offload 不损害模型质量**；token-exact 匹配 no-offload baseline **从来不是正确的正确性判据**。

### 对论文正确性判据的修正
- 不要用 "token 序列逐位相等" 当 correctness 指标——greedy 在近简并位必然偶发翻转。
- 正确判据：①决定性位置（top-2 margin > 阈值，如 0.3 nat）top-1 一致率应 100%；
  ②整体 logprob 偏差 / 每 token KL 散度应 < 小阈值（实测 ~0.08 nat 量级）；③困惑度 (PPL) 对齐。

### 仍存的张力（需单列调查，不影响上面数值结论）
- 先前推断 captured 路径读持久 `-1` buffer → 应路由 slot[-1] 出乱码。但实测 **G(captured)==H(eager)** 逐 token 相等且连贯。
- 若 `-1` buffer 理论成立，captured 不该与正确的 eager 路由结果一致。⟹ 要么 captured 实际未按我追踪的方式读 stale buffer，
  要么 pre-capture eager warmup（FANOUT=128 强制 SLOT_CACHE_PATH）已把 slot/buffer 置于可用态。**此张力需独立排查**（下一阶段）。

## 2026-06-16 续 — 重大修正：captured 路径真读 -1 buffer，2 层被近简并掩蔽（NPU 探针实证）

### 方法
给 `_maybe_apply_moe_offload_plan` 两分支加 env 门控探针（`SEW_OFFLOAD_PROBE`，默认惰性）。
NPU5 graph 模式跑 G 同配置（NUM_SLOTS=128、non-resident={2,3}、GRAPH_COMPATIBLE=1）。
第一次探针含 `.item()` → 在 CAPTURE_SAFE 分支触发 107027 "Not allow to synchronize captured-stream"
→ **副产实证：CAPTURE_SAFE 分支确在真捕获期执行**。改 sync-free 后跑通。

### 探针序列（决定性）
```
warmup  : EAGER layer=2/3 capturing=False n_active=8     (eager 预热, 填 slot)
capture : CAPTURE_SAFE layer=2/3 buf_numel=128           (捕获期 wire 持久 -1 buffer)
prefill : EAGER layer=2/3 capturing=False n_active=51/45 (batch>1 eager)
decode  : —— 零探针行 ——                                  (batch=1 纯 replay 捕获图)
```

### 三条硬事实
1. CAPTURE_SAFE 在真捕获期触发并 wire 持久 `-1` buffer（107027 崩溃 + 95/96 行双重证明）。
2. decode 步**零探针行 → 纯 replay 读 `-1` 的捕获图**。captured 路径确实 mis-route 非驻留层 {2,3}。
3. 本次输出 `[3555,525,279,22146,323,63625,315,20980]` 与 BASE 一致到 **pos6**，仅 pos7 翻；
   而原 G 在 **pos3** 翻。**同配置不同翻转位 = run-to-run 非确定性。**

### 修正结论（推翻上一会话"-1 buffer 无害"的记录）
- captured 图**真读 stale `-1` buffer**、真 mis-route layers{2,3}。但 48 层中仅 2 层被污染，
  扰动落在 greedy **近简并带**内 → 输出看似 ≈BASE、在噪声 tip 的近简并位翻转（一次 pos3、一次 pos7）。
  **这是"掩蔽 (masking)"，不是"正确 (correctness)"。**
- 上一会话观察到的 `G==H==[1376...]` 是**巧合的同位翻转**，被我误读为"-1 buffer 无害"。**误判已纠正。**
- **eager 路径 (H, --enforce-eager) 不同**：每步走 EAGER 分支 → fresh **正确** log2phy → 与 BASE 真数值等价；
  其近简并翻转才是真正的数值等价证据。
- ⟹ **eager-SEW 正确性成立；captured-SEW 正确性依赖 staging hook。**
  FROZEN 的 hook 设计稿（docs/sew-offload/12）前提 **被坐实，而非推翻** —— 应 unfreeze。

### 待验证（下一步）
- offload 层数 scaling 实验：若 captured `-1` 是真 mis-route，则非驻留层↑ → 偏离 BASE 单调↑（数值发散），
  可作 (A)正确 vs (B)掩蔽缺陷 的决定性判据 + 论文图。注意显存：每 offload 层在 slot bank 复制整套 expert
  (~1.19 GiB/层)，56.9 GiB base + N×1.19，单 64 GiB 卡需控 N（之前选 2 层正为此）。

## 2026-06-16 续2 — offload-层数 scaling 实验：捕获 -1 mis-route 决定性坐实

### 设计
单卡 NPU5 顺序链跑 6 配置（脚本 `tools/sew_offload/run_offload_scaling.sh`，--logprobs 20，同 prompt/seed/max_tokens=8）：
BASE(SEW off) · captured N∈{1,2,4,6}（嵌套非驻留 {2}/{2,3}/{2,3,4,5}/{2..7}）· eager-SEW N=4 对照。
判据从"token 逐位相等"升级为"决定性位 top-1 是否翻 + logprob 重排幅度"。

### 结果：发散随非驻留层数单调增大（假说 A 成立，B 推翻）
| 配置 | 路由 | 输出 | 首个分叉位 |
|---|---|---|---|
| BASE | — | 3555,525,279,22146,323,63625,315,1667 | — |
| cap_N1 {2} | captured -1 | …,315,1741 | pos7（近简并）|
| cap_N2 {2,3} | captured -1 | …,315,20980 | pos7（近简并）|
| cap_N4 {2,3,4,5} | captured -1 | 3555,525,**862**,279,22146,315,279,323 | **pos2（决定性）** |
| cap_N6 {2..7} | captured -1 | 3555,525,279,1376,6813,315,**3555,525** | pos3 + **退化重复** |
| eager_N4 {2,3,4,5} | **正确** | 3555,525,279,**1376**,6813,315,1741,4119 | pos3（近简并）|

### 决定性 A/B（同 N=4，两条路由路径）—— pos2 logprob 量化
- BASE   : 279 @ -0.68338 (r1)，862 @ -1.30838 (r2)，margin 0.625 nat，279 决定性胜。
- eager_N4: 279 @ -0.68502 (r1)，862 @ -1.31002 (r2)，与 BASE 一致到 **~0.002 nat**，279 决定性胜。✓ 数值等价
- cap_N4 : **862 @ -0.90495 (r1，选中)**，279 **被贬到 r2 @ -1.27995**。279 跌 ~0.6 nat + 862 升 ~0.4 nat
  = **~1 nat 决定性位重排**（约 12× 数值噪声基底 ~0.08 nat），整张分布重洗（20980 从底部跳到 r7）。✗ 路由被污染

### 结论（坐实修正裁决）
1. captured `-1` buffer 造成**真实 mis-route**：非驻留层数↑ → 发散单调↑（pos7→pos2→退化重复），决定性位被 ~1 nat 翻转。
2. **同 N=4、eager 路由**保持 pos2 决定性、与 BASE 一致到 0.002 nat → 缺陷**专属于捕获路径**，非"offload 本身扰动"。
3. eager-SEW 数值等价成立；captured-SEW 正确性**必须**靠 staging hook（写持久 log2phy buffer）。
   ⟹ FROZEN 的 hook 设计稿（docs/sew-offload/12）前提**完全坐实，应 UNFREEZE 并作为 M2 收口**。

### 论文价值
- 这组 N-scaling + eager/captured A/B 是"control-plane/data-plane 解耦正确性"章节的核心实证图：
  x=非驻留层数，y=决定性位 logprob 偏移 / 首分叉位；captured 单调劣化 vs eager 平坦。
- 同时给出"为何 capture-pass ≠ token-correct"的量化边界：未装 staging hook 时捕获图静默 mis-route，
  仅在少数层时被 greedy 近简并掩蔽，层数一多即暴露。

## 续3（2026-06-16）：staging hook 实现的代码事实与设计取舍

- 核实 host_store 独立性：`host_store.py:88-89` `w13_weight[e].detach().cpu().clone()`
  —— host store 在 register_layer 时即建独立 CPU 副本，故 staging（host_store.get →
  transfer_engine.load_sync）**不依赖 NPU 原始权重**，可在 release_original 之后存活。
  ⟹ 先 stage 后 release 顺序安全，且正确性与顺序无关。
- 核实 num_logical_experts 来源：`log2phy_buffer(layer_id).numel()` == register 时
  `w13_weight.shape[0]`（runtime.py:308），无需再读私有权重 shape，封装内自洽。
- 设计取舍：Regime A hook 不放 model_runner（capture_model 前置 pass，原推荐 (b)），
  改放 SEW 自有 fused_moe.py 注册点。理由：(1) 不触 model_runner → 不触发架构评审；
  (2) 注册点已是 load 后、capture 前的天然锚点；(3) lazy-forward 注册分支也需兜底
  staging，统一封装为捕获期安全 no-op 后两路均可无条件调用。
- 捕获期安全 no-op 的必要性：lazy 注册分支在 forward 内执行，可能处于 capture；
  `stage_fixed_slot_plan` 捕获期会抛错，故封装层先查 `_is_current_graph_capturing()`
  返回 False（canonical 流已在 load-time eager 完成 staging）。
- fail-closed 保留：num_slots < n 时底层 working-set 守卫（runtime.py:519）抛
  "exceeds num_slots"，封装不吞——已加单测 test_full_residency_hook_fail_closed。

## 续4（2026-06-16）：Regime B 路径① ACLGraph 切分粒度勘验(纯读代码)

**问题**:能否让每个 MoE 层成为独立捕获单元,router 与 grouped MLP 之间留 eager 缝隙做
host-side router 预跑 + stage？

**确认成立的机制事实**:
1. PIECEWISE 已是"每层约一个子图"(acl_graph.py:147-150 注释)。每 piece 独立 NPUGraph
   + 独立 ACLGraphWrapper。replay 时逐 piece `entry.aclgraph.replay()`。
2. 切分点 = splitting_ops,在 FX 图层面切(backends.py:548 split_graph + should_split)。
3. **splitting-op 子图 eager 运行、不被捕获**(backends.py:1203-1207:
   `submod_names_to_compile = [... if not item.is_splitting_graph]`)。只有非-splitting
   子图被编译/捕获。这正是 attention 机制:attn op 夹在 piece 间 eager 跑(GraphParams/
   ExternalEvent 每步更新参数),前后非-attn 算子被捕获。
4. 默认 splitting_ops = _attention_ops(compilation.py:739)+ kv_cache_update,**无任何
   MoE/all2all 算子** → 整段 MoE(gate→select_experts→grouped MLP→combine)在同一 piece
   内,一起读 -1 buffer。这就是 Regime A 缺陷的图结构根因。
5. custom op 模板(mla.py:177-211):void func + mutates_args=["output"] 原地写 +
   get_forward_context().no_compile_layers[layer_name] 间接拿层 + dispatch_key=PrivateUse1
   + fake_impl。fx 不会重排/消除 mutates_args 的 custom op。
6. MoE 块算子是被 trace 展开进 fx 图的(故能被捕获),非整体 custom op → 可在 select_experts
   之后插入一个 split 点。

**路径① 设计骨架(成立,待落地验证)**:
- 注册 `vllm::moe_offload_stage(topk_ids, layer_name) -> topk_ids`(或 void+mutates log2phy),
  内部:D2H 读 active experts → stage_fixed_slot_plan(本步 active set)→ 写持久 log2phy。
- 在 fused_moe.apply 里把 select_experts 输出经此 op 再喂 grouped MLP(图兼容分支时)。
- 把 "vllm::moe_offload_stage" 加入 splitting_ops(platform.py:436 旁 extend)。
- 效果:piece A(router)→ eager stage op(捕获外,D2H+H2D 合法)→ piece B(grouped MLP
  读已更新 slot+log2phy)。打破"要 stage 必先 replay"环:本层 router 在 piece A 已 replay
  出 topk_ids,stage op 在层内、MLP 之前 eager 消费它。**层间数据依赖不再是障碍**——
  每层在自己的 piece 边界内解决,不需要预跑全部层。

**仍需验证/风险(诚实标注)**:
- (R1) 在 MoE 块中间插 split 点需改 fused_moe.apply 结构(图兼容分支),要确保不破坏
  现有 eager/全驻留路径(默认关闭开关)。侵入 SEW 自有文件,不碰 model_runner。
- (R2) D2H(topk_ids,小)+ H2D(miss experts,大)每 offload 层每 step 一次 = T_stall 来源。
  正确性成立,性能由 deadline prefetch/phase_split 优化——这才是论文性能章节。
- (R3) torch.compile 对"输出 feed 给后续 piece 的 custom op"的 partition 行为需实测
  (理论同 attention,但 attention 输出走 output buffer,moe_offload_stage 输出 topk_ids
  作为下一 piece 输入,数据依赖方向需确认 fx 正确串接)。
- (R4) num_slots<n 时 working-set 超 slot 要 fail-closed 或淘汰(slot_bank 已有 LRU 雏形)。

**结论**:路径①机制层面成立,是 Regime B 最贴合"控制面/数据面解耦"叙事的方案。下一步可做
最小 prototype(单层、num_slots<n)验证 R3 partition 行为,再扩全层 + 接淘汰策略。

## 续5（2026-06-17）：Regime B 路径① R3 实测 = NEGATIVE(根因坐实)

**结论:naive "seam op 放在 apply() 内" 不工作。根因已由探针数据坐实,非猜测。**

NPU1 双探针(SEW_SEAM_PROBE + SEW_OFFLOAD_PROBE),slots=128,nonres={2,3,4,5}:
| 阶段        | SEW_SEAM EAGER_STAGED | SEW_PROBE        |
|-------------|----------------------|------------------|
| warmup      | 4                    | EAGER            |
| capture     | 4(CAPTURING no-op 也各1) | CAPTURE_SAFE ×4 |
| prefill     | 4 (count9-12,n=51/45/43/39) | EAGER     |
| **decode×8**| **0**                | **0**            |

→ decode 8 步**MoE body 内零 Python 执行**。tokens 错(mis-route 签名)。

**根因**:moe_forward 经 direct_register_custom_op 注册(/root/vllm-hust 
vllm/model_executor/layers/fused_moe/runner/moe_runner.py:154)。torch.compile 视整个
MoE 区为**单个不透明 FX 节点**,不 trace 其 body。我的 torch.ops.vllm.moe_offload_stage
调用在 apply() 体内 = moe_forward 节点内部,**对顶层 split_graph 不可见**。放进
splitting_ops 匹配不到任何顶层节点 → **inert**。整个 MoE(router+MLP+seam)被整体捕获进
两 attention 间的 piece。
- prefill 不走捕获图 → body 全跑 → seam staging 生效(故 count9-12)。
- decode 走捕获图 → body 是冻结 kernel → apply() Python(seam + eager staging)全不执行
  → log2phy 冻结在 prefill 末值 → decode routing 命中 -1/错配区 → 错 token。

**路径① 机制本身仍成立**(splitting op 在 piece 间 eager 跑,attention 即如此),
但**seam 必须是顶层 FX 图的节点**,而 moe_forward 把它藏住了。

**正确修法(三选一,均超出 SEW 自有文件、触核心路径需评审)**:
- (A) 把 staging 提到 model 层(Qwen3MoeSparseMoeBlock.forward,gate 后/experts 前):
  但 top-k select_experts 现在在 moe_forward 内,model 层只算 router_logits。要在顶层拿
  active experts 需把 select_experts 移出 → **改路由路径,约束禁止**。
- (B) 拆分 moe_forward 为 moe_router(custom op,捕获)| moe_offload_stage(splitting,eager)
  | moe_mlp(custom op,捕获)三段 → seam 成为真 FX 切点。**vllm 核心 layer.py/moe_runner.py
  改动,需架构评审**。架构最贴论文叙事(MoE compute 仍在捕获区内,仅 staging 在区外)。
- (C) 把 "vllm::moe_forward" 整体加入 splitting_ops → 整个 MoE 区 eager 跑,apply() 现有
  eager offload path 每 decode 步执行。简单且配置门控,但 splitting_ops 全局 → **所有** MoE
  层(含 resident)失去捕获 = 接近 enforce_eager baseline,**非论文目标**(目标是捕获 MoE 
  同时 offload)。

**R3-b(slots=16)旁证**:fail-closed "working set 51 exceeds num_slots=16"。prefill 512
token 的 expert 并集 51 ≫ decode top_k=8 ≫ 16 → fixed-slot Regime B 需 eager-prefill 回退或
淘汰(R4),guard 正确未静默 corrupt。

**留存**:seam 接线全部 default-off(VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=0),不影响现有
Regime A / eager 路径。探针 env-gated 默认 inert。作为"what we tried"记录保留,待 (B) 评审。
