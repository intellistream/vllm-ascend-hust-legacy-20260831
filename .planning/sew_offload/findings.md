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
