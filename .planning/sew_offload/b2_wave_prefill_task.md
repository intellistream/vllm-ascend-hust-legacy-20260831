# B2 — 容量分波 prefill (wave-streamed prefill, num_slots < prefill fanout)

目标: 让服务命令真实工作点(num_slots=8 ≪ prefill 并集~51)能在【图捕获】下跑 offload。
B2 分波【只发生在 eager prefill】, 永不碰捕获图结构 / 地址稳定契约。decode 仍走 B1 单波。

## 决定性设计支点 (已代码核实)
- 三段 seam (router|stage|mlp) prefill+decode 都走, 区别只在 ACLGraph 是否 replay。
- prefill 变长 shape → ACLGraph 不捕获 → prefill 永远 eager。大 fanout(51) 只在 eager 阶段。
- moe_mlp → runner._forward_impl → moe_comm_method.fused_experts (fused_moe.py:403),
  phase_split 执行骨架已接在此处 (moe_comm_method.py:181-199, 默认关) + prefill/decode 分类器(:168)。
- 缺口: ① 容量分波 planner ② 每波 staging 钩子。执行器切片/scatter 累加现成。

## 阶段 1 (当前, 纯 CPU 纯新增可单测)
- [x] phase_split.py 新增 plan_capacity_bounded_phases(expert_slices, active_expert_ids, num_slots)
      → ⌈N/num_slots⌉ 波, 每波 ≤num_slots, 产出 MoEPhasePlan (复用 MoEPhase, is_hit=False)。
      N≤num_slots 退化单波 (reason=capacity_single_wave)。
- [x] execute_phased_mlp 加可选 stage_wave_fn 回调 (每波 matmul 前 stage; 有钩子时跳过单波 fast-path)。
- [x] CPU 单测 41/41 绿: 分波结果逐元素==单波; 边界 (N≤slots, 整除, 余数, slots=1); stage 钩子按波触发。
      注: expert_indices 是 group_list/weights 的【位置】(0..N-1, 同 plan_hit_miss_phases 契约),
      stage 钩子收到的是位置; 阶段 2 需把位置→逻辑 expert_id 映射后再 stage 进 slot bank。

## 阶段 2 (接 prefill + NPU)
- [架构发现] B2 不能只在 MLP 执行器层做。offload prefill 有三处"单遍全专家"假设, 且 fail-closed
  在 MLP 之前:
  1. _maybe_apply_moe_offload_plan eager 分支调 prepare_fixed_slot_plan → runtime.py:538
     fanout>num_slots 直接 raise (51>8 在此崩, 到不了执行器)。
  2. token_dispatch 用 log2phy[topk_ids] 单遍重排; 51 逻辑专家映射不到 8 物理槽 = 无合法 log2phy。
  3. mlp weights 指向 slot bank (只 num_slots 行), 装不下 51 个给执行器切。
- [决策] 用户选: 独立 B2 prefill 路径 (绕过单遍 prepare/dispatch)。每波: stage≤num_slots 专家 +
  wave-local log2phy 局部 dispatch + partial matmul + combine 累加。decode/现有 prefill 全不变。
- [数学核心 keystone] token MoE 输出 = Σ_{e∈topk} gate·expert_e; 加法可结合 ⇒ 按专家分不相交波、
  每波掩码只算本波 (token,专家) 贡献、跨波累加 == 单遍全专家。先 CPU 证此等式再碰 live/NPU。
- [x] CPU 数值测试: 分波+掩码+累加 逐位==单遍 (含 top_k>1, 余数波[4,4,2], slots=1, 单波)。
      45/45 绿。keystone 成立 ⇒ 独立 B2 prefill 路径数学正确, 可放心接 live。
- [x] overlap-ready 接口重构 (边传边算预留, 实现仍串行):
      - WaveStager 两段式契约 issue(发起搬运)/wait(等就绪) + buffer_count; 串行基类。
      - _CallbackWaveStager 适配旧 stage_wave_fn (buffer_count=1, issue 同步/wait no-op, 语义不变)。
      - execute_phased_mlp 改软件流水循环: prefetch_depth(默认0=串行) + max_in_flight=
        min(depth+1, buffer_count) 守卫, 保证缓冲未被消费前不重发 (single-buffer 恒串行)。
      - CPU 测试: 串行==direct; depth=1/双缓冲 输出==串行; prefetch 确实提前 issue;
        depth=5 但 buffer=2 时在投运前 issue≤2 (容量上界); 互斥参数报错。phase_split 51/51 绿。
      - 全 moe_offload UT 247 passed/1 skipped 无回归。
      - 注: NPU async stager (独立 transfer stream + SetFlag/WaitFlag + double-buffer slot_bank)
        是 WaveStager 子类, drop-in, 不动执行器/planner。本阶段不实现, 先串行过 NPU 正确性。
- [ ] 接 live: 独立 prefill 执行路径 (gate: 仅 prefill∧eager∧fanout>num_slots)。【大改, 分小步】
  - [x] 小步1: 开关+gate+探针, 不改计算。
        - envs VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL (默认0); config.b2_wave_prefill。
        - runtime.should_use_b2_wave_prefill(layer_id, active_expert_count, is_prefill):
          config 开∧prefill∧offload 固定槽层∧fanout>num_slots 才 True; 纯 Python 无设备动作。
        - test_b2_wave_prefill_gate.py 7 例绿 (decode/resident/容量内/精确容量/默认关 全覆盖)。
        - phase_split+gate+seam 三套 105 passed 无回归。
  - [ ] 小步2: fused_experts 加 B2 分支 + 探针 (gate 命中打 marker, 仍走原 fail-closed; 不改计算)。
  - [x] 小步2 完成: moe_comm_method 在 active_experts 算出后插 B2 探针 (SEW_B2_PROBE/SEW_OFFLOAD_PROBE),
        gate 命中只打 "SEW_B2 branch=GATE_FIRES" marker, 路径全不变 (仍走原 fail-closed/slot)。
        is_prefill = hidden_states.shape[0]>1 (同行 168 分类器)。全 moe_offload UT 254 passed/1 skipped。
  - [ ] 小步3: 独立分波 prefill 执行体 (每波 mask→局部 dispatch→stage→matmul→combine→累加, 串行 WaveStager)。
    - [x] 小步3a: build_wave_expert_map(wave_logical_experts, n) → 本波专家映射到槽位 0..k-1、其余 -1。
          坐实 AllGather dispatcher 原生 drop 机制 (expert_map[topk_ids]==-1 → topk_weights*=0,
          token_dispatcher.py:360-361)。CPU 测试: builder 3 例 + 每波 expert_map 累加==全 map 单遍。
          phase_split 55/55 绿。⇒ B2 无需新 mask 方案, 复用 dispatcher drop。
    - [ ] 小步3b: fused_experts 接 B2 分支: 命中 gate → 分波循环 (每波换 expert_map + stage + 跑
          现有 dispatch/matmul/combine + 累加), 跳过 fail-closed。串行 WaveStager。
    - [x] 小步3a': build_b2_wave_routing(physical_topk_ids, topk_weights) → -1→slot0 + 该位权重置0。
          坐实 offload 路径(physical_expert_count 模式, expert_map=None, topk_ids 经 log2phy 预remap)
          不自动 zero, 故须显式置 0。CPU: routing 3 例 + live-style log2phy 累加==全单遍。59/59 绿。
    - [x] 小步3b: moe_comm_method live 接线 (默认关, 早分支零开销):
          - fused_experts 顶部加 _maybe_run_b2_wave_prefill 早分支 (offload∧b2_wave_prefill∧非捕获∧
            gate 命中 才进; 否则 return None 走原路径, 既有 262 passed/1 skipped 无回归)。
          - _run_b2_wave_prefill: active 并集切 ⌈N/num_slots⌉ 波; 每波 prepare_fixed_slot_plan(wave)
            stage≤num_slots 专家 + 拿 log2phy → _run_b2_single_wave → 累加 routed_out。
          - _run_b2_single_wave: log2phy[topk_ids]→physical, build_b2_wave_routing 掩码,
            指 slot 权重, expert_map=None/physical_expert_count, 跑现有 dispatch/matmul/combine。
          - 探针 SEW_B2 branch=WAVE_RUN (n_active/num_slots/n_waves)。
- [x] NPU 验证 (NPU5, eager 非 seam, num_slots=8 ≪ prefill 并集, nonres={2,3,4,5}):
      - B2 真触发: WAVE_RUN layer2 n_active=51→7 波, layer3 45→6, layer4 43→6, layer5 39→5
        (num_slots=8 远小于 51, B1 会 fail-closed; B2 分波正常跑)。
      - 输出 tokens 8/8 逐位==BASE [3555,525,279,1376,6813,315,1741,4119]; 全 8 位 top-1 一致。
      - 但 logprob【非】1e-5 位等: top 档 ~0.08-0.16 nat 漂移 (pos6 一个尾部 token 0.42)。
        原因(非 bug): captured-B1 跑【同一次】slot-packed grouped matmul → 逐位等; B2【重结合】
        一次 matmul 为 7 个 bf16 求和波, bf16 加法非结合 ⇒ 必然漂移。CPU keystone 已证 fp32 下
        算法精确到 1e-6; NPU 漂移纯属 bf16 reduction order。漂移量级 ~0.5-1% (logit~15-20),
        top-1 margin~1nat 远大于漂移 ⇒ 无 token 翻转。
      - 结论: B2 数值正确性 = "fp32 精确 + NPU top-1 全保持 + 漂移与 bf16 重结合一致", 而非
        "bit-identical"。这是 wave-streaming 把单次 reduction 拆成多波的【固有】代价, 非缺陷。
      - NPU5 跑后 "No process in device" 无泄漏。
- [ ] (可选) 用 fp32 或强制单波 B2 进一步隔离漂移; B2-with-seam (图捕获) 集成; autoconfig 接线。

## B2 + seam (图捕获) 集成 — 已完成 + NPU 验证 (NPU5)
- 改动: moe_offload_stage_op.py 在算出 active_experts 后加 B2 deferral: 当 should_use_b2_wave_prefill
  命中 (prefill∧b2 开∧fanout>num_slots) → seam NO-OP 不写 log2phy (reason=b2_wave_defer), 把 staging
  交给 moe_mlp→fused_experts 的 B2 波循环。decode (fanout≤slots) 仍走 seam 单波 staging, 捕获不受影响。
  capture guard: _maybe_run_b2_wave_prefill 在 torch.unique().cpu() 前先 _is_current_graph_capturing()
  →None, 故捕获期不触 D2H。UT seam+gate+phase_split 113 passed。
- NPU5 (GRAPH_COMPATIBLE=1, STAGE_SEAM=1, B2=1, num_slots=8, --no-enforce-eager):
  - 拓扑正确: seam defer 4× (prefill 层 2/3/4/5) + B2 WAVE_RUN 4× (51→7波/45→6/43→6/39→5) +
    CAPTURING=48 (decode 捕获图 replay)。
  - tokens [3555,525,279,1376,6813,315,1741,4119] == eager-BASE == eager-B2 (三个 eager-prefill
    家族全一致, 0 top-1 flips); vs eager-BASE max|Δlogprob|=0.30 nat (bf16 波重结合+captured decode)。
  - vs captured-BASE 分叉 (pos3 22146 vs 1376) = V-D 已记录的 eager-vs-captured 近简并 tiebreak,
    与 B2 无关 (B2+seam 的 prefill 是 eager 波, 故落 eager 家族选择)。
  - 结论: B2+seam 正确 = "与 eager-prefill 家族 token 全一致 + 拓扑正确(defer+wave+capture)";
    这是服务命令目标态 (小 num_slots 省 HBM + 图捕获 decode 提速 + B2 prefill 不 fail-close)。NPU5 无泄漏。
- 下一步: autoconfig 把 --ascend-moe-offload-gb 数据面切到 SEW (开 GRAPH_COMPATIBLE/STAGE_SEAM/B2,
  去 enforce_eager); 端到端跑; README。

## autoconfig SEW 数据面接线 — 已完成
- 新 env VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE (默认 0)。autoconfig.apply_moe_offload_defaults:
  on 时 setdefault GRAPH_COMPATIBLE=1/STAGE_SEAM=1/B2_WAVE_PREFILL=1 且【不】wire PrefetchOffloader
  (offload_backend 不设); off 时维持原 prefetch 路径。num_slots/resident 仍按 GB 预算推导(共用)。
  engine_args._ascend_moe_offload_sew_dataplane 标记。
- 测试: 2 新例 (sew on→seam/B2 armed + 无 offload_backend; 默认→prefetch + 无 seam)。
  autoconfig 18 passed; 全 moe_offload UT 264 passed/1 skipped 无回归。
- e2e 脚本 run_e2e_sew_service.sh: 只设 GB=14 + SEW_DATAPLANE=1, autoconfig 派生其余 (真服务路径)。
- [进行中] NPU e2e (GB=14 → ~12 层 offload, num_slots=8): tokens vs captured-BASE + HBM 占用。

## 端到端 SEW 服务路径 — 已验证 (NPU5)
- 配置: 只设 VLLM_ASCEND_MOE_OFFLOAD_GB=14 + SEW_DATAPLANE=1, --no-enforce-eager。autoconfig 派生
  num_slots=8、offload 层 {3,7,11,...,47}(12 层)、arm seam+B2、不 wire PrefetchOffloader。
- 结果:
  - HBM: 56.90 GB → 44.24 GB = 省 12.66 GB (22%)。账本吻合: 12 层×(128-8)专家×9MiB=12.66GiB。
    (对比 B1 只省 1.125GB — 这才是 B2 的真实 HBM 价值。)
  - 拓扑: WAVE_RUN=12 (12 层全跑 B2 波) + b2_defer=12 (seam 全 defer) + CAPTURING=48 (decode 捕获
    → B1 的 3× decode 提速继承)。WAVE_RUN 层集 = autoconfig gb=14 派生的 {3,7,...,47}。
  - tokens [3555,525,279,22146,323,63625,315,1667] == captured-BASE 逐位一致; 全 8 位 top-1 一致;
    max|Δlogprob|=0.244 nat (bf16 波重结合, 无翻转)。
  - 脚本 run_e2e_sew_service.sh。NPU5 无泄漏。
- ⟹ 服务命令现可在 SEW 数据面跑: 省 ~12.7GB HBM + 保图捕获 decode + B2 prefill 不 fail-close。
- [x] README (英/中) 更新端到端数据: B2 分波 + SEW_DATAPLANE 服务命令 + 56.90→44.24GB(省12.66/22%)
      表格 + 0.24nat 漂移说明 + 两段式 issue/wait overlap 预留。全部完成。

## 阶段 3 (autoconfig + README)
- [ ] autoconfig 开 GRAPH_COMPATIBLE/STAGE_SEAM、去 enforce_eager、数据面切 SEW。
- [ ] 修正 README.md / README.zh.md: 服务命令(原生 offload) vs B1/B2(SEW 图捕获) 明确分开。

## 关键正确性点
- 每波 "stage→partial matmul→scatter 累加" 严格串行: wave1 算完 scatter 后才被 wave2 覆盖 slot,
  故 log2phy 被覆写无害。num_slots 容量内每波合法。
- 不改 router/top-k/gate/combine; 不动主路径; env-gated 默认关。

## 约束
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python; ASCEND_RT_VISIBLE_DEVICES(非CUDA_);
空闲卡; 端口 8016 不碰; Model=/data/shared-models/Qwen3-30B-A3B。
