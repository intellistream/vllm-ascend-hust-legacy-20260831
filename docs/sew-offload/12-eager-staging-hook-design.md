# Eager Pre-Replay Staging Hook 设计稿（M2 真边界 / 评审门控）

> 状态：**✅ 解冻（UNFROZEN）— 前提经 NPU 实测决定性坐实，进入实现。**
>
> 2026-06-16 NPU 探针 + offload-层数 scaling 实验决定性坐实本文档前提：
> 1. **探针实证**：env 门控探针证明 `CAPTURE_SAFE` 分支在真捕获期 wire 持久 `-1`
>    buffer；decode 步零探针行 → 纯 replay 读 `-1` 的捕获图。captured 路径**确实
>    mis-route** 非驻留层。
> 2. **scaling 实证**：非驻留层数 N↑ → 发散单调↑（N=1/2 仅 pos7 近简并；N=4 翻
>    **决定性位 pos2**，279→862 约 1 nat 重排；N=6 退化重复）。同 N=4 的 **eager
>    对照**保持 pos2 与 BASE 一致到 0.002 nat → 缺陷**专属捕获路径**，非 offload 本身。
>
> **上一会话"H==G ⟹ -1 buffer 无害、hook 仅必要不充分"的记录是误读**（巧合的近简并
> 同位翻转）。已纠正：`-1` buffer 是 captured 路径真实缺陷，**staging hook 是充要修复**。
> eager 路径每步重算正确 log2phy，本就数值等价；captured 路径必须靠本 hook 在 replay
> 前把正确 log2phy 写入持久 buffer。详见 `.planning/sew_offload/findings.md`
> "offload-层数 scaling 实验"一节。

## 1. 核实结论（问题陈述）

graph-compatible offload（Option 2）当前能让 ACLGraph **捕获通过**（G-run 验证：
SEW-only flag1 capture PASS + generate 跑通），但**捕获图静默 mis-route 非驻留层**
（读持久 `-1` buffer）。少数非驻留层时被 greedy 近简并掩蔽（输出看似 ≈BASE），
层数一多即在决定性位暴露：

```
BASE (no offload):           [3555, 525, 279, 22146, 323, 63625, 315, 1667]
captured N=4 {2,3,4,5}:      [3555, 525, 862, 279,   22146,315,  279,  323]
                             └5┘ └ pos2 决定性位翻转 (279→862, ~1 nat) ─────┘
eager   N=4 {2,3,4,5}:       [3555, 525, 279, 1376,  6813, 315,  1741, 4119]
                             └─ pos2 保持正确; 仅 pos3 近简并翻 (数值等价) ─┘
```

即 **capture-pass ≠ token-correct**：未装 staging hook 时捕获图按 `-1` 路由。本文档
定位根因并提出修复设计。


## 2. 证据链（根因，已在代码层面 airtight）

持久 log2phy buffer 的"写 / 读"生命周期：

| 角色 | 位置 | live 调用点 |
|------|------|-----------|
| 分配 + 初始化为全 `-1` | `runtime.py:309`（`register_layer_for_fixed_slots`） | 有（注册时） |
| **唯一写入者** `stage_fixed_slot_plan` → `buf.copy_(...)` | `runtime.py:592-593` | **0（仅 tests/ut + 1 条注释）** |
| 捕获期读取者 `capture_safe_slot_weights` → `log2phy=mapping.logical_to_physical` | `runtime.py:602` → `slot_mapping.py:145` | 有（`moe_comm_method.py:341`） |
| 捕获图内 gather `topk_ids = log2phy[topk_ids]` | `moe_comm_method.py:149 / 529` | 有（录进图，replay 复用 buf 内容） |

**结论**：捕获路径**读** persistent buf；唯一**写**者在生产路径**无调用点**。
故 offload 层（2,3）的 buf 恒为初值 `-1`，每次 replay 的 `log2phy[topk_ids]` 取到
`-1` → slot 索引错误 → 路由错误 → 输出分叉。

> 这是一个**缺失的接线（missing wire）**，不是 SEW 控制面原语的逻辑 bug。eager
> 路径（`_is_current_graph_capturing()=False`，`moe_comm_method.py:346+`）用
> `torch.unique().cpu()` + slot_mapping 生成 **fresh 正确 log2phy**，不读持久 buf，
> 故预期 token 正确——由 `H_sew_eager_flag1` 对照实测确认。

## 3. 核心难点：控制面/数据面的时序冲突

修复不是"加一行调用"那么简单，因为存在一个**数据依赖时序冲突**：

- `stage_fixed_slot_plan(layer_id, active_experts, ...)` 需要本层的 **active_experts**
  才能决定 slot 分配 + H2D 搬运 + 写 log2phy。
- 但 active_experts 来自本层 router 输出（top-k gating），**该计算在捕获图内部**。
- staging 必须在 **replay 之前**（图外、eager）执行，否则又会把 host-sync
  （`torch.unique().cpu()` + `load_sync`）带回捕获流，重新触发 107027/107030。

即：**要 stage 必先知道 active_experts，要知道 active_experts 必先 replay，
要 replay 必先 stage** —— 环形依赖。这正是 graph-compatible offload 的根本张力，
也是论文"控制面/数据面解耦"贡献要回答的核心问题。

## 4. 方案分层（按 num_slots 与 working-set 关系拆两个 regime）

### Regime A：`num_slots >= num_logical_experts`（全装得下，无淘汰）

此时**每个 logical expert 都有固定 slot，log2phy 是静态映射**，与某一步的
active_experts 无关。环形依赖被打破：可在**模型加载后、首次 replay 前一次性**
为每个 offload 层调用 staging（`active_experts = 全部 expert`），把全部权重从
host_store 搬进 slot 并写定 log2phy buffer，此后 buffer 内容不再变化。

- 这正是 `H_sew_eager_flag1` 对照所处的配置（num_slots=128 = expert 数）。
- **token 正确性**：log2phy 写入真实 logical→physical 映射，捕获 gather 读到正确
  slot 索引 → 与 BASE 一致。
- **代价**：等价于"全专家驻留到 slot"，不省 HBM（slot bank ≈ 原权重大小）。它**不是
  最终 offload 目标**，但它是**证明 staging hook 正确性的最小闭环**，且本身即一个
  合法的中间产物（固定地址 + 图兼容的全驻留）。

> **推荐先落地 Regime A**：改动最小、token 正确性可由对照实测背书、不触及调度逻辑。

### Regime B：`num_slots < num_logical_experts`（真 offload，有淘汰）

log2phy 随每步 active working-set 变化，环形依赖真实存在。三条候选路径（按侵入性排序）：

1. **Host-side router 预跑（推荐研究方向）**：在 replay 前用一个**轻量 host/eager
   前置 pass** 跑到各 offload 层的 router，拿到本步 active_experts，再逐层
   `stage_fixed_slot_plan`，最后 replay 主图。代价是 router 重复计算（但 router 相对
   MLP 极廉价）。需要把 router 从被捕获的 MLP 段中可单独 eager 调用地暴露出来。
2. **Phase-split 预测（用上一步 routing 作预测器）**：以 step N-1 的 active_experts
   预 stage step N，miss 的专家在 Phase 1 补搬。命中率依赖 routing 时间局部性，**会引入
   token 误差除非 miss 时 fail-closed 回退 eager**。与 `phase_split.py` 现有原型对接。
3. **两遍 forward**：第一遍 eager 只为收集 routing，第二遍 replay。最简单但最慢，
   仅作正确性参照，不作生产路径。

> Regime B 不在本次落地范围；列出供评审讨论论文主线走向。先用 Regime A 锁定
> 正确性闭环，再在论文中把 Regime B 作为"控制面/数据面解耦"的核心贡献展开。

## 5. Regime A 接线点设计（已落地）

> **实现更新（2026-06-16）**：实际落地点优于原 (b) 方案。挂点选在 **SEW 自有
> 的 MoE 前端集成文件** `fused_moe.py` 的 fixed-slot **注册点之后**（非 model_runner），
> 因此**完全不触碰 model_runner**，规避了架构评审门槛，且与 §5.3 不变量全部相容。
> CPU 侧单测（11+50 例）全绿。下面 5.1 记录原候选讨论，5.2/5.3 为实际接口与不变量。

### 5.1 插入位置（评审重点）

候选时机：**模型权重加载完成、ACLGraph 首次 capture 之前**，逐 offload 层调用一次
staging。`process_weights_after_loading`（`fused_moe.py:134`）已在加载后注册 fixed
slot（`fused_moe.py:165`），是天然的相邻锚点；但**实际 H2D 搬运 + 写 buffer 应在
slot bank/host_store 就绪后、capture 之前**触发。两个可选挂点：

- **(a) 紧跟注册之后**（在 `process_weights_after_loading` 内，注册成功的同一层）：
  内聚、改动局部，但把 H2D 搬运耦合进权重加载阶段。
- **(b) capture_model 之前的独立 pass**（`model_runner_v1.py:3890 capture_model` 入口前）：
  与"捕获前置准备"语义对齐，集中可控，便于评审一处审计所有 offload 层。**推荐 (b)**。

> 任一挂点都**不改 router / top-k / gate / combine 语义**，只在图外把权重搬进既有
> slot、把真实映射写进既有 buffer——符合 SEW "只改权重驻留与访问方式" 的约束。

### 5.2 接口（不新增 runtime 公共原语，复用现有）

> **已实现**：在 `runtime.py` 新增**薄封装** `stage_full_residency_slot_plan(layer_id)`
> —— Regime A 一次性全专家 staging。它做四道门控（`should_use_fixed_slots ∧
> graph_compatible_offload`、非 resident、已注册、非捕获期）后，从持久 buffer 的
> `numel()` 取 `num_logical_experts`，对全专家调用既有 `stage_fixed_slot_plan`。
> 捕获期是**安全 no-op**（返回 False，不抛错），故可被 load-time 与 lazy-forward
> 两条注册路径无条件调用。`num_slots < n` 时由底层 working-set 守卫 fail-closed 抛错。
>
> 接线点（两处，均紧跟 `register_layer_for_fixed_slots`、在 release 之前）：
> - `fused_moe.py` `process_weights_after_loading`（load-time 主路径）
> - `fused_moe.py` `apply` 内 lazy 注册分支（eager warmup 首触兜底）
>
> 从 host_store 的**独立 CPU 副本**（`host_store.py:88-89` `.detach().cpu().clone()`）
> staging，故先 stage 后 release 顺序安全、且正确性与顺序无关。

```python
# runtime.py 实际签名（薄封装，零新增公共 staging 原语）
def stage_full_residency_slot_plan(self, *, layer_id: int) -> bool: ...

# fused_moe.py 接线（注册之后、release 之前）
if moe_offload_runtime.should_use_fixed_slot_plan_for_layer(layer_id):
    moe_offload_runtime.register_layer_for_fixed_slots(layer, slot_device=...)
    moe_offload_runtime.stage_full_residency_slot_plan(layer_id=layer_id)  # Regime A
    if moe_offload_runtime.config.release_original_expert_weights:
        moe_offload_runtime.release_original_expert_weights_if_ready(layer)
```

- **零新增 runtime 决策原语**：`stage_fixed_slot_plan` / `is_resident_layer` /
  `is_layer_registered` / `log2phy_buffer` 均已存在；新封装仅"接上调用 + 门控"。
- Regime A 要求 `num_slots >= n`，否则 `prepare_fixed_slot_plan` 在
  `runtime.py:519` 抛 working-set 超限——既有保护，符合 fail-closed（已加单测覆盖）。

### 5.3 不变量（评审 checklist）

1. 默认关闭：仅当 `enabled ∧ num_slots>0 ∧ graph_compatible_offload` 才触发。
2. staging 全部在 **eager**（capture 前）完成；`stage_fixed_slot_plan` 内已有
   `_is_current_graph_capturing()` 断言（`runtime.py:579`）防止误入捕获流。
3. resident 层一律不 stage（`prepare_fixed_slot_plan` 在 `runtime.py:508` 已 fail-closed）。
4. 不触碰 scheduler / token dispatcher 主路径；不改 router 语义。
5. 持久 buffer 地址在 staging 前后不变（`buf.copy_` 原地写，`runtime.py:593`）。

## 6. 验证计划

| 步骤 | 内容 | 通过判据 | 状态 |
|------|------|---------|------|
| V0（前置） | eager-SEW 对照 `eager_N4_control`（--enforce-eager + 同 SEW env） | tokens 决定性位 == BASE → eager staging 逻辑被实测背书 | ✅ 已过（0.002 nat 一致） |
| V3a | CPU 单测 `test_graph_compatible_offload.py` 新增 Regime A hook 用例 | 全绿（fill / off / capturing / unregistered / fail-closed） | ✅ 11 例通过 |
| V3b | 邻近 fixed-slot + moe_comm_method 套件无回归 | 全绿 | ✅ 50 例通过 |
| V1 | NPU captured SEW（Regime A，num_slots=128，graph_compat=1，hook 已接） | tokens == BASE（pos2 决定性位修复，不再 279→862） | ✅ 已过（见下） |
| V2 | HBM/内存账本核对 `memory_ledger()` | offload 层 slot_bank 已填、host_store 已建、log2phy 非 `-1` | ✅ 已过（见下） |
| V4 | resident 全量路径无回归（默认关闭开关下 BASE 不变） | tokens == BASE | ⬜ 待 NPU 资源 |

> **V2 实测通过（2026-06-16，NPU5，SEW_OFFLOAD_LEDGER 探针）**：cap_N4 配置每个
> offload 层 {2,3,4,5} `log2phy_staged=128/128`（无 `-1` 残留），slot_bank/host_store
> 逐层建好，OUTPUT_TOKENS 再次 == BASE。账本（num_slots=128）：每层每副本 **1.125 GiB**，
> `slot_bank_bytes == host_store_bytes == original_expert_weight_bytes`。⟹ Regime A 同层
> 在 NPU 同时持 original + slot 两副本（未 release 时 2.25 GiB/层），**不省 HBM 反增**；
> 全 48 层 offload 仅 slot_bank 即 54 GiB > 64GB 卡。**这正是 §4 Regime B（淘汰式真
> offload）的动机数据** —— Regime A 是正确性地基，不是 offload 终点。

> **V1 实测决定性通过（2026-06-16，NPU5，after-hook）**：scaling 全 6 配置重跑，
> before-hook 日志备份于 `.planning/sew_offload/logs/before_hook/`。cap_N1/N2/N4/N6
> **逐 token 完全等于 BASE**，before-hook 的单调发散全部塌平。决定性位 pos2：
> cap_N4 BEFORE `chosen=862`(mis-route) → AFTER `chosen=279`(修复)。**强于判据**：
> after cap_N4 pos2 logprob 逐位精确等于 BASE 到 1e-5（`279:-0.68338` 两侧一致）⟹
> captured 路径与全驻留**数值等价**，非近似。eager_N4_control pos2 仍 279 → 缺陷
> 专属捕获路径，hook 补的就是缺失接线。这组 before/after 是论文"控制面/数据面解耦
> 正确性"核心实证图。

> **门控顺序**：V0 必须先过（证明 eager staging 正确），才动 model_runner 实现 V1。
> **V0 已过（2026-06-16）**：scaling 实验的 `eager_N4_control`（--enforce-eager + 同
> SEW env，N=4 非驻留 {2,3,4,5}）pos2 与 BASE 一致到 0.002 nat → eager staging 逻辑
> 经实测背书。可进入 V1（model_runner 接挂点）——但 model_runner 改动需严格架构评审，
> 落地前须与用户对齐。

## 7. 与论文叙事的衔接

- Regime A 给出 **capture-pass → token-correct** 的最小闭环，证明 SEW 控制面原语
  在图兼容前提下可正确执行——即"控制面/数据面解耦"在**全驻留**下成立。
- Regime B（§4）的环形依赖求解（host-side router 预跑 / phase-split 预测）是论文
  **核心贡献**：在 HBM 不足、必须淘汰时，如何在不破坏 ACLGraph 捕获的前提下完成
  数据依赖的 staging。本设计稿把它与 Regime A 的正确性地基显式区分，便于评审。

