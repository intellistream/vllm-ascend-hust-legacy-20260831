# moe_forward 三段拆分设计稿（Option B / 图模式真边界 · 评审门控）

> 状态：**🟢 评审通过 — P1 进行中。** 6 个决策已拍板：②=B1(topk 注入短路)、
> ⑥=首版限单卡(identity-prepare)、①③④⑤=接受默认。P1 已落 `vllm::moe_router` op +
> 4 UT(含 native 路径逐位等价),20/20 绿;`AscendMoERunner.forward` 接线(主路径,
> default-off)为 P2。
>
> 本文档是 graph-mode skill「第二步：方案确认」的产物。R3（续13）已**数据坐实**：
> 把 seam op 放进 `fused_moe.apply()` 内部**不能**产生 FX 切点，因为整个 MoE 区域被
> `moe_forward` 这个 `direct_register_custom_op` 注册的**不透明 op** 包住，decode 步在
> MoE body 内执行**零 Python**（SEW_SEAM 计数停在 12=prefill；SEW_PROBE decode 0 行）。
> 本设计提出把 `moe_forward` 解构成 **router | offload_stage | mlp** 三段，使 staging 成为
> 真正的**顶层 FX 切点**，从而在 decode 每步以 eager 形式运行控制面、捕获图复用数据面。
>
> **所有改动默认关闭**（`VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=0`），关闭时
> `AscendMoERunner.forward` 回退到 `super().forward()`（原 monolithic `moe_forward`），
> 对 Regime A / eager 路径**零影响**。

---

## 1. 问题陈述（为什么必须拆 moe_forward）

SEW-Offload 的捕获兼容方案依赖一条不变式：

> **控制面（host 决策：哪些 expert 该就绪、写 log2phy 持久 buffer）必须在 ACLGraph
> 每次 replay 前以 eager 形式执行；数据面（gather + grouped matmul + combine）则录进
> 捕获图，每步复用固定地址。**

R3 探针证明当前实现无法满足该不变式：

| 阶段 | MoE body 是否执行 Python | 后果 |
|------|--------------------------|------|
| prefill（eager，未捕获） | 是（SEW_SEAM 计到 12） | staging 正常跑，token 正确 |
| decode（捕获/replay） | **否（0 行）** | staging **从不执行**，捕获图读旧 log2phy buffer |

根因（续5，代码层 airtight）：`moe_forward` 在
[`moe_runner.py:154`](../../../vllm-hust/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L154)
经 `direct_register_custom_op` 注册为**不透明 FX 节点**。torch.compile 不 trace 其
body，故 body 内任何 op（包括我放的 `vllm::moe_offload_stage`）都**不是顶层节点**，
列进 `splitting_ops` 匹配不到任何顶层节点 = 完全失效（inert）。

**结论**：要让 staging 成为真切点，它必须在 `moe_forward` **之外**作为顶层 op 出现。
但 staging 需要 `topk_ids`，而 `topk_ids` 由 `select_experts` 产出，后者目前在
`moe_forward` **内部**（`quant_method.apply` 里，[`fused_moe.py:690`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L690)）。
鸡生蛋 → 必须把 router 也提到顶层。最小可行解即**三段拆分**。

---

## 2. 目标拓扑（三段 = 最小切分）

```
            ┌─────────────── torch.compile 追踪的图（decoder layer fwd）───────────────┐
            │                                                                          │
 hidden ───►│  [vllm::moe_router]  ──topk_ids/topk_weights──►  vllm::moe_offload_stage  ──►  [vllm::moe_mlp]  ──► routed_out
            │   (captured piece 1)        (splitting op, EAGER between replays)            (captured piece 2)  │
            └──────────────────────────────────────────────────────────────────────────────────────────────┘
                       │                              │                                        │
              router_logits→select_experts    D2H unique(topk_ids)                    gather log2phy[topk_ids]
              纯计算/捕获安全/定形              stage_fixed_slot_plan                  grouped matmul + combine
                                              写持久 log2phy buffer                   读持久 buffer（捕获期固定地址）
```

- **`vllm::moe_router`**（不透明 custom op，**录进**捕获图）：
  `hidden_states (+ gate) → router_logits → select_experts → (topk_ids, topk_weights)`。
  纯计算、定形、无 host sync。**语义与现状逐位一致**——只是把 `select_experts` 的调用
  点从 `quant_method.apply` 内提前到这里，不改 logits / top-k / gate / renormalize 任一参数。

- **`vllm::moe_offload_stage`**（已存在于
  [`moe_offload_stage_op.py`](../../vllm_ascend/ops/fused_moe/moe_offload_stage_op.py)，
  列入 `splitting_ops`，**eager**）：
  `topk_ids → D2H unique → stage_fixed_slot_plan → 写持久 log2phy buffer → 返回 topk_ids.clone()`。
  `clone()` 强制下游数据依赖，保证它落在 piece1 与 piece2 之间、不被 DCE。
  **decode 每步以 eager 运行**——这正是 R3 缺的那一环。

- **`vllm::moe_mlp`**（不透明 custom op，**录进**捕获图）：
  `hidden_states + topk_ids + topk_weights → prepare/dispatch → gather log2phy[topk_ids]
  → grouped matmul → combine → routed_out`。捕获期 gather 读持久 buffer（固定地址），
  buffer 内容由前一步 eager `moe_offload_stage` 刚写好。

切分由 PIECEWISE `split_graph` 在 `moe_offload_stage`（splitting op）处发生：
piece1（router）与 piece2（mlp）各自被捕获，中间的 stage op 被排除在编译外、走 eager
（已由 [`backends.py:1203-1207`](../../../vllm-hust/...) `is_splitting_graph` 证实）。

---

## 3. 落点：为什么能不碰 vllm core

`AscendMoERunner(MoERunner)`（[`fused_moe.py:401`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L401)）
已经子类化 `MoERunner` 并 override 了 `forward_impl` / `_forward_impl`，但**没有** override
`forward` 与 `_select_forward`。因此：

- 关闭态：`AscendMoERunner.forward` 不存在 → 用基类 `MoERunner.forward` → 走原
  `torch.ops.vllm.moe_forward` 单 op。**与今天逐行相同。**
- 打开态：在 `AscendMoERunner` 新增 `forward` override，当
  `config.offload_stage_seam=True` 时，按 §2 拓扑依次调用三个 **vllm-ascend 自注册** op；
  否则 `return super().forward(...)`。

→ **vllm core `moe_runner.py` / `layer.py` 一行不改**。这把「修改 model_runner / 核心
forward 主路径」的评审风险降到最低：改动面 = 仅 vllm-ascend、仅 offload 开启时、仅
`AscendMoERunner` 一个子类方法 + 三个 op 的注册。

> 待评审确认点 ①：是否接受在 `AscendMoERunner` 覆写 `forward`（而非 core）。我的判断是
> 这是约束允许的最小面——它不触碰 scheduler / model_runner / token_dispatcher 主路径，
> 且默认关闭、有 `super().forward()` 回退。

---

## 4. 核心难点与待评审决策

### 难点 A：把 select_experts 移出 quant_method.apply（侵入面最大）

现状标准路径（非 multistream-gate）**不**在 `forward_impl` 里单独调 `select_experts`，
而是把 `router_logits` 传进 `quant_method.apply`，topk 在 apply **内部**算
（[`fused_moe.py:690`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L690)）。三段拆分要求：

1. `moe_router` op 先算好 `(topk_ids, topk_weights)`；
2. `moe_mlp` op 调用的 `apply` 必须**接收预算好的 topk**，而不是自己重算。

`MoEFusedExpertsInput`（[`moe_stage_contracts.py:68`](../../vllm_ascend/ops/fused_moe/moe_stage_contracts.py#L68)）
已含 `topk_weights` / `topk_ids` 字段——脚手架**已预期**传入预算 topk。但 quant_method
当前 `apply` 签名是「传 router_logits、内部 select」。需要一条**接收预算 topk 的 apply
路径**。

> 待评审决策 ②（三选一）：
> - **B1（推荐，最小侵入）**：`moe_mlp` op 内仍调用现有 `apply`，但通过 forward-context
>   旁路把 `moe_router` 算出的 `(topk_ids, topk_weights)` 透传进去，令 `select_experts`
>   命中「已提供 topk 则跳过重算」的短路。需在 `select_experts` / `apply` 加一个
>   **default-off 的「预算 topk 注入」短路**，不改默认数值路径。
> - **B2**：新增并联的 `apply_with_topk` 方法（量化方法侧），`moe_mlp` 显式调用。侵入
>   quant_method 接口，面更大。
> - **B3**：`moe_router` 只算 `router_logits` 不算 topk，`moe_mlp` 内照旧 select。**否决**
>   ——这样 stage op 拿不到 topk_ids，回到鸡生蛋。

无论哪种，**约束铁律**：`select_experts` 的 logits / top-k / group / renormalize /
scoring / e_score_bias 参数与调用语义**逐位不变**，只改「在哪调一次」。需有 UT 证明
拆分前后 `(topk_ids, topk_weights)` 逐元素相等。

### 难点 B：prepare / dispatch 与 staging 的时序

`moe_comm_method.prepare`（dispatch，[`fused_moe.py:672`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L672)）
当前在 select 与 matmul 之间。staging 必须在**捕获期 gather `log2phy[topk_ids]` 之前**
写好 buffer。gather 发生在 `moe_comm_method` 内（捕获，`moe_comm_method.py:149/529`）。
故顺序须为：`router → stage（写 buffer）→ mlp(prepare/dispatch + gather + matmul + combine)`。

> 待评审决策 ③：`moe_offload_stage` 放在 `prepare` **之前**（即 `moe_mlp` op 把
> prepare+gather+matmul+combine 全包进捕获 piece2）。这与现有
> `_maybe_apply_moe_offload_plan`（`moe_comm_method.py:320-410`）的 CAPTURE_SAFE 分支
> 一致——捕获期只读持久 buffer、不做 host sync。需确认 prepare 内部无第二处 host 决策。

### 难点 C：multistream-gate 路径

`multistream_overlap_gate=True` 时 select_experts 在
[`fused_moe.py:651`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L651) 的 gate stream 上，
topk 经 `set_flash_common3_context` 旁路传递。三段拆分**首版不支持** multistream-gate +
offload 同开。

> 待评审决策 ④：首版限制 `offload_stage_seam` 与 `multistream_overlap_gate` 互斥
> （二者同开则启动期报错或自动关 seam 并告警）。Qwen3-30B-A3B 验证路径不依赖 gate 流。

### 难点 E：router 消费 pre- 还是 post-`prepare` 的 logits（数值等价新约束）

**代码实测发现（grounding）**：标准路径 `select_experts` 在 `quant_method.apply` **内部**
运行，消费的是 **post-`prepare`** 的 `x`/`router_logits`（apply 收到的是
[`fused_moe.py:680/693`](../../vllm_ascend/ops/fused_moe/fused_moe.py#L680) 重新赋值后的
张量）。而 AllGather 的 `prepare` **会改写** `router_logits`（pad / all_gather /
tensor_split per TP rank，[`prepare_finalize.py:354/395-399`](../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L354)）。

把 `moe_router` 提到 `prepare` **之前**（§2 拓扑），只有当 `prepare` 对 logits 是**恒等
变换**时才与现状逐位一致。实测：单卡 **TP=DP=PCP=1** 时——
- `_prepare_with_dp_group`：`dp_size>1` False、`pcp_size>1` False → logits **不变**（恒等）；
- `_prepare_with_ep_group`：`ep_size=1` 时 `all_gather` 恒等 → logits **不变**。

即**单卡 Qwen3-30B-A3B 验证目标（已验证服务命令）下,router-before-prepare 逐位等价**。
多 TP/DP/PCP 时 prepare 非恒等 → router 必须消费 post-prepare logits。

> 待评审决策 ⑥（新增）：首版**仅支持 identity-prepare 拓扑**（单卡 TP=DP=PCP=EP=1）。
> seam=1 且检测到 `dp_size>1 / pcp_size>1 / ep_size>1` → 启动期 assert 报错或自动关 seam
> 并告警。**推荐**：先按此 scope 落 P1（匹配已验证目标），把「prepare 折叠进 piece1
> （router op 返回 prepared_hidden_states + topk，使任意拓扑下 select 都吃 post-prepare
> logits）」列为后续多卡扩展项。理由：单卡 bf16 AllGather offload 下 prepare 对 logits/
> hidden 均恒等、mc2_mask/pertoken_scale/padded_shape 全 None，折叠收益为零、徒增 op 输出复杂度。

### 难点 D：moe_forward_shared（带共享专家）

`_shared_experts is not None` 时基类走 `moe_forward_shared`（返回 tuple）。Qwen3-30B-A3B
路由专家无共享专家路径走 `moe_forward`。

> 待评审决策 ⑤：首版三段拆分**仅覆盖 `_shared_experts is None`** 分支；带共享专家时
> `offload_stage_seam` 自动回退 `super().forward()`（即不拆，仍 monolithic）。

---

## 5. 数值正确性论证计划（用数据说话）

拆分必须证明在**关闭态零影响**、**打开态 Regime A 与 BASE 数值等价**、**打开态
Regime B decode 每步真跑 eager staging**。验证矩阵：

| 编号 | 配置 | 期望 | 证据来源 |
|------|------|------|----------|
| V-A | seam=0 | 与今天逐行相同（走 monolithic moe_forward） | UT：`AscendMoERunner.forward` 未注册时 `_forward_entry` 不变；NPU：captured==今天 |
| V-B | seam=1, Regime A (slots≥n), eager | router 拆分前后 `(topk_ids,topk_weights)` 逐元素相等 | 新增 UT + eager probe |
| V-C | seam=1, Regime A, **captured** (no enforce_eager) | captured tokens == BASE（到 1e-5） | `run_graph_compat_capture_probe.py` |
| V-D | seam=1, Regime B (slots<n), captured | decode 每步 SEW_PROBE 出现 EAGER 行（≠R3 的 0 行）；token 正确 | probe 计数 |
| V-E | split 切点验证 | 编译图 dump 中 `moe_router` / `moe_mlp` 为顶层节点、`moe_offload_stage` 为 splitting boundary | torch.compile graph dump |

V-D / V-E 是**本设计成败的判据**：若 decode probe 仍 0 行，则切点未生效，设计推翻。

---

## 6. 分阶段实现计划（评审通过后）

> 每阶段结束跑 16/16 现有 UT + 新增 UT，保持绿；NPU 实测用 NPU1（用户指定，确认空闲）。

- **P1 — router op + 关闭态回退骨架**：注册 `vllm::moe_router`（不透明、捕获安全）；
  `AscendMoERunner.forward` override，seam=0 走 `super().forward()`，seam=1 暂时只插入
  router op 后仍调原 mlp（先证 router 拆分数值等价 V-B），共享专家/multistream 自动回退。
- **P2 — mlp op + topk 注入短路（决策②）**：注册 `vllm::moe_mlp`，按选定的 B1/B2 接收
  预算 topk；串成 router | stage | mlp 三段。跑 V-C。
- **P3 — 切点验证 + Regime B**：graph dump 验顶层节点（V-E）；NPU captured decode probe
  验每步 eager staging（V-D）。
- **P4 — 文档与消融**：结果记入 `findings.md` / `progress.md` 与 memory；补 V-A 无回归表。

---

## 7. 风险与回退

| 风险 | 缓解 |
|------|------|
| router 拆分引入数值偏差 | V-B 逐元素相等 UT 作为合并门槛；不等则推翻 |
| topk 注入短路污染默认路径 | 短路 default-off，仅 seam=1 且注入非空时激活；UT 覆盖 seam=0 路径不变 |
| split 未在预期处切 | V-E graph dump 验证；未切则 §2 拓扑不成立，回退本设计 |
| 共享专家/multistream 组合 | 首版互斥 + 自动回退 super().forward()，不阻塞 Qwen3-30B-A3B 主验证 |
| 评审认为覆写 runner.forward 仍属主路径改动 | 全程 default-off + super() 回退；可进一步把 override 限定到 offload 注册过的 layer |

---

## 8. 待你确认的评审清单（开发前）

1. **①** 接受在 `AscendMoERunner.forward`（vllm-ascend 子类）覆写、不碰 vllm core？
2. **②** select_experts 外提方案选 **B1（推荐）** / B2 / B3？
3. **③** `moe_offload_stage` 置于 `prepare` 之前、`moe_mlp` 包住 prepare+gather+matmul+combine？
4. **④** 首版 `offload_stage_seam` 与 `multistream_overlap_gate` 互斥？
5. **⑤** 首版仅覆盖 `_shared_experts is None`，带共享专家自动回退 monolithic？

确认后我按 §6 进入 P1。在你确认前，**不修改任何 vllm core 文件，也不动 `AscendMoERunner`**。
