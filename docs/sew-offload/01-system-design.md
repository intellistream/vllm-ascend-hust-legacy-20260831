# SEW-Offload 系统设计

## 1. 目标与问题边界

SEW-Offload 面向单机单卡 Ascend NPU 上的 MoE expert offloading 场景。目标是在 HBM 无法容纳完整 expert 权重时，通过 NPU 友好的固定窗口、专家预取和并行执行编排，尽量隐藏 host 到 HBM 的 expert 加载时间。

本设计聚焦两个关键环节：

1. **专家高效预取**：在 expert 真正进入 grouped MLP 计算前，提前决定需要搬运哪些 expert、搬到哪个固定 HBM slot、何时开始搬运。
2. **并行计算掩盖加载时间**：当 expert miss 已经发生时，不让计算流被动等待所有 miss expert 加载完成，而是优先执行已在 HBM 中的 expert，并用计算阶段覆盖 miss expert 的加载时间。

SEW-Offload 不修改 MoE 模型语义：

- 不重训练 router。
- 不修改 router logits。
- 不改变 top-k expert 激活。
- 不 drop token。
- 不把现有 vLLM Ascend 已经实现的 per-expert count / grouped execution 作为新贡献。

SEW-Offload 的核心贡献是把动态 expert 工作集映射为 Ascend 友好的固定 expert window，并围绕这个 window 做 deadline-aware prefetch 和 hit-first phased execution。

## 2. 现有基础与新问题

vLLM Ascend 现有 MoE 后端已经完成了 token 到 expert 的 grouped execution 表示。也就是说，系统已经能从动态 `topk_ids` 得到 per-expert token count、`group_list` 或类似的 grouped matmul 输入。SEW-Offload 不需要重新设计 token dispatch。

但现有 grouped MoE 路径通常隐含一个前提：expert 权重已经在 HBM 中常驻。offloading 场景下，这个前提不再成立。HBM 中只能保存部分 expert 权重，其余 expert 需要从 host 侧加载。

如果把 GPU-style expert cache 直接搬到 Ascend NPU 上，会遇到三个问题：

1. **动态 buffer 破坏静态执行规律**  
   expert miss 后如果临时分配 HBM buffer，权重地址、layout、expert 顺序可能每轮变化，难以复用 ACLGraph/static-kernel 需要的稳定入口。

2. **per-expert 小执行单元代价高**  
   如果每个 miss expert 都单独加载、单独执行，会引入大量小 kernel、小 graph 或队列同步，容易被 host launch 和 stream 资源限制放大。

3. **cache hit rate 不是最终目标**  
   offloading 的真实性能损失来自暴露在关键路径上的加载时间。即使 hit rate 不高，只要 miss 加载能被计算覆盖，端到端延迟仍可能可控。

因此，本系统的目标指标不是单纯的 `hit rate`，而是：

```text
exposed_stall = max(0, expert_load_time - overlap_time)
```

其中 `expert_load_time` 是 miss expert 从 host 加载到 HBM slot 并准备好执行的时间，`overlap_time` 是这段加载能被 routing、dispatch、resident expert compute、后续 layer compute 或下一 decode step 前窗口覆盖的时间。

## 3. Ascend NPU 特性与设计机会

### 3.1 固定地址与静态图

Ascend 上的 ACLGraph / static graph 路径通过 capture/replay 降低 host launch overhead。graph replay 更偏好稳定输入 shape、稳定 tensor 地址和有限数量的 graph bucket。

SEW-Offload 利用这一点，将 HBM 中的 expert 权重空间预先划分为固定 slots：

```text
slot_0: fixed address, fixed shape, fixed layout
slot_1: fixed address, fixed shape, fixed layout
slot_2: fixed address, fixed shape, fixed layout
...
```

运行时动态变化的是映射关系：

```text
(layer 17, expert 42) -> slot_3
(layer 17, expert 8)  -> slot_5
```

执行图看到的仍然是 `slot_3`、`slot_5`，而不是每轮新出现的 expert buffer。

### 3.2 显式预取与数据搬运

vLLM Ascend 已经存在 `torch_npu.npu_prefetch`、prefetch stream 和 weight prefetch pipeline。现有机制主要用于把已经在设备侧的权重提前预热到更靠近计算单元的缓存层级。

SEW-Offload 在此基础上扩展一层更大的 offloading pipeline：

```text
host expert store -> HBM expert slot -> device-side prefetch/cache -> Cube compute
```

host 到 HBM 的 expert 加载由 SEW-Offload 调度；HBM 内的后续预取可以复用或借鉴现有 `npu_prefetch` 机制。

### 3.3 Grouped Matmul 与少量 phase

Ascend MoE 后端已经支持 grouped MoE 执行。SEW-Offload 应避免 one-expert-one-kernel，而是将 active experts 合并为少量 execution phases：

```text
phase 0: slot-hit experts grouped MLP
phase 1: miss-ready experts grouped MLP
```

这样既保留 grouped matmul 的吞吐优势，也避免大量小 kernel 或小 graph 带来的队列开销。

### 3.4 MTE/Cube/Vector 分工

Ascend C 暴露出 GM、UB、L1、L0A/L0B/L0C、MTE、Cube、Vector 等层次。对 SEW-Offload 来说，这意味着 expert slot 不能只是普通缓存条目，还应尽量保持：

- 固定 dtype。
- 固定 shape。
- 固定 alignment。
- 固定 layout。
- 固定 slot tensor 地址。
- 尽量避免运行时 repack 或 layout transform。

这为后续静态 kernel、tile-level prefetch 或 slot-local grouped matmul 优化留下空间。

## 4. 总体架构

SEW-Offload 由六个高内聚模块组成，建议放在 `vllm_ascend/moe_offload/` 中，并默认关闭。

```text
                 topk_ids / group_list / expert_token_nums
                              |
                              v
                    +--------------------+
                    |  TraceCollector    |
                    +--------------------+
                              |
                              v
+----------------+   +--------------------+   +-------------------+
| HostExpertStore|-->| PrefetchPlanner    |-->| ExpertSlotBank    |
+----------------+   +--------------------+   +-------------------+
                              |                         |
                              v                         v
                    +--------------------+   fixed slot weights
                    | PhaseScheduler     |----------------------+
                    +--------------------+                      |
                              |                                  v
                              +-----------------------> grouped MoE backend
```

### 4.1 HostExpertStore

`HostExpertStore` 管理 host 侧完整 expert 权重。它负责：

- 保存每层每个 expert 的权重元数据。
- 支持按 whole expert 加载。
- 后续可扩展为按 `gate_up` / `down` 分段加载。
- 后续可扩展为 tile-level loading。
- 记录 host 内存位置、NUMA 亲和性和加载耗时。

MVP 阶段可以先实现 whole-expert granularity，即一个 expert 的 `w13/gate_up` 和 `w2/down` 作为一个加载单元。优化阶段再拆成 staged loading。

### 4.2 ExpertSlotBank

`ExpertSlotBank` 是 HBM 中固定 expert slots 的管理器。它负责：

- 预分配固定数量 HBM slots。
- 保持 slot tensor 地址稳定。
- 维护 `expert_id -> slot_id` 与 `slot_id -> expert_id`。
- 标记 slot 状态：`empty`、`loading`、`ready`、`computing`、`evictable`。
- 支持 replacement，但 replacement 不改变 slot 地址。

slot 是 Ascend-specific 的核心抽象。它让动态 expert residency 对上层策略可变，但对下层执行图保持稳定。

### 4.3 PrefetchPlanner

`PrefetchPlanner` 决定哪些 expert 应该被提前加载到 slot。

它的输入包括：

- 当前 layer 的 active experts。
- 当前 layer 的 per-expert token count。
- 上一 decode step 同层 active experts。
- 相邻 layer 的历史 expert overlap。
- slot 当前驻留状态。
- expert 加载耗时估计。
- slot budget。

它的输出是一个 prefetch plan：

```text
[(layer_id, expert_id, target_slot, priority, deadline), ...]
```

预取优先级不应只根据是否命中，而应根据 miss 是否会暴露在关键路径上。一个可用的初始评分函数是：

```text
score(e) =
    P_use(e) * token_count(e) * load_penalty(e)
    / max(deadline(e) - now, epsilon)
```

其中：

- `P_use(e)` 表示 expert 被使用的概率。
- `token_count(e)` 表示 expert 对应的 token 数。
- `load_penalty(e)` 表示加载该 expert 的代价。
- `deadline(e)` 表示该 expert 最晚必须 ready 的时间。

### 4.4 TransferEngine

`TransferEngine` 执行具体搬运。它负责：

- 在独立 load stream 上发起 host 到 HBM slot 的加载。
- 维护 load event。
- 避免覆盖正在计算的 slot。
- 在加载完成后触发 slot 状态切换。
- 可选触发 device-side `npu_prefetch`，把 slot 中即将使用的权重进一步预热。

设计上，host 到 HBM 的加载与 device-side prefetch 分开处理：

```text
TransferEngine: host -> HBM slot
NPU prefetch:   HBM slot -> cache / closer memory hierarchy
```

这样可以清楚地区分 offloading 本身和 NPU 内部缓存预热。

### 4.5 PhaseScheduler

`PhaseScheduler` 是隐藏加载时间的核心。它把当前层 active experts 分成 hit 和 miss：

```text
hit_experts  = active experts already ready in slots
miss_experts = active experts not ready in slots
```

然后决定执行方式：

1. 如果 miss 加载很快，等待短时间后执行单个 grouped phase。
2. 如果 hit compute 足够长，先执行 hit phase，同时加载 miss experts。
3. 如果 miss experts 到齐，再执行 miss phase。
4. 如果 miss 太多或 hit 太少，退化到同步加载再执行，避免 phase split 开销超过收益。

初始决策规则可以写成：

```text
if predicted_load_time > split_overhead
   and predicted_hit_compute_time > useful_overlap_threshold:
       run hit-first phased execution
else:
       run wait-and-single-phase execution
```

### 4.6 CostModel

`CostModel` 记录和预测以下时间：

- expert host-to-HBM load time。
- slot replacement time。
- device-side prefetch time。
- grouped MLP phase compute time。
- phase split overhead。
- dispatch/combine overhead。

MVP 阶段可以用在线 profiling 的滑动平均值。后续可引入按 layer、expert、weight size、token count 细分的模型。

## 5. 专家高效预取设计

### 5.1 预取窗口

SEW-Offload 至少使用三类预取窗口。

第一类是 layer-local window：

```text
layer l routing 完成后，立即知道当前层 active experts。
对 miss experts 发起加载。
```

这个窗口短，但信息准确。

第二类是 cross-layer window：

```text
执行 layer l 时，预测 layer l+1 或 l+k 的 expert 工作集。
```

这个窗口更长，但预测不确定。

第三类是 cross-step decode window：

```text
decode step t 的 layer l active experts
用于预测 decode step t+1 的 layer l active experts。
```

MoE decode 通常存在一定 expert locality。这个窗口很有价值，因为从 step t 的 layer l 结束到 step t+1 再次到达 layer l，中间隔着许多后续层的计算，适合隐藏 host-to-HBM load。

### 5.2 预取粒度

预取粒度分三阶段推进：

1. **Whole-expert prefetch**  
   一个 expert 的全部 MLP 权重作为加载单元。实现简单，适合 MVP。

2. **Split-weight prefetch**  
   将 `gate_up/w13` 与 `down/w2` 分开加载。优先加载 `gate_up`，让 GMM1 尽早开始，同时继续加载 `down`，把 `down` 加载隐藏在 GMM1 与 activation 之后。

3. **Tile-level prefetch**  
   对 hot expert 的权重 tile 做细粒度加载。这个阶段更接近 Ascend C/MTE 优化，不建议作为第一版目标。

推荐路线是：MVP 使用 whole-expert，论文优化版本实现 split-weight prefetch。

### 5.3 Slot replacement

replacement 不应只用 LRU。因为 MoE offloading 的目标是隐藏 stall，替换策略要考虑 deadline 和 token count。

一个初始 victim 选择规则：

```text
evict_score(slot) =
    future_use_prob(slot.expert)
    * expected_future_token_count(slot.expert)
    / reload_cost(slot.expert)
```

选择 `evict_score` 最低且不在 computing/loading 状态的 slot 作为 victim。

为了降低 correctness 风险，MVP 阶段可以使用保守策略：

- 禁止驱逐当前层 active expert。
- 禁止驱逐 loading slot。
- 禁止驱逐 computing slot。
- 只在 layer boundary 做 replacement。

### 5.4 预取失败处理

预取失败或未按 deadline 完成时，系统不能改变模型语义。可选 fallback：

1. 等待 miss expert 完成加载，再执行。
2. 如果 hit phase 已完成但 miss 仍未 ready，进入 blocking wait。
3. 如果 slot 不足，退化到同步加载单 phase。

不允许：

- 跳过 expert。
- 用近似 expert 替代。
- 修改 top-k。
- 改变 gate weight。

## 6. 并行计算掩盖加载时间设计

### 6.1 Hit-first phased execution

当前层 MoE active experts 被划分后，优先执行 hit experts：

```text
compute stream:
    route -> dispatch hit tokens -> grouped MLP(hit) -> partial combine

load stream:
    load miss experts into slots -> optional npu_prefetch -> signal ready

compute stream:
    dispatch miss tokens -> grouped MLP(miss) -> final combine
```

这样 miss expert 的加载时间可以被 hit expert 的计算覆盖。

### 6.2 为什么不是 per-expert execution

per-expert execution 表面上更灵活，但在 Ascend 上可能产生几个问题：

- 小 kernel 数量多。
- device queue 串行化风险高。
- graph 变体过多。
- stream 资源压力更大。
- grouped matmul 吞吐优势被削弱。

因此 SEW-Offload 只允许少数 phase：

```text
phase 0: all ready hit experts
phase 1: all ready miss experts
phase 2: rare fallback phase, optional
```

默认最多两个 phase，除非 profiling 显示第三个 phase 有明显收益。

### 6.3 Phase split 的收益判断

phase split 有额外成本，包括 dispatch、grouped matmul launch、combine 和同步事件。因此不是所有 miss 都应该 split。

收益判断公式：

```text
benefit =
    min(predicted_load_time, predicted_hit_compute_time)
    - split_overhead
```

只有当 `benefit > 0` 时才启用 hit-first phased execution。

不同 workload 的策略不同：

- decode 小 batch：hit compute 短，更依赖 cross-step prefetch。
- prefill 大 batch：hit compute 长，更适合 phased execution。
- mixed batch：优先把 prefill 的 resident expert compute 用来覆盖 decode miss load。

### 6.4 Partial combine

hit phase 和 miss phase 的输出最终必须回到原 token 顺序。为了保持 correctness，可以采用两种方式：

1. 每个 phase 写入全局 output buffer 的对应 token 位置。
2. 每个 phase 产生 phase-local output，最后统一 combine。

MVP 推荐第一种，因为它更直接地保持 token index 映射，但需要严格验证写入位置与 gate weight 对齐。

## 7. 运行时数据流

完整数据流如下：

```text
1. Router 产生 topk_ids / topk_weights
2. 现有 vLLM Ascend dispatcher 产生 group_list / expert_token_nums
3. TraceCollector 记录 active experts 和 token count
4. PrefetchPlanner 生成当前层 miss load 和未来层 prefetch plan
5. ExpertSlotBank 查询 hit/miss，分配 target slots
6. TransferEngine 在 load stream 发起 host->HBM slot 加载
7. PhaseScheduler 决定 single phase 还是 hit-first phases
8. grouped MoE backend 使用 slot weights 执行 GMM
9. combine 输出，保持原 MoE 语义
10. CostModel 更新 load/compute/overlap 统计
```

## 8. 与 vLLM Ascend 的集成边界

SEW-Offload 应作为默认关闭的独立 runtime 接入。

建议包结构：

```text
vllm_ascend/moe_offload/
  __init__.py
  config.py
  host_store.py
  slot_bank.py
  prefetch_planner.py
  transfer_engine.py
  phase_scheduler.py
  cost_model.py
  trace_collector.py
  runtime.py
```

建议最小 hook 点：

```text
Ascend fused MoE expert execution boundary
```

也就是在 router/top-k 和 token dispatch 已经完成之后，在 grouped MLP 使用 expert weights 之前，插入 SEW-Offload runtime：

```text
topk_ids/topk_weights
    -> existing token dispatcher
    -> SEW-Offload slot prepare + phase plan
    -> existing grouped MLP backend
    -> existing combine/finalize
```

首版不要修改：

- scheduler 主路径。
- model runner 主路径。
- 现有 Ascend C grouped matmul kernel。
- router 逻辑。

新增环境变量必须集中定义在 `vllm_ascend/envs.py`，例如：

```text
VLLM_ASCEND_MOE_OFFLOAD_ENABLED
VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS
VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY
VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES
VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY
```

## 9. Correctness 约束

SEW-Offload 只改变 expert 权重驻留和执行时序，不改变数学语义。必须保证：

1. 每个 token 的 top-k expert 不变。
2. 每个 expert 的 gate weight 不变。
3. 每个 expert 使用正确的权重版本。
4. slot remap 不改变 expert ID 语义。
5. hit phase 和 miss phase 的输出最终写回正确 token 位置。
6. fallback 路径与原始 grouped MoE 输出一致。

建议增加三类检查：

- `expert_id -> slot_id` checksum。
- layer-level output 对齐测试。
- E2E replay 测试：同一批输入在 offload disabled/enabled 下输出一致。

## 10. 评价指标

主要指标：

- TPOT。
- ITL。
- P50/P95/P99 latency。
- throughput。
- HBM 使用量。
- host-to-HBM load time。
- exposed stall。
- load/compute overlap ratio。

辅助指标：

- slot hit rate。
- prefetch accuracy。
- prefetch timeliness。
- phase split 次数。
- 每层 active expert count。
- 每层 token count 分布。
- graph replay 命中率。

关键消融：

1. 无 offload，全量 expert 常驻。
2. 同步 expert load，无预取。
3. LRU expert cache。
4. deadline-aware prefetch。
5. deadline-aware prefetch + hit-first phased execution。
6. fixed slot + graph/static window。
7. whole-expert prefetch vs split-weight prefetch。

## 11. 分阶段实现路线

### Phase 0: Trace-only

只记录 routed expert 工作集，不改变执行。

产物：

- 每层 active expert set。
- 每层 expert token count。
- decode step 间 expert locality。
- slot budget simulator。

### Phase 1: Sync fixed slot

实现 host expert store 和 HBM fixed slots，但 miss 时同步加载。

目标：

- 跑通 correctness。
- 验证 slot remap。
- 验证固定 slot 替代全量 expert 常驻的可行性。

### Phase 2: Async prefetch

加入 load stream 和 deadline-aware prefetch。

目标：

- 测量 host-to-HBM load time。
- 测量 prefetch timeliness。
- 降低 blocking miss。

### Phase 3: Hit-first phased execution

加入 hit/miss phase split。

目标：

- 用 hit expert compute 覆盖 miss expert load。
- 降低 exposed stall。
- 控制 phase split overhead。

### Phase 4: Ascend-specific static window

固定 slot 数、capacity tier 和 phase 数，尝试 ACLGraph/static kernel 友好的执行窗口。

目标：

- 验证固定 slot 地址对 graph replay 的帮助。
- 验证少量 phase 是否比 per-expert dynamic path 更稳定。

### Phase 5: Split-weight / tile-level optimization

将 expert 加载从 whole expert 拆成 `gate_up` 和 `down`，进一步利用 GMM1/GMM2 的自然顺序隐藏加载。

目标：

- 优先加载 `gate_up`。
- 在 GMM1/activation 期间加载 `down`。
- 探索 tile-level prefetch 是否值得进入 Ascend C kernel 层。

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| miss expert 太多 | hit phase 太短，无法隐藏 load | 增加 cross-step prefetch；限制 phase split；同步 fallback |
| phase split overhead 过大 | 吞吐下降 | 最多两个 phase；用 CostModel 动态判断 |
| slot remap 错误 | 输出错误 | checksum、逐层输出对齐、E2E replay |
| host-to-HBM 带宽不足 | offloading 慢于全量常驻 | 量化 expert store、split-weight prefetch、NUMA binding |
| graph 收益不明显 | 论文贡献变弱 | 保留 prefetch-hidden scheduling 作为主贡献，graph 作为 Ascend-specific ablation |
| 工程侵入过大 | 影响主仓稳定性 | 独立包、默认关闭、trace-only 起步 |

## 13. 设计总结

SEW-Offload 的研究点不是简单做 expert cache，而是把 Ascend 单卡 MoE offloading 重构成一个固定窗口调度问题：

```text
动态 expert 工作集
    -> 固定 HBM expert slots
    -> deadline-aware prefetch
    -> hit-first phased grouped execution
    -> graph/static-kernel friendly window
```

固定 slot 解决 Ascend 能否稳定执行；deadline-aware prefetch 解决哪些 expert 应提前加载；hit-first phased execution 解决 miss 已经发生时如何用并行计算隐藏加载时间。

最终目标是降低：

```text
exposed_stall = max(0, expert_load_time - overlap_time)
```

而不是仅仅提高 cache hit rate。
