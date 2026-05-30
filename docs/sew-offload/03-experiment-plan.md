# SEW-Offload 实验计划

## 1. 实验目标

SEW-Offload 的实验目标是回答三个问题：

1. 在单卡 64GB Ascend 910B3 上，MoE expert offloading 的主要瓶颈是否来自 host-to-HBM expert 加载暴露在关键路径上？
2. 固定 expert slots 与 deadline-aware prefetch 是否能减少 blocking miss 和 exposed stall？
3. hit-first phased execution 是否能用 resident expert compute 掩盖 miss expert 加载时间？

实验不只报告 slot hit rate，而要报告 load/compute overlap 和 exposed stall。

## 2. 实验环境

当前目标硬件：

```text
NPU: Ascend 910B3
HBM: 64GB per card
CANN: 8.5.1
OS: openEuler 24.03 aarch64
```

实验原则：

- 使用单机单卡。
- 选择空闲 NPU。
- 保留真实 KV cache 压力。
- 通过 slot budget 控制 expert HBM 驻留容量。
- 所有 SEW-Offload 功能默认关闭，实验显式启用。

## 3. 模型

### 3.1 主模型

```text
Qwen3-30B-A3B
```

用途：

- 论文主结果。
- 单卡 HBM 受限 offloading 评估。
- decode-heavy 与 mixed workload。

前置条件：

- 下载或挂载模型权重。
- 确认 vLLM Ascend 可识别该模型。
- 确认单卡 eager 或受限 slot 模式能进入 MoE 路径。

### 3.2 压力模型

```text
/data/models/Qwen3.5-122B-A10B
```

用途：

- expert locality trace。
- offline simulator。
- slot budget 压力测试。

注意：

- 不要求第一阶段完整单卡 serving。
- 可先用 trace/simulator 评估策略上界。

### 3.3 环境基线模型

```text
Qwen3-8B
Qwen3-32B
```

用途：

- 验证 vLLM Ascend 环境。
- 验证 ACLGraph。
- 验证 weight prefetch。
- 不作为 MoE offloading 主要结果。

## 4. Workloads

### 4.1 Decode-heavy

模拟在线服务常见 decode 阶段：

```text
prompt length: 128 / 512
output length: 128 / 256 / 512
batch size: 1 / 4 / 8 / 16
```

关注：

- TPOT。
- ITL。
- P99 latency。
- cross-step expert locality。
- prefetch timeliness。

### 4.2 Prefill-heavy

```text
prompt length: 2048 / 4096 / 8192
output length: 32
batch size: 1 / 2 / 4
```

关注：

- active expert 数量。
- 每个 expert token count。
- hit phase compute 是否足够覆盖 miss load。

### 4.3 Mixed prefill/decode

模拟连续批处理：

```text
prefill requests + decode requests mixed in one serving window
```

关注：

- prefill compute 是否能覆盖 decode expert load。
- mixed batch 下 phase split 是否带来收益。
- 尾延迟是否恶化。

### 4.4 Locality stress

构造两类请求：

1. 主题重复 prompt：预期 expert locality 更强。
2. 随机 prompt：预期 expert locality 更弱。

关注：

- deadline-aware prefetch 对不同 locality 的鲁棒性。
- 相比 LRU 的收益差异。

## 5. Baselines

| Baseline | 描述 | 目的 |
| --- | --- | --- |
| Full-resident | expert 全量常驻 HBM | 性能上界，但可能不满足单卡 HBM |
| Sync-load | miss expert 同步加载，加载完成后执行 | offloading 下界 |
| LRU-cache | 固定 slot + LRU replacement，无 deadline-aware prefetch | GPU-style cache baseline |
| Predict-prefetch | 基于上一 step 预测 expert，但无 fixed phase scheduling | 分离 prefetch 贡献 |
| SEW-prefetch | fixed slot + deadline-aware prefetch | 验证高效预取 |
| SEW-prefetch-phase | fixed slot + deadline-aware prefetch + hit-first phased execution | 验证并行掩盖加载 |
| SEW-static-window | 加入 fixed capacity tier / graph-friendly window | 验证 Ascend-specific 静态执行收益 |

## 6. Ablations

### 6.1 Slot budget

```text
num_slots: 2 / 4 / 8 / 16 / 32
```

报告：

- HBM 使用量。
- slot hit rate。
- exposed stall。
- TPOT / P99。

### 6.2 Prefetch policy

```text
none
LRU
last-step
token-count weighted
deadline-aware
oracle
```

`oracle` 用于估计策略上界，不作为真实在线系统。

### 6.3 Phase policy

```text
single-phase
always-split
cost-model split
oracle split
```

目标是证明 cost-model split 能避免 always-split 在小 batch 上的额外开销。

### 6.4 Prefetch granularity

```text
whole expert
gate_up first + down later
tile-level simulated
```

MVP 先做 whole expert；论文优化版本重点评估 split-weight prefetch。

### 6.5 Graph/static window

```text
eager
ACLGraph compatible window
capacity tier window
```

关注：

- graph replay hit。
- padding 开销。
- phase 数量稳定性。
- static kernel 可用性。

## 7. Metrics

### 7.1 端到端指标

- throughput。
- TPOT。
- ITL。
- P50 latency。
- P95 latency。
- P99 latency。

### 7.2 Offloading 指标

- host-to-HBM load time。
- blocking miss count。
- async prefetch count。
- prefetch accuracy。
- prefetch timeliness。
- exposed stall。
- hidden load time。
- load/compute overlap ratio。

### 7.3 MoE 指标

- active expert count per layer。
- expert token count distribution。
- slot hit rate。
- slot replacement count。
- phase split count。
- phase size distribution。

### 7.4 Ascend-specific 指标

- ACLGraph replay ratio。
- graph fallback count。
- static window bucket distribution。
- NPU stream wait time。
- AICore utilization。
- HBM usage。

## 8. Trace 实验

Trace-only 阶段不改变执行，只记录：

```json
{
  "step_id": 0,
  "layer_id": 12,
  "expert_token_counts": {"3": 18, "7": 4},
  "num_tokens": 128,
  "phase": "decode"
}
```

Trace 分析输出：

- 每层 active expert 数量。
- 每层 top expert 分布。
- step-to-step overlap。
- layer-to-layer overlap。
- slot budget simulator 上界。
- oracle prefetch 上界。

## 9. Simulator 实验

Simulator 输入 trace，输出不同策略下的预测结果：

```text
slot_hit_rate
miss_count
predicted_load_ms
predicted_overlap_ms
predicted_exposed_stall_ms
replacement_count
```

Simulator 用途：

- 在实现真实 offloading 前筛选策略。
- 估算 slot budget。
- 找到 deadline-aware policy 的参数。
- 找到 phase split 的收益边界。

## 10. NPU 实测实验

### 10.1 Sync fixed slot

验证：

- correctness。
- slot remap。
- 同步加载开销。

### 10.2 Async prefetch

验证：

- load stream 能否与 compute stream 重叠。
- prefetch 是否按 deadline 完成。
- blocking miss 是否减少。

### 10.3 Hit-first phased execution

验证：

- hit phase compute 是否覆盖 miss load。
- phase split overhead 是否可控。
- P99 latency 是否改善。

### 10.4 Static window

验证：

- fixed slot 地址是否降低 graph fallback。
- capacity tier 是否降低动态图开销。
- static window 是否值得进入最终论文贡献。

## 11. Correctness 验证

每个性能实验前必须通过：

1. 单层 MoE 输出对齐。
2. 多层 replay 输出对齐。
3. offload disabled/enabled 文本输出一致。
4. slot remap checksum。
5. phase split output 写回位置检查。

允许的数值差异：

- bf16 路径使用合理容差。
- token 级输出需人工检查或 replay 检查。
- 不允许 expert id 错配。

## 12. 结果图表计划

论文图表建议：

1. **Figure 1**：GPU-style offloading vs SEW-Offload 固定窗口。
2. **Figure 2**：host-to-HBM load 暴露时间分解。
3. **Figure 3**：slot budget 对 TPOT/P99 的影响。
4. **Figure 4**：不同 prefetch policy 的 exposed stall。
5. **Figure 5**：hit-first phased execution 的 timeline。
6. **Figure 6**：whole-expert vs split-weight prefetch。
7. **Table 1**：模型与硬件配置。
8. **Table 2**：ablation 总结。
9. **Table 3**：correctness 与 fallback 统计。

## 13. 实验顺序

推荐顺序：

1. dense 模型验证 vLLM Ascend 环境。
2. Qwen3-30B-A3B 或 Qwen3.5-122B-A10B trace-only。
3. offline simulator。
4. mock/small MoE sync fixed slot。
5. Qwen3-30B-A3B sync fixed slot。
6. async prefetch。
7. hit-first phased execution。
8. graph/static window ablation。
9. split-weight prefetch。

## 14. 停止条件

如果出现以下情况，需要暂停并重新设计：

- trace 显示几乎没有 expert locality，且 oracle prefetch 上界很低。
- host-to-HBM 加载时间远大于可覆盖 compute window。
- phase split 在所有 batch 下都带来负收益。
- fixed slot 无法接入现有 grouped MoE 权重路径。
- correctness 无法稳定保证。

如果 graph/static window 收益较弱，不中止项目；将其降级为 Ascend-specific ablation，保留 prefetch-hidden scheduling 作为主线。
