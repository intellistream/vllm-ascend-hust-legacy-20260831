# SEW-Offload 项目 Charter

## 1. 项目名称

**SEW-Offload: Static Expert-Window Scheduling for Prefetch-Hidden MoE Offloading on Ascend NPUs**

中文简称：**面向昇腾 NPU 的静态专家窗口编排与预取隐藏 MoE Offloading**。

## 2. 核心目标

SEW-Offload 研究单机单卡 Ascend NPU 上的 MoE expert offloading。当单卡 HBM 无法容纳完整 expert 权重时，系统不应只做被动 expert cache，而应利用 Ascend NPU 的固定地址、静态图、显式预取和 grouped MoE 后端，将 expert 加载变成可预测、可编排、可与计算重叠的流水。

项目目标是降低：

```text
exposed_stall = max(0, expert_load_time - overlap_time)
```

而不是单纯最大化 cache hit rate。

## 3. 研究问题

现有 MoE offloading 方法通常把 expert 权重视为动态 cached device objects，主要通过 cache replacement、prefetch prediction 和 copy/compute overlap 降低传输开销。这个抽象在 Ascend NPU 上不完整：HBM 受限时，动态 expert loading 会与 Ascend 偏好的 stable weight addresses、fixed execution windows、graph/static-kernel replay 和显式数据搬运发生冲突。

SEW-Offload 的研究问题是：

> 在不修改 MoE router、不改变 top-k 激活、不 drop token 的前提下，如何把动态 expert 工作集映射为 Ascend 友好的固定 HBM expert slots，并通过 deadline-aware prefetch 与 hit-first phased execution 隐藏 host-to-HBM expert 加载时间？

## 4. 目标硬件与软件栈

当前目标设备：

- Ascend 910B3。
- 单卡约 64GB HBM。
- CANN 8.5.1。
- vLLM Ascend 当前开发仓库：`/root/vllm-ascend-hust`。

目标软件边界：

- 基于 vLLM Ascend。
- 低侵入集成。
- 新功能默认关闭。
- 优先在 `vllm_ascend/moe_offload/` 新建独立 runtime。
- 尽量不修改 scheduler、worker 主路径和现有 Ascend C grouped matmul kernel。

## 5. 目标模型

### 5.1 主模型

**Qwen3-30B-A3B**

理由：

- MoE 模型，total 参数与 activated 参数差异明显。
- 适合单卡 HBM 受限 offloading 场景。
- vLLM Ascend 官方部署建议通常使用多 NPU，说明单卡运行存在真实 HBM 压力。

### 5.2 压力模型

**Qwen3.5-122B-A10B**

本地路径：

```text
/data/models/Qwen3.5-122B-A10B
```

已知配置：

- `qwen3_5_moe`
- 48 层
- 256 experts
- top-8
- bf16

用途：

- 后期 offloading 压力测试。
- trace 和 simulator。
- 验证 slot budget、expert locality 和 prefetch window。

### 5.3 环境基线模型

Qwen3-8B / Qwen3-32B dense 模型只用于验证 vLLM Ascend、ACLGraph、weight prefetch、benchmark 工具链，不作为 MoE offloading 核心实验。

## 6. 核心设计原则

### 6.1 不改变模型语义

SEW-Offload 不改变：

- router logits。
- top-k expert ids。
- gate weights。
- token dispatch 语义。
- expert combine 语义。
- 输出精度目标。

### 6.2 固定 slot，不固定 expert

HBM 中预分配固定 expert slots：

```text
slot_0, slot_1, ..., slot_N
```

slot 的地址、shape、dtype、layout 保持稳定。运行时动态变化的是：

```text
expert_id -> slot_id
```

这样可以让动态 expert residency 与 Ascend 静态图偏好之间解耦。

### 6.3 优化暴露等待，而非只优化命中率

系统主要优化：

- miss 加载是否落在关键路径上。
- 加载时间是否被 hit expert compute 覆盖。
- 加载时间是否被 cross-layer 或 cross-step 窗口覆盖。

cache hit rate 是辅助指标，不是最终目标。

### 6.4 少量 grouped phases，不做 per-expert 小执行

Ascend NPU 上应避免 one expert one kernel / one expert one graph。SEW-Offload 采用少量 phase：

```text
phase 0: hit experts grouped MLP
phase 1: miss-ready experts grouped MLP
```

默认最多两个 phase。

## 7. 预期贡献

1. **问题重定义**  
   将 Ascend 单卡 MoE offloading 从动态 expert cache 问题重定义为固定 expert-window scheduling 问题。

2. **Static Expert Slots**  
   提出固定 HBM expert slot 抽象，让 offloaded expert 权重在动态 residency 下仍能保持稳定设备入口。

3. **Deadline-Aware Expert Prefetch**  
   基于 active expert、token count、历史 locality、加载代价和 deadline 进行预取编排。

4. **Hit-First Phased Execution**  
   在 miss 发生时优先执行 slot-hit experts，用 grouped compute 覆盖 miss expert load。

5. **Ascend-Specific Evaluation**  
   系统评估固定 slot、prefetch、phase split、ACLGraph/static window 在 Ascend 910B3 上的收益和限制。

## 8. 非目标

SEW-Offload 当前阶段不做：

- router 重训练。
- router 微调。
- expert pruning。
- expert drop。
- 用近似 expert 替代 miss expert。
- 修改模型结构。
- 大规模改写 vLLM scheduler。
- 第一阶段改写 Ascend C grouped matmul kernel。
- 把已有 per-expert count/grouped dispatch 包装成新贡献。

## 9. 交付物

### 9.1 论文线

- `paper/sew_offload_design.tex`
- `paper/research_question_reframing.md`
- 后续正式 paper skeleton
- related work matrix
- experiment plan

### 9.2 Slide 线

- `slide/sew_offload_report.tex`
- `slide/sew_offload_report.pdf`

### 9.3 工程文档线

- `docs/sew-offload/00-charter.md`
- `docs/sew-offload/01-system-design.md`
- `docs/sew-offload/02-implementation-plan.md`
- `docs/sew-offload/03-experiment-plan.md`
- `docs/sew-offload/04-reproduction.md`

### 9.4 Runtime 线

计划新建：

```text
vllm_ascend/moe_offload/
```

并逐步实现：

- trace-only observer
- offload simulator
- host expert store
- fixed slot bank
- deadline-aware prefetch
- hit-first phased execution
- metrics and profiling

## 10. 成功标准

### 10.1 Correctness

- offload disabled/enabled 输出一致。
- top-k expert 不变。
- gate weight 不变。
- token output 写回位置正确。
- slot remap 有校验。

### 10.2 Performance

在相同 HBM slot budget 下，相比同步加载或简单 LRU cache：

- 降低 exposed stall。
- 降低 TPOT / ITL。
- 改善 P95/P99 latency。
- 提高 load/compute overlap ratio。

### 10.3 Ascend-specific 证据

论文和实验需要证明收益不是“普通 GPU cache 策略自然得到”，而来自：

- 固定 slot 地址。
- 少量 grouped phases。
- Ascend prefetch stream / `npu_prefetch` 机制。
- ACLGraph/static window 友好性。

## 11. 当前风险

| 风险 | 影响 | 当前策略 |
| --- | --- | --- |
| Qwen3-30B-A3B 本地权重缺失 | 主模型无法立刻实验 | 先用 Qwen3.5-122B-A10B trace/simulator 和 dense 模型验证环境 |
| 默认 Python 环境缺少 torch/torch_npu/vllm | 不能直接运行 serving | 后续确认正确容器或 conda 环境 |
| host-to-HBM 加载太慢 | offloading 可能拖慢端到端 | 用 split-weight prefetch、cross-step prefetch 和 phase split 隐藏 |
| phase split 开销过大 | 小 batch decode 可能变慢 | CostModel 动态判断是否 split |
| graph 收益不稳定 | 静态窗口贡献变弱 | 以 prefetch-hidden scheduling 为主贡献，graph 作为 ablation |

## 12. 决策记录

- 2026-05-28：确认不能把 per-expert count/grouped execution 作为新贡献，vLLM Ascend 已经实现相关表示。
- 2026-05-28：将研究主线从 fixed cache/slot 修正为 prefetch-hidden offloading scheduling。
- 2026-05-29：确认当前设备为 Ascend 910B3 64GB HBM；确定 Qwen3-30B-A3B 为主模型，Qwen3.5-122B-A10B 为压力模型。
- 2026-05-29：将系统设计拆分为 Expert Prefetch Planner 与 Overlap Execution Scheduler 两个核心子系统。
