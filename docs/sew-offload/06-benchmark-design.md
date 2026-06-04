# SEW-Offload Minimal Benchmark

## 目标

这个 benchmark 只做一件事：为当前单卡 Ascend NPU 上的 MoE offloading 实验固定一个高复用、可比较的基准口径。

当前阶段只评估 vLLM Ascend 原生 offloading 方法；SEW 后续改进、额外 baseline、复杂 artifact 规范和 validity gate 暂不纳入本 benchmark 定义。

后续所有相关实验先检查：

```text
docs/sew-offload/benchmark_config.yaml
```

只要实验用于对比或写入论文/slide，就必须说明是否符合这个配置。

## 1. 固定模型

主模型固定为：

```text
/data/shared-models/Qwen3-30B-A3B
```

基本设置：

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen3-30B-A3B |
| 架构 | Qwen3 MoE |
| 层数 | 48 |
| Experts | 128 per MoE layer |
| Top-k | 8 |
| dtype | bf16 |
| Tensor parallel | 1 |
| 设备 | 单张 Ascend 910B3 |

约束：

- 不重训练 router。
- 不修改 top-k。
- 不修改 expert 激活语义。
- 不 drop token。
- 不 drop expert。

## 2. 固定数据集

正式 benchmark 数据集固定为：

```text
ShareGPT_V3_unfiltered_cleaned_split
```

本地文件：

```text
/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

原因：

- 来自真实人类对话请求，更接近 serving 场景。
- prompt 长度和内容分布丰富，适合观察 MoE expert offloading 的真实压力。
- 可以用固定 seed 和 tokenizer length filter 生成可复现请求集合。

当前阶段固定生成 1000 条请求：

```text
artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/requests.jsonl
```

**强制约束（适用于所有实验，包括 smoke / 调试）**：

- 所有 benchmark 请求都必须从上述 ShareGPT 数据集采样。
- 禁止使用随机 / 合成 / 拼接生成的 prompt（`random_dataset_allowed: false`、`synthetic_smoke_allowed: false`）。
- smoke 测试也必须用 ShareGPT 真实 prompt，只是请求数更少；不允许再用 synthetic seed-text 重复填充。


## 3. 固定 Workload Buckets

请求集合固定分成五类：

| Bucket | 数量 | Prompt tokens | Output tokens | 目的 |
| --- | ---: | ---: | ---: | --- |
| short_chat | 200 | 128-256 | 128 | 低延迟短对话 |
| medium_chat | 300 | 512-1024 | 128 | 常规服务请求 |
| long_prefill | 200 | 2048-4096 | 128 | prefill 压力与 active expert 多样性 |
| decode_heavy | 200 | 128-512 | 512 | 长 decode 下的 offloading 暴露开销 |
| burst_mixed | 100 | mixed | 256 | 混合请求和尾延迟压力 |

主并发固定为：

```text
1, 4, 8
```

## 4. 固定 Offload Budget

当前实测结果：

| Case | Resident weight log | Peak HBM |
| --- | ---: | ---: |
| no offload | 56.9001 GB | 63,886 MB |
| vLLM Ascend native prefetch expert offload | 43.4001 GB | 50,696-50,697 MB |

因此 benchmark 固定 offload 规模为：

```text
target_offloaded_weight_gb = 13.5
target_resident_weight_gb = 43.4
tolerance_gb = 0.5
```

解释：

- 13.5GB 来自当前真实 baseline 中 full resident 与 native prefetch offload 的 resident weight 差值。
- 后续同类实验应尽量保持约 13.5GB expert weight 被卸载，避免不同卸载规模导致结果不可比。

## 5. 固定指标

### 端到端指标

| 指标 | 说明 |
| --- | --- |
| TTFT | Time To First Token |
| TPOT | Time Per Output Token |
| ITL | Inter-Token Latency |
| latency p50/p90/p99 | 请求延迟分位数 |
| output tokens/s | 输出吞吐 |
| success rate | 成功完成比例 |

### Offloading 指标

| 指标 | 说明 |
| --- | --- |
| resident_weight_gb | 日志中的 resident model weight |
| peak_hbm_mb | 单卡 HBM 峰值 |
| host_to_hbm_bytes | CPU 到 NPU HBM 的权重加载量 |
| host_to_hbm_copy_time_ms | 权重加载耗时 |
| prefetch_wait_time_ms | 推理等待权重到达的时间 |
| exposed_stall_per_output_token_ms | 每个输出 token 暴露出来的 offloading stall |

当前最重要的核心指标是：

```text
exposed_stall_per_output_token_ms
```

它直接对应我们的研究目标：offloading 场景下，专家加载时间到底有多少暴露在关键路径上。
