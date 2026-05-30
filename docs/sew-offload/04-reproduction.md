# SEW-Offload 复现说明

## 1. 复现范围

本文档说明如何复现 SEW-Offload 的研究实验。当前阶段以规划和原型为主，命令会随着 runtime 实现推进逐步补齐。

复现分四层：

1. 环境检查。
2. trace-only 数据采集。
3. offline simulator。
4. NPU runtime 实测。

如果当前默认 Python 环境无法 import `torch`、`torch_npu`、`vllm` 或 `transformers`，需要先进入 vLLM Ascend 正确运行环境或容器。

## 2. 硬件检查

查看 NPU：

```bash
npu-smi info
```

期望看到 Ascend 910B3，单卡约 64GB HBM。

选择空闲卡：

```bash
npu-smi info | less
```

记录：

- NPU id。
- HBM used/free。
- 已有进程。
- 温度和健康状态。

## 3. 软件环境检查

在 vLLM Ascend 运行环境中执行：

```bash
python3 - <<'PY'
import importlib

for name in ["torch", "torch_npu", "vllm", "vllm_ascend", "transformers"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "no __version__"))
PY
```

必须成功 import：

```text
torch
torch_npu
vllm
vllm_ascend
transformers
```

如果失败，说明当前 shell 不是 serving 环境。

## 4. 模型准备

### 4.1 主模型

推荐主模型：

```text
Qwen3-30B-A3B
```

本地需要准备模型路径，例如：

```text
/data/models/Qwen3-30B-A3B
```

### 4.2 压力模型

当前机器已发现：

```text
/data/models/Qwen3.5-122B-A10B
```

可用于 trace/simulator 和后期压力实验。

### 4.3 环境基线模型

可使用：

```text
/data/models/models--Qwen--Qwen3-8B
/data/models/models--Qwen--Qwen3-32B
```

这些 dense 模型只用于确认 vLLM Ascend 运行环境，不用于 MoE offloading 主结果。

## 5. 环境变量

SEW-Offload 计划使用以下环境变量：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=0
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=none
```

trace-only：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1
```

fixed slot 实验：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=8
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=none
```

prefetch + phased execution：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=8
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=2
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=deadline
```

## 6. 单元测试

运行 SEW-Offload 单测：

```bash
pytest tests/ut/moe_offload -q
```

运行现有 fused MoE 相关单测：

```bash
pytest tests/ut/ops/test_fused_moe.py -q
pytest tests/ut/ops/test_moe_runtime_args.py -q
```

如果当前环境没有 `torch` 或 `torch_npu`，这些命令应在正确容器中执行。

## 7. Trace-only 复现

trace-only 不改变模型执行，只记录 expert 工作集。

示例命令：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1

python benchmarks/sew_offload/collect_moe_trace.py \
  --model /data/models/Qwen3-30B-A3B \
  --npu-id 3 \
  --output artifacts/sew_offload/traces/qwen3_30b_a3b_decode.jsonl \
  --workload decode
```

输出格式：

```json
{"step_id": 0, "layer_id": 12, "expert_token_counts": {"3": 18, "7": 4}, "num_tokens": 128, "phase": "decode"}
```

需要保存：

```text
artifacts/sew_offload/traces/
```

## 8. Offline simulator 复现

运行 slot policy simulator：

```bash
python benchmarks/sew_offload/simulate_slot_policy.py \
  --trace-jsonl artifacts/sew_offload/traces/qwen3_30b_a3b_decode.jsonl \
  --num-slots 8 \
  --policy deadline \
  --output artifacts/sew_offload/sim/qwen3_30b_a3b_slots8_deadline.json
```

需要报告：

```text
slot_hit_rate
miss_count
replacement_count
predicted_load_ms
predicted_overlap_ms
predicted_exposed_stall_ms
```

对比策略：

```bash
for policy in lru last_step token_count deadline oracle; do
  python benchmarks/sew_offload/simulate_slot_policy.py \
    --trace-jsonl artifacts/sew_offload/traces/qwen3_30b_a3b_decode.jsonl \
    --num-slots 8 \
    --policy "${policy}" \
    --output "artifacts/sew_offload/sim/qwen3_30b_a3b_slots8_${policy}.json"
done
```

## 9. NPU runtime 实测

### 9.1 Full-resident baseline

SEW-Offload 关闭：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0
```

运行 serving benchmark：

```bash
python benchmarks/benchmark_serving.py \
  --model /data/models/Qwen3-30B-A3B \
  --backend vllm \
  --dataset-name random \
  --num-prompts 128
```

如果 full-resident 因单卡 HBM 不足无法运行，记录为：

```text
OOM under single-card full-resident setting
```

这本身支持 offloading 动机。

### 9.2 Sync-load baseline

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=8
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=none
```

运行同一 benchmark，记录：

- TPOT。
- ITL。
- P99。
- miss count。
- exposed stall。

### 9.3 Deadline-aware prefetch

```bash
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
```

比较 sync-load：

- blocking miss 是否减少。
- load 是否提前完成。
- exposed stall 是否下降。

### 9.4 Hit-first phased execution

```bash
export VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=2
```

比较 deadline-aware prefetch：

- hit phase 是否覆盖 miss load。
- phase split overhead 是否可控。
- P95/P99 是否改善。

## 10. Profiling 数据

每次实验保存：

```text
artifacts/sew_offload/runs/<date>/<case>/
  config.json
  env.txt
  npu_smi_before.txt
  npu_smi_after.txt
  benchmark_stdout.txt
  benchmark_metrics.json
  sew_metrics.json
  trace.jsonl
```

`sew_metrics.json` 至少包含：

```json
{
  "slot_hit_rate": 0.0,
  "blocking_miss_count": 0,
  "prefetch_count": 0,
  "prefetch_timely_count": 0,
  "host_to_hbm_load_ms": 0.0,
  "hidden_load_ms": 0.0,
  "exposed_stall_ms": 0.0,
  "phase_split_count": 0
}
```

## 11. Correctness 复现

每个性能实验前运行：

```bash
pytest tests/ut/moe_offload -q
pytest tests/ut/ops/test_fused_moe.py -q
```

输出一致性检查：

```bash
python benchmarks/sew_offload/check_outputs.py \
  --baseline artifacts/sew_offload/runs/baseline/outputs.jsonl \
  --candidate artifacts/sew_offload/runs/sew/outputs.jsonl
```

检查项：

- token ids 一致。
- 文本一致。
- 每层 expert id checksum 一致。
- phase split 后 token 写回位置一致。

## 12. 结果整理

生成表格：

```bash
python benchmarks/sew_offload/collect_results.py \
  --runs artifacts/sew_offload/runs \
  --output artifacts/sew_offload/tables/summary.csv
```

建议表格字段：

```text
model
workload
num_slots
policy
max_phases
throughput
tpot_ms
itl_ms
p99_ms
hbm_gb
slot_hit_rate
exposed_stall_ms
overlap_ratio
```

## 13. 故障排查

### 13.1 无法 import torch_npu

现象：

```text
ModuleNotFoundError: No module named 'torch_npu'
```

处理：

- 进入 vLLM Ascend 开发容器。
- 或激活包含 torch-npu 的 Python 环境。
- 重新执行软件环境检查。

### 13.2 单卡 OOM

处理：

- 降低 `gpu_memory_utilization`。
- 降低 batch size。
- 降低 `max_model_len`。
- 降低 resident slot 数。
- 记录 OOM 配置，不静默跳过。

### 13.3 phase split 变慢

处理：

- 将 `VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1`。
- 检查 hit compute time 是否过短。
- 检查 split overhead。
- 在 CostModel 中提高 useful overlap threshold。

### 13.4 prefetch 未及时完成

处理：

- 增加 cross-step prefetch。
- 提高 slot budget。
- 改用 token-count weighted policy。
- 检查 host NUMA 亲和性。

### 13.5 输出不一致

处理：

- 禁用 phase split。
- 禁用 replacement。
- 只保留 sync fixed slot。
- 检查 `expert_id -> slot_id` checksum。
- 检查 token output 写回 index。

## 14. 复现记录模板

每次实验记录：

```markdown
# SEW-Offload Run Record

Date:
Commit:
NPU id:
Model:
Workload:
Environment:

Config:
- num_slots:
- policy:
- max_phases:
- trace_only:

Results:
- throughput:
- TPOT:
- ITL:
- P99:
- slot_hit_rate:
- exposed_stall:
- overlap_ratio:

Notes:
- correctness:
- failures:
- anomalies:
```
