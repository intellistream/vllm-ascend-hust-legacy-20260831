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

SEW-Offload 使用以下环境变量：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=0
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=2
export VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS=4096
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
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=lru
```

prefetch + phased execution：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=8
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=2
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=1
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

MVP-B 提供 `tools/sew_offload/collect_moe_trace.py`，它会强制启用
trace-only 模式，运行一个小 workload，并在结束后把内存 trace 导出成 JSONL。

准备 synthetic smoke manifest：

```bash
PYTHON=${PYTHON:-python}
ASCEND_RT_VISIBLE_DEVICES=4 \
$PYTHON tools/sew_offload/collect_moe_trace.py \
  --config docs/sew-offload/benchmark_config.yaml \
  --output artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl \
  --manifest artifacts/sew_offload/traces/qwen3_30b_a3b_smoke_requests.jsonl \
  --prepare-smoke-manifest \
  --prepare-only \
  --buckets short_chat \
  --smoke-requests-per-bucket 1
```

采集 trace：

```bash
PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-30B-A3B}
ASCEND_RT_VISIBLE_DEVICES=4 \
$PYTHON tools/sew_offload/collect_moe_trace.py \
  --config docs/sew-offload/benchmark_config.yaml \
  --model "$MODEL_PATH" \
  --output artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl \
  --manifest artifacts/sew_offload/traces/qwen3_30b_a3b_smoke_requests.jsonl \
  --buckets short_chat \
  --max-requests 1 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-mb 512
```

输出格式：

```json
{"active_experts": [0, 7, 31], "expert_token_counts": {"0": 1, "7": 3, "31": 2}, "layer_id": 3, "mode": "decode", "num_experts": 128, "num_tokens": 2, "step_id": 11, "top_k": 8}
```

需要保存：

```text
artifacts/sew_offload/traces/
```

## 8. Offline simulator 复现

运行 slot policy simulator：

```bash
PYTHON=${PYTHON:-python}
$PYTHON tools/sew_offload/simulate_expert_slots.py \
  --trace artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl \
  --num-slots 8 \
  --policy lru \
  --expert-bytes 14680064 \
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
for policy in lru sticky_layer_lru; do
  PYTHON=${PYTHON:-python}
  $PYTHON tools/sew_offload/simulate_expert_slots.py \
    --trace artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl \
    --num-slots 8 \
    --policy "${policy}" \
    --output "artifacts/sew_offload/sim/qwen3_30b_a3b_slots8_${policy}.json"
done
```

对比 slot budget：

```bash
PYTHON=${PYTHON:-python}
$PYTHON tools/sew_offload/simulate_expert_slots.py \
  --trace artifacts/sew_offload/traces/qwen3_30b_a3b_smoke.jsonl \
  --slot-range 8:64:8 \
  --policy lru \
  --expert-bytes 14680064 \
  --output artifacts/sew_offload/sim/qwen3_30b_a3b_slot_sweep_lru.json
```

`recommended_num_slots` 是达到最低 `host_to_hbm_bytes` / `miss_count` 的最小
slot 数；`sweep` 数组保留每个 slot budget 的完整 replay summary。

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

先跑最小 fixed-slot correctness smoke。这个命令不是正式 benchmark，只用于确认
MVP-D 窄路径是否能越过 native prefetch 曾经遇到的 CPU/NPU tensor mixing 失败点。

准备 synthetic smoke manifest：

```bash
PYTHON=${PYTHON:-python}
ASCEND_RT_VISIBLE_DEVICES=4 \
$PYTHON tools/sew_offload/run_fixed_slot_smoke.py \
  --output-dir artifacts/sew_offload/runs/fixed_slot_smoke_prepare \
  --manifest artifacts/sew_offload/traces/qwen3_30b_a3b_fixed_slot_smoke_requests.jsonl \
  --prepare-smoke-manifest \
  --prepare-only \
  --buckets short_chat \
  --smoke-requests-per-bucket 1
```

运行 fixed-slot sync smoke：

```bash
PYTHON=${PYTHON:-python}
ASCEND_RT_VISIBLE_DEVICES=4 \
$PYTHON tools/sew_offload/run_fixed_slot_smoke.py \
  --mode fixed_slot_sync \
  --output-dir artifacts/sew_offload/runs/fixed_slot_sync_smoke_slots8_inline \
  --inline-prompt "Hello" \
  --inline-max-output-tokens 1 \
  --num-slots 8 \
  --offload-backend prefetch \
  --offload-group-size 4 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --offload-params experts \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-mb 512
```

结果查看：

```bash
cat artifacts/sew_offload/runs/fixed_slot_sync_smoke_slots8_inline/summary.json
```

MVP-D.5 correctness 对照要用独立进程分别运行 baseline 和 candidate，避免在同一
Python 进程中同时保留多个 vLLM/NPU runtime 状态。先运行 SEW 默认关闭 baseline：

```bash
PYTHON=${PYTHON:-python}
ASCEND_RT_VISIBLE_DEVICES=4 \
$PYTHON tools/sew_offload/run_fixed_slot_smoke.py \
  --mode no_offload \
  --output-dir artifacts/sew_offload/runs/no_offload_smoke_inline \
  --inline-prompt "Hello" \
  --inline-max-output-tokens 1 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-mb 512
```

再运行 fixed-slot candidate，要求 prompt、采样参数、max output tokens 与 baseline
一致。两次运行都会写出 `outputs.jsonl`，用严格 token id 对照检查：

```bash
PYTHON=${PYTHON:-python}
$PYTHON tools/sew_offload/compare_smoke_outputs.py \
  --baseline artifacts/sew_offload/runs/no_offload_smoke_inline/outputs.jsonl \
  --candidate artifacts/sew_offload/runs/fixed_slot_sync_smoke_slots8_inline/outputs.jsonl \
  --output artifacts/sew_offload/runs/fixed_slot_sync_smoke_slots8_inline/correctness_compare.json
```

如果对照失败，不允许通过 clamp/drop/remap 非法 expert id 来掩盖差异；应回到
router logits、`log2phy`、`physical_expert_count`、slot tensor layout 或 runner
生命周期排查根因。

更长输出或多 prompt correctness smoke 可使用 JSONL inline prompt 文件，仍然保持独立进程
baseline/candidate 对照。示例：

```bash
mkdir -p artifacts/sew_offload/runs
cat > artifacts/sew_offload/runs/fixed_slot_correctness_prompts.jsonl <<'EOF'
{"request_id":"p0","prompt":"Hello","max_output_tokens":8}
{"request_id":"p1","prompt":"Hi","max_output_tokens":8}
EOF

ASCEND_RT_VISIBLE_DEVICES=4 \
${PYTHON:-python} tools/sew_offload/run_fixed_slot_smoke.py \
  --mode no_offload \
  --output-dir artifacts/sew_offload/runs/no_offload_smoke_2prompt_8tok \
  --inline-prompts-jsonl artifacts/sew_offload/runs/fixed_slot_correctness_prompts.jsonl \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-mb 512

ASCEND_RT_VISIBLE_DEVICES=4 \
${PYTHON:-python} tools/sew_offload/run_fixed_slot_smoke.py \
  --mode fixed_slot_sync \
  --output-dir artifacts/sew_offload/runs/fixed_slot_sync_smoke_2prompt_8tok \
  --inline-prompts-jsonl artifacts/sew_offload/runs/fixed_slot_correctness_prompts.jsonl \
  --num-slots 8 \
  --offload-backend prefetch \
  --offload-group-size 4 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --offload-params experts \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-mb 512

${PYTHON:-python} tools/sew_offload/compare_smoke_outputs.py \
  --baseline artifacts/sew_offload/runs/no_offload_smoke_2prompt_8tok/outputs.jsonl \
  --candidate artifacts/sew_offload/runs/fixed_slot_sync_smoke_2prompt_8tok/outputs.jsonl \
  --output artifacts/sew_offload/runs/fixed_slot_sync_smoke_2prompt_8tok/correctness_compare.json
```

已验证的 `num_slots=8` 结果：

- `Hello` 与 `Hi` 两条短 prompt、每条 8 token：baseline
  `artifacts/sew_offload/runs/no_offload_smoke_20260601_2short_8tok/outputs.jsonl`
  与 candidate
  `artifacts/sew_offload/runs/fixed_slot_sync_smoke_20260601_2short_8tok/outputs.jsonl`
  严格 token id 一致，`matched=2`。
- 把第二条 prompt 换成 `Briefly explain mixture-of-experts models.` 时，baseline 成功，
  但 `num_slots=8` candidate 在 prefill 阶段 fail closed：`active expert working set
  size 46 exceeds num_slots=8`。这表示 active expert 并集超过 slot budget，不能通过
  drop、clamp 或把多个 logical expert 合并到同一 slot 来绕过。

如果使用 synthetic benchmark manifest 做 smoke，可加
`--override-max-output-tokens 8` 把 benchmark bucket 的正式输出长度临时收缩到 correctness
调试范围；这不改变 `docs/sew-offload/benchmark_config.yaml` 的正式实验定义。

### 9.2.1 Fixed-slot memory ledger

MVP-D fixed-slot sync 仍是 correctness prototype。它会保留原始 expert 参数，同时创建
CPU host store clone 和 NPU/CPU slot bank backing tensors；因此不能把现有 prefetch
backend 的模型驻留下降直接归因于 SEW 自身 HBM saving。先用离线账本估算 fixed-slot
容量成本：

```bash
PYTHON=${PYTHON:-python}
$PYTHON tools/sew_offload/estimate_fixed_slot_memory.py \
  --num-slots 8 \
  --output artifacts/sew_offload/runs/fixed_slot_memory_estimate_qwen3_30b_slots8.json

$PYTHON tools/sew_offload/estimate_fixed_slot_memory.py \
  --num-slots 64 \
  --output artifacts/sew_offload/runs/fixed_slot_memory_estimate_qwen3_30b_slots64.json
```

在默认 Qwen3-30B-A3B 估算下，`num_slots=8` 的 per-layer slot bank 约
`5.64 GB`，`num_slots=64` 约 `45.10 GB`。这说明直接把 slot budget 放大并不是
免费的 correctness 参数；释放/替换原始 full expert 参数前，不应把大 slot budget
作为默认 smoke 路径。

当前 release readiness 仍是只读 guard，不会释放或替换
`layer.w13_weight/layer.w2_weight`。`plan_original_weight_release(...)` 会先检查：
默认路径是否被调用方证明保留、目标 MoE 层是否都已注册、`HostExpertStore` 是否覆盖
每个 expected expert，以及 host bundle 的 shape/dtype/stride/CPU device 是否与注册时
的 post-processed layout 一致。任何 blocker 都应视为 fail closed；通过该 guard 也只表示
可以进入参数所有权转移设计，不表示已经产生 HBM saving。

如果 summary 记录 `status=failed` 且错误是 `backend device mismatch`，
说明 slot tensor 尚未被放置到 NPU；如果越过该检查后在 grouped MoE backend
失败，则继续记录真实 backend 错误栈，用于定位 layout 或 token dispatch 契约。

后续 serving benchmark 才使用同一配置扩展请求规模：

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
PYTHON=${PYTHON:-python}
$PYTHON tools/sew_offload/compare_smoke_outputs.py \
  --baseline artifacts/sew_offload/runs/baseline/outputs.jsonl \
  --candidate artifacts/sew_offload/runs/sew/outputs.jsonl \
  --output artifacts/sew_offload/runs/sew/correctness_compare.json
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
