# nano-PEARL on Ascend

## Status

The native runtime is an Ascend port of the functionality implemented by
upstream nano-PEARL at commit `7d020b6`. Draft and target ranks share one HCCL
world, own independent persistent KV caches, and execute the upstream
`pre-verify -> gamma draft -> target verify -> rollback` pipeline.

The primary API mirrors upstream nano-PEARL:

```python
from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams
```

`PEARLEngine` starts all draft and target workers with `spawn`, creates the
HCCL topology, accepts queued string or token-ID prompts, and exposes
`generate`, `AR_generate`, and `bench_generate`.

The repository also retains an OpenAI-compatible bridge. That bridge uses
vLLM's speculative scheduler and a separate draft service; it is useful for
serving, but is not the native cross-group PEARL pipeline.

## Feature Parity

| Upstream implemented feature | Ascend implementation |
| --- | --- |
| Qwen2, Qwen3, and Llama | Native TP models and Hugging Face safetensors loader |
| Independent draft and target TP | Disjoint HCCL model groups in one world |
| Dynamic TP 3, 6, and 7 | Upstream-compatible zero padding for heads, KV heads, MLP, and vocabulary |
| Static request scheduling | Parent request queue, bounded by `max_num_seqs` and `max_num_batched_tokens` |
| Paged KV cache and prefix reuse | Shared, lazily allocated CANN page pool with full-page prefix caching |
| CUDA Graph | `torch.npu.NPUGraph` plus replay-time CANN FIA/PA graph-task metadata updates |
| FlashAttention and Triton KV write | CANN fused-infer attention, paged attention, and `_npu_reshape_and_cache` |
| Target temperature sampling | Exponential-race sampling and upstream stochastic verification rule |
| Per-request stopping | `max_tokens`, `ignore_eos`, and matching EOS validation |
| Automatic gamma | Startup profiles for batch buckets 1, 2, 4, 8, 16, and 32 |
| AR and fixed-step benchmarks | `AR_generate` and `bench_generate` |
| MAT reporting | Per-request `num_acc_tokens`, including target correction tokens |

The target logits are cropped to the draft vocabulary before sampling or
comparison. This supports pairs such as Qwen2.5-0.5B-Instruct with a 151,936
entry model vocabulary and Qwen2.5-14B-Instruct with 152,064 entries, provided
the draft token IDs are an unchanged prefix of the target mapping.

Ascend uses 128-token KV pages. Therefore `kvcache_block_size` is present in
the compatible configuration surface but must be 128; this is the native
vLLM-Ascend paged-attention constraint rather than the upstream CUDA default.
`gpu_memory_utilization` or an explicit `num_kvcache_blocks` controls the
shared page-pool capacity.
`max_aclgraph_entries` defaults to 16 and bounds both graph-resident workspaces
and total capture attempts; unseen low-frequency shapes use eager execution
after the limit is reached, including when earlier captures were rejected.
FIA graphs are admitted only for the full request batch with uniform per-request
query lengths. Mixed verification shapes and shrinking batch tails execute
eagerly without consuming the capture budget.

`elapsed_time` follows upstream benchmark semantics and includes model prefill
plus generation. ACLGraph capture is warmed outside the reported interval. The
first real replay of each graph shape is checked against eager execution and
that one-time safety check is included in generation time. Native result
metadata also separates `prefill_elapsed_seconds` and
`decode_elapsed_seconds`.

## Native API

```python
from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams


def main():
    config = PEARLConfig(
        draft_model_path="/data/shared-models/Qwen2.5-0.5B-Instruct",
        target_model_path="/data/shared-models/Qwen2.5-14B-Instruct",
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=2,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        gamma=4,
    )
    params = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=False)
    with PEARLEngine(config) as engine:
        engine.add_request("What is 2 + 2?", params)
        output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
        print(output_text, num_tokens, num_acc_tokens, elapsed_time)


if __name__ == "__main__":
    main()
```

The `__main__` guard is required by Python's `spawn` multiprocessing mode. The
controller sets an automatic HCCL NPU socket-port range so it can coexist with
other HCCL jobs in the same container. Worker initialization and generation use
a 300-second deadline by default; set `worker_timeout_seconds` in `PEARLConfig`
or `--worker-timeout-seconds` in the example CLI to change it.

`PEARLConfig.draft_config`, `target_config`, `eos`, and `world_size`, plus
`PEARLEngine.log()`, retain the corresponding upstream compatibility surface.

The runnable version is
`examples/offline_inference_nano_pearl.py`. For the validated Qwen2.5 pair:

```bash
cd /root/data/vllm-ascend-hust
ASCEND_RT_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. \
  /root/miniconda3/envs/vllm-hust-dev/bin/python \
  examples/offline_inference_nano_pearl.py \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 4 \
  --max-model-len 1024 --max-num-seqs 32 --max-tokens 64 \
  'What is 2 + 2?'
```

Set `--mode target-ar` for target-only autoregressive generation or
`--mode bench --num-pearl-steps 100` for the upstream fixed-step benchmark.
Set `--gamma -1` to profile and select gamma automatically. Positive
`--temperature` values use target sampling; the draft remains greedy, matching
the current upstream implementation.

## Direct Worker CLI

The lower-level runtime can still be launched under `torchrun`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  -m vllm_ascend.spec_decode.pearl.native_engine \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 4 \
  --max-model-len 1024 --max-tokens 64 \
  --prompt 'What is 2 + 2?'
```

The direct CLI also accepts GSM8K parquet input through `--gsm8k`, and supports
`--temperature`, `--ignore-eos`, `--seed`, prefix-cache control, ACLGraph
control, `--gpu-memory-utilization`, an explicit `--num-kvcache-blocks`,
batched PEARL, and target AR mode.

The JSON summary reports both raw draft-token acceptance and upstream MAT.
`aggregate_acceptance_rate` is accepted draft tokens divided by verified draft
tokens. `aggregate_mat` is the mean of each request's `num_acc_tokens` segments,
matching upstream benchmark scripts; these metrics are not interchangeable.
Per-request metadata also reports `aclgraph_captures`, `aclgraph_replays`, and
`aclgraph_failed_captures`, plus `aclgraph_capture_attempts` and
`aclgraph_capacity_fallbacks`, plus `aclgraph_shape_fallbacks`. A failed
first-replay check disables that graph shape and falls back to eager execution
instead of returning unchecked tokens. The controller aggregates these
counters across every draft and target worker.

For a target-only comparison against production vLLM-Ascend, use the two
benchmark entry points below. Both apply the same chat template and emit the
first prompt/output token IDs so input and greedy-output parity can be checked:

```bash
python examples/benchmark_nano_pearl_target_only.py \
  --model /data/shared-models/Qwen3-0.6B \
  --batch-sizes 1 8 32 --max-tokens 32 \
  --prompt 'What is 2 + 2?'

python examples/benchmark_nano_pearl_native_target_only.py \
  --draft-model /data/shared-models/Qwen3-0.6B \
  --target-model /data/shared-models/Qwen3-0.6B \
  --batch-sizes 1 8 32 --max-tokens 32 \
  --prompt 'What is 2 + 2?'

python examples/benchmark_nano_pearl_speculative.py \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-tp-size 1 --target-tp-size 2 --gamma 4 \
  --batch-sizes 1 8 32 --max-tokens 32 \
  --gsm8k /data/datasets/gsm8k/test.parquet
```

## Ascend Runtime Mapping

Groups are created in a globally fixed order:

```text
draft group:          [0, ..., draft_tp - 1]
target group:         [draft_tp, ..., world_size - 1]
verification group:   [draft leader, all target ranks]
```

Packed prefill and target-only autoregressive decode use CANN paged attention.
Layers with the same PA operator shape share one graph workspace instead of
retaining one scratch allocation per transformer layer.
PEARL draft decode and packed target verification use the same TND fused-infer
attention path selected by vLLM-Ascend for speculative decoding. For
`gamma <= 16`, these speculative calls run in `torch.npu.NPUGraph`; their
request-level query boundaries, KV lengths, page tables, and shared FIA
workspace are rebound before replay. Larger gamma values exceed the CANN TND
speculative-query limit and use the correct eager paged-attention fallback.
Dense SDPA exists only as the CPU test fallback.

## OpenAI-Compatible Bridge

The bridge is registered through vLLM's
`speculative_config.method="custom_class"` interface. Its engines cannot form
one cross-engine HCCL group, so it transports prompt and token IDs over a
mode-`0600` Unix socket while the target keeps vLLM-Ascend TP, paged attention,
verification, and ACLGraph:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python \
  -m vllm_ascend.spec_decode.pearl.launcher \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --draft-devices 0,1 --draft-tensor-parallel-size 2 \
  --target-devices 2,3 --num-speculative-tokens 4 \
  --draft-llm-kwargs '{"dtype":"bfloat16","gpu_memory_utilization":0.70}' \
  -- \
  /data/shared-models/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16 --port 8000
```

The bridge is greedy because it transfers draft token IDs rather than draft
probability distributions. Use the native API for upstream PEARL stochastic
target verification.

## Upstream TODO Boundary

The following are not implemented in upstream nano-PEARL commit `7d020b6` and
are intentionally not represented as completed migration features:

- non-zero draft-model temperature;
- continuous batching and chunked prefill;
- context-adaptive gamma changes during generation;
- PEARL-2 draft-model training or distillation.

The native runtime, like upstream, is text-only and does not add vLLM features
such as multimodal input, LoRA routing, or structured-output constraints.

## Verification

CPU migration tests:

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q \
  tests/ut/spec_decode/test_pearl.py \
  tests/ut/spec_decode/test_pearl_bridge.py \
  tests/ut/spec_decode/test_pearl_native.py \
  tests/ut/spec_decode/test_pearl_vocab.py
```

HCCL protocol smoke test:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m vllm_ascend.spec_decode.pearl.smoke
```

Validated NPU paths include Qwen2.5 heterogeneous-vocabulary TP1+TP2, Qwen3
greedy and positive-temperature PEARL, Qwen3 target AR, fixed-step batched
benchmark generation, automatic gamma, Qwen3 dynamic TP3+TP3, and Llama PEARL
weight loading and generation. The speculative FIA ACLGraph path has an
identical-model oracle at batch sizes 1, 8, and 32: Qwen3-0.6B reaches at least
99.62% raw acceptance with zero failed graph captures.
