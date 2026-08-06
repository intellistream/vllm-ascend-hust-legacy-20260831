# First nano-PEARL Ascend prototype

This directory contains the second process-level smoke test:

- Draft worker: physical Ascend card 4, Qwen3-0.6B
- Target worker: physical Ascend card 5, Qwen3-8B
- `enforce_eager=True`
- one independent vLLM V1 engine per worker
- Unix domain socket control channel
- no cross-card HCCL communication
- token-ID Draft -> Target handoff
- greedy `accepted_len` prototype using `prompt_logprobs=1`
- no persistent KV-cache rollback yet

Run from the vLLM-Ascend-HUST checkout:

```bash
python pearl_coordinator.py \
  --draft-model /data/shared-models/Qwen3-0.6B \
  --target-model /data/shared-models/Qwen3-8B \
  --draft-device 4 \
  --target-device 5 \
  --max-model-len 1024 \
  --gamma 4 \
  --probe-max-tokens 1 \
  --rounds 2 \
  --prompt "The capital of France is" \
  --prompt "2+2="
```

The coordinator launches both workers, waits for both models to finish
loading, sends `ping`, tokenizes the prompt on CPU, asks Draft for four token
IDs, sends `prefix_token_ids + draft_token_ids` to Target, and uses Target's
prompt log probabilities to calculate a greedy `accepted_len` and bonus token.
The accepted tokens are appended to the next round's CPU-side prefix before
the next Draft request.  At the end of each prompt, the coordinator compares
the committed speculative token IDs with a normal Target-only greedy request.

This is still not the final nano-PEARL performance path: every request
recomputes the complete prefix and transfers prompt logprob metadata to the
host.  The next implementation stage must move the verification into the
vLLM rejection-sampler path and preserve committed KV blocks in each worker.
