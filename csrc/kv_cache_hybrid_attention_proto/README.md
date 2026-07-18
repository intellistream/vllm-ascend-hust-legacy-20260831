# Mixed host/device KV attention prototype

This research-only operator tests the dataflow behind hybrid decode attention
and promote-on-first-use.  It is intentionally separate from production
attention and CPU-offload connectors.

For every selected logical KV block it computes:

```text
source = source_kinds[i] ? mapped_host_pages[source_block_ids[i]]
                         : device_pages[source_block_ids[i]]
scores[i] = dot(source, query)
```

When `promote_flag[0] != 0`, the same resolved source is also copied to
`promoted_pages[i]`.  A benchmark can therefore use mixed host/device sources
for token zero and use the compact device-resident `promoted_pages` for all
later tokens.  Setting the flag to zero and retaining the mixed mapping models
permanent hybrid access.

The dot/reduction is the tiled K/Q consumption core of decode attention, but
the prototype deliberately omits softmax and value aggregation.  Its result is
useful for comparing mapped-host traffic and promotion amortization; it is not
a claim about complete attention latency.

The initial implementation is correctness-first.  Promotion uses a separate
UB queue to preserve the source before the in-place multiply, and only fp32 is
supported.  Both choices should be revisited if the experiment justifies a
production fused kernel.

Run the token sweep with:

```bash
python tools/benchmark_hybrid_kv_attention_promotion.py \
  --selected-blocks 128 --block-elems 1024 4096 \
  --host-fractions 0.25 0.5 1.0 --tokens 1 2 4 8 16 32
```

The benchmark performs CPU-reference score checks and exact promotion-payload
checks before timing `device`, `permanent_hybrid`, and `promote_first_use`.
