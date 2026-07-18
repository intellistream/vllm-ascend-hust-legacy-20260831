# ACLGraph Decode + CPU KV Prototypes (2026-07-18)

## Why these experiments exist

The isolated `mapped-host gather` versus `span-copy` result was encouraging,
but it did not model the serving pipeline we actually care about. On this NPU
stack, an AI Core gather and decode attention do not provide useful concurrent
compute, while H2D copies on a copy stream may overlap an ACLGraph replay.
The production-shaped comparison is therefore:

```text
graph decode || span H2D
versus
mapped-host gather -> graph decode
```

We also wanted to test three follow-up ideas independently:

1. capture mapped gather together with decode to remove the eager launch edge;
2. let one consumer read a mixture of host and device KV directly;
3. promote host KV into device memory on first use, then reuse it there.

The work below consists of decision prototypes and microbenchmarks, not a
serving performance claim.

## Artifact correction before final measurement

An early graph-capture run and the first overlap run accidentally used the
day's rejected `TQueBind` operator package. The `libcust_opapi.so` hash alone
could not distinguish the device kernels because the payload kernels live in
the vendor package's `op_impl` tree.

Those numbers were discarded. The retained two-queue, `BUFFER_NUM=2` source
was rebuilt and all final gather numbers below use it:

- kernel source SHA256:
  `2a514335ed2423eebaaa98db2029b252358ac9ea0289659355909da6d4ab8fa5`;
- all six compiled kernel-object hashes are recorded in the graph-capture
  manifests and in
  `branch_development_notes/work/aclgraph-kv-restore-overlap-production-twoqueue-20260718/OPERATOR_SHA256SUMS`;
- CANN 9.0.0, fp16 gather, Ascend 910B2.

This provenance check is part of the result: future comparisons must hash the
kernel source and `op_impl` objects, not only the ACLNN host library.

## P0/P1: graph decode versus the two restore pipelines

[`tools/benchmark_aclgraph_kv_restore_overlap.py`](../../tools/benchmark_aclgraph_kv_restore_overlap.py)
captures real Q/K/V attention math with fixed buffers. It is model-independent,
but preserves the scheduling question. The tested restore is two KV parts,
256 blocks, 16 KiB per block: 8 MiB total. The same logical mapping is emitted
as 256 one-block spans, 32 eight-block spans, or one 256-block span.

The span backend runs on a copy stream alongside graph replay. The mapped
backend is measured with the connector-shaped restore-before-decode barrier.
An additional `graph || mapped` diagnostic confirms that putting mapped gather
on a second stream does not create useful overlap on this stack.

Physical NPU 0, 10 warmups and 30 single-step samples:

| context | mapping | graph `||` span | mapped `->` graph | mapped pipeline gain |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 256 spans | 22.878 ms | 0.848 ms | +96.3% |
| 512 | 32 spans | 3.189 ms | 0.844 ms | +73.5% |
| 512 | 1 span | 0.700 ms | 0.842 ms | **-20.2%** |
| 2048 | 256 spans | 22.958 ms | 1.567 ms | +93.2% |
| 2048 | 32 spans | 4.337 ms | 1.626 ms | +62.5% |
| 2048 | 1 span | 1.321 ms | 1.541 ms | **-16.6%** |

For the one-span cases, mapped gather is still faster in isolation (roughly
0.60--0.64 ms versus 0.71--0.74 ms). Nevertheless, enough of the span H2D is
hidden under graph replay that the complete span pipeline wins. For fragmented
mappings, the many Python-issued copy operations dominate and mapped gather
wins decisively even after span-copy receives its overlap opportunity.

**Decision:** reject a global “mapped gather always wins” policy. Preserve
span-copy for a small number of long spans and prefer mapped gather for
fragmented mappings. The final threshold must come from real connector traces,
because the crossover depends on span count, span length, payload and decode
shape.

Raw output:

`branch_development_notes/work/aclgraph-kv-restore-overlap-production-twoqueue-20260718/`

## P2a: capture gather together with decode

[`tools/benchmark_kv_gather_graph_capture.py`](../../tools/benchmark_kv_gather_graph_capture.py)
keeps one mapped host arena, ID buffers, destination and surrogate inputs at
fixed addresses. Changing the contents of the fixed `src_block_ids` buffer
between replays passed full gather and consumer correctness checks.

Physical NPU 1, 50 samples:

| payload | graph only | eager mapped + graph | captured mapped + graph | capture saving |
| ---: | ---: | ---: | ---: | ---: |
| 64 KiB | 0.01746 ms | 0.06512 ms | 0.02299 ms | 64.70% |
| 2 MiB | 0.08913 ms | 0.18443 ms | 0.18088 ms | 1.92% |
| 8 MiB | 0.08749 ms | 0.45116 ms | 0.44996 ms | 0.27% |

Capture removes most of the wrapper/launch boundary for a tiny payload. It
does not remove the mapped-host traffic or overlap the dependent gather and
decode; at 2--8 MiB the saving is only 0.001--0.004 ms.

There is also a correctness constraint: with the normal
`TASK_QUEUE_ENABLE=1`, the current `OpCommand` custom handler is deferred until
after the graph-capture context, so replay silently omits the gather. The
prototype only captures correctly with `TASK_QUEUE_ENABLE=0` set before
importing `torch_npu`. Globally disabling the task queue is not a production
proposal; a capture-aware submission path is required.

Detailed note and provenance:
[`mapped_host_gather_graph_capture_20260718.md`](mapped_host_gather_graph_capture_20260718.md).

## P2b: mixed host/device consumer and promotion

The research-only operator under
[`csrc/kv_cache_hybrid_attention_proto`](../../csrc/kv_cache_hybrid_attention_proto/)
selects mapped-host or device GM independently for each block, computes a
tiled fp32 `K dot Q`, and can simultaneously write the resolved block to a
compact device promotion cache. The benchmark compares:

- `device`: every block is already device-resident;
- `permanent_hybrid`: host blocks are read through the mapping every token;
- `promote_first_use`: token zero reads mixed sources and promotes them;
  later tokens read the promoted device cache.

The physical-NPU smoke passed both score comparison against a CPU reference
(`rtol=2e-4`, `atol=2e-3`) and exact promotion-payload comparison. This proves
that mixed mapped-host/device addressing and promotion can live in one AI Core
consumer.

The complete measured package is retained under
`branch_development_notes/work/hybrid-kv-promotion/op-package/`. Its hashes
are:

- kernel source: `cc9472e782a274b3923dbff6bc656fa49fa983744707fa5b331b043c839f1a8c`;
- normal kernel object: `9bb0ced8c7218de1b758bb56bf007773031978e3567d44b276642db864b74f10`;
- relocatable kernel object: `94b8a2256e653d6fc264d545ad3a296b502aa58a70347fcbf757478014a6931e`;
- Torch extension: `c0a8649d2f75085f587889499dc5d3bad5e86aa357325c91d54a8adbbd84971a`;
- ACLNN host library: `de903f69654e3ae336b030710fa85c42aaab438513d75a053672a05fed3967f5`.

The retained minimal matrix used 64 blocks x 4 KiB, host fractions 50% and
100%, and 1/2/4/8-token sequences. It used only one warmup and three samples,
so its timings are deliberately treated as exploratory:

| host fraction | tokens | device | permanent hybrid | promote first use |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 1 | 0.12985 ms | 0.14029 ms | 0.15037 ms |
| 50% | 2 | 0.20405 ms | 0.20969 ms | 0.20257 ms |
| 50% | 4 | 0.32477 ms | 0.30889 ms | 0.33204 ms |
| 50% | 8 | 0.51856 ms | 0.46776 ms | 0.46417 ms |
| 100% | 1 | 0.13044 ms | 0.12669 ms | 0.12325 ms |
| 100% | 2 | 0.19067 ms | 0.17946 ms | 0.18517 ms |
| 100% | 4 | 0.27608 ms | 0.28907 ms | 0.29571 ms |
| 100% | 8 | 0.44795 ms | 0.47447 ms | 0.46407 ms |

The non-monotonic result (including 50% host sometimes appearing faster than
all-device) is larger than the claimed policy differences. The script's
mechanical first-win points therefore are **not** reliable promotion
crossovers. At 100% host and eight tokens, permanent hybrid is 5.92% slower
than device and promotion is 2.19% faster than permanent hybrid, but three
samples are not enough to make that a policy decision.

This kernel contains K/Q consumption and reduction only; it omits softmax and
V aggregation, and its tile pipeline is correctness-first. It does not prove
that a compute-bound production attention kernel hides mapped-host latency.
The prototype validates feasibility, not profitability. A decision run needs
at least 10 warmups, 100+ samples, repeated orderings, larger block/payload
sweeps and an arithmetic-intensity sweep or a real attention kernel.

Raw output:

`branch_development_notes/work/hybrid-kv-promotion/matrix/results.{csv,json}`

## P0b: current connector hooks under full ACLGraph replay

The connector starts layer zero outside model forward. Each Python attention
hook synchronizes that load and launches the next layer. A full graph replay
does not call the Python model again.

The four-layer device reproduction changed all host values, poisoned every
device destination, called the normal per-step `start_load_kv`, and even waited
for layer zero before replay. For both `TASK_QUEUE_ENABLE=0` and `1`:

- the initial graph baseline replay was correct (`46`);
- replay executed zero Python wait hooks and zero next-layer load calls;
- outputs were exactly `-2900` and `-2800`, rather than `406` and `806`:
  only the new layer zero plus three poisoned layers were consumed.

Thus Python layerwise loads are neither rerun nor automatically captured from
the external load stream. Source audit also found no CPU-offload-specific eager
guard in the full-graph dispatch path. This is a production correctness risk,
not just a performance detail. A real engine trace still needs to establish
which graph mode/backend a concrete request selects.

Detailed evidence:
[`aclgraph_cpu_offload_hook_semantics_20260718.md`](aclgraph_cpu_offload_hook_semantics_20260718.md).

## Overall decision

1. **Continue mapped gather, but only as an adaptive fragmented-mapping
   backend.** The graph-overlap experiment is the governing microbenchmark.
2. **Do not expect graph capture to make a large gather free.** It is launch
   cleanup for small payloads and currently needs capture-aware submission.
3. **Keep mixed attention and promotion as research prototypes.** Their data
   paths are correct on hardware, but the present timing matrix cannot choose
   permanent hybrid versus promotion.
4. **Fix or gate CPU restore under full ACLGraph before serving A/B tests.**
   Forcing restore steps eager/piecewise is the smallest correctness fallback;
   graph-owned restore scheduling is the more ambitious design.
5. Next serving evidence should record per request: graph mode, attention
   backend, span count/length distribution, logical bytes, selected restore
   backend, layer hook counts, and end-to-end decode latency.

## Validation summary

- production two-queue gather direct smoke: passed;
- P0/P1 six-case NPU overlap matrix and output validation: passed;
- P2a dynamic-ID graph replay and three payload measurements: passed;
- P2b mixed-source score and exact promotion correctness: passed;
- connector replay controls under both task-queue modes: passed;
- focused benchmark/probe unit tests after the final hybrid score-padding
  change: passed (`27 passed`).
