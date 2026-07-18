# Mapped-host Gather ACLGraph Capture Prototype (2026-07-18)

## Question

Can the existing `kv_cache_block_gather` task be captured with a decode-like
ACLGraph, so steady-state replay avoids the separate eager custom-op boundary?

This is deliberately an independent prototype. It does not change the CPU
connector or claim to reproduce model decode latency.

## Prototype

[`tools/benchmark_kv_gather_graph_capture.py`](../../tools/benchmark_kv_gather_graph_capture.py)
keeps these objects alive and at fixed addresses for the full experiment:

- one CPU host arena registered once with `ACL_HOST_REGISTER_MAPPED`;
- one fixed-shape NPU `src_block_ids` buffer;
- one fixed-shape NPU `dst_block_ids` buffer;
- one fixed-shape NPU KV destination;
- the decode surrogate's weights and ACLGraph outputs.

It captures a stack of fp16 matmuls as a decode dependency surrogate and
measures four modes:

1. `graph_only`;
2. `mapped_only`;
3. eager `mapped_then_graph`;
4. `graph_capture_gather_decode`, where gather and its consumer are in one
   ACLGraph.

The fixed source-ID buffer is updated in place with a second random mapping.
Replay is then checked against both the expected gathered blocks and a separate
decode-only graph. This confirms that replay reads the current ID contents and
that the decode consumer observes the captured gather; it is not just replaying
capture-time data.

## Capture Semantics Finding

The current Torch binding submits ACLNN through an `OpCommand` custom handler.
With the container's normal `TASK_QUEUE_ENABLE=1`, the handler is deferred
until after the `torch.npu.graph(...)` capture context has ended. Capture does
not raise an error, and the gather executes during the capture call, but graph
replay contains only the decode consumer. Resetting the destination before
replay exposes this immediately as a full output mismatch.

With `TASK_QUEUE_ENABLE=0` set **before importing `torch_npu`**, the handler
submits the gather while stream capture is active. Capture/replay then passes:

- complete destination-block comparison;
- a second source-ID mapping in the same fixed-address ID tensor;
- captured decode output versus eager-gather + decode-only-graph output.

Therefore, the kernel/ACLNN task is capturable, but the current asynchronous
Torch submission path is not production-capture-safe. A production version
would need a capture-aware direct submission path (or an equivalent task-queue
integration); globally disabling the task queue is only a research control.

## Ascend 910B2 Results

Physical NPU 1 was exposed as `npu:0` in a clean CANN 9.0.0 container. Results
use 50 wall-time samples with device synchronization around batches of graph
replays. Registration and graph capture are excluded from steady state.

| logical gather | graph only | mapped only | mapped then graph | captured gather + graph | capture saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KiB | 0.01746 ms | 0.03729 ms | 0.06512 ms | 0.02299 ms | 0.04214 ms (64.70%) |
| 2 MiB | 0.08913 ms | 0.09676 ms | 0.18443 ms | 0.18088 ms | 0.00355 ms (1.92%) |
| 8 MiB | 0.08749 ms | 0.36496 ms | 0.45116 ms | 0.44996 ms | 0.00121 ms (0.27%) |

Raw retained results are under the ignored local directory:

`branch_development_notes/work/kv-gather-graph-capture-20260718/production-twoqueue/`

The 64 KiB case used 50 replays per sample; the 2 MiB and 8 MiB cases used 20.
All final numbers use the retained production two-queue, two-buffer kernel, not
the slower `TQueBind` experiment.

### Operator provenance

The complete vendor package is retained at:

```text
branch_development_notes/work/kv-gather-tquebind-20260718-173553/source/
  csrc/build/_CPack_Packages/Linux/External/
  cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
```

It was rebuilt from the current production header. The build log is
`branch_development_notes/work/kv-gather-graph-capture-twoqueue-build.log`.
Every result manifest records the absolute vendor path and these hashes:

- kernel source: `2a514335ed2423eebaaa98db2029b252358ac9ea0289659355909da6d4ab8fa5`;
- `163e...o`: `811169dfdeb30b878338e9ac5ab17a0735ec18c5f4c3febefc699ec5665c6f94`;
- `163e...relocatable.o`: `67450ade99fd4f18aadaf312525b962f814bb171a85cc78ce5ad10387ed5fb6d`;
- `2aeb...o`: `7e0efa49d0c3acebe06fd05a80b86dc110c6cf7b8621c47323116d71bc339c00`;
- `2aeb...relocatable.o`: `95c4a1a122e798f961f2f95334e1e63cfd6881870526fca3ff84312748f577ca`;
- `5193...o`: `a870476e4c978b896492c6b0d71a1d4edd05f136da39f539b743e79e37e4f8eb`;
- `5193...relocatable.o`: `d4aed45a722ae97299a18210428784613b50224d82c7b9ef2376deb2cd7ca344`.

The retained build container is `graph-gather-twoqueue-build`; the retained
benchmark container is `graph-gather-twoqueue-bench`.

## Interpretation

ACLGraph capture removes most of the eager wrapper/dispatch boundary for very
small transfers. This is why the 64 KiB case is dramatically better than
`mapped_then_graph`.

For production-sized payloads, capture does not overlap gather with decode and
does not remove the mapped-host read itself. The captured graph remains a
serial gather followed by decode. At 2 MiB and 8 MiB, only about 0.003 ms is
saved; payload movement dominates and captured execution is still respectively
0.09175 ms and 0.36246 ms above `graph_only`.

This rules out graph capture alone as the mechanism that makes a large mapped
gather free. It is useful as launch-overhead cleanup for small transfers, but
the important serving comparison remains:

```text
captured mapped gather + graph decode (serial)
versus
memcpy overlapped with graph decode
```

## Limitations and Production Debt

- The decode workload is a dependency-preserving matmul surrogate, not real
  attention or a serving benchmark.
- The experiment covers one 910B2 and fp16.
- `TASK_QUEUE_ENABLE=0` is required by the existing `OpCommand` adapter and is
  not proposed as a production setting.
- ACLNN currently reports a 16 MiB workspace on every gather. The device kernel
  does not dereference it, so replay survives the binding's temporary workspace
  lifetime. If the kernel starts using workspace, graph capture requires a
  fixed long-lived workspace allocation.
- Host arena lifetime is graph lifetime. Unregistering or replacing it while a
  graph can replay would leave the graph with a stale mapped device address.
- Dynamic block count is not tested. The prototype intentionally fixes ID
  tensor shape and only changes its contents; production needs capture buckets
  or a fixed maximum plus valid-count contract.

## Validation

- benchmark mapping/shape/report/provenance unit tests: passed (`8 passed`);
- direct NPU capture/replay correctness: passed with task queue disabled;
- dynamic fixed-address source-ID replay: passed;
- 64 KiB, 2 MiB, and 8 MiB measured runs: passed;
- `TASK_QUEUE_ENABLE=1` negative control: replay correctness failed as expected.
