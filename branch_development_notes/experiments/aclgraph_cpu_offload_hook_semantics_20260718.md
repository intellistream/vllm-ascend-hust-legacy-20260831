# CPU Offload Layer Hooks under ACLGraph Replay (2026-07-18)

## Question

The current `CPUOffloadingConnector` starts layer 0 before model forward,
then uses each attention layer's Python `wait_for_layer_load` hook to wait for
that layer and launch the next one. Does that Python-driven pipeline execute on
every full ACLGraph decode replay, or does capture somehow preserve its
load-stream operations?

## Static call-chain audit

The relevant production ordering is:

```text
KVConnectorModelRunnerMixin._get_kv_connector_output
  bind_connector_metadata
  start_load_kv                 # outside self._model_forward
  yield
    Ascend model runner
      self._model_forward
        ACLGraphWrapper
          capture: self.runnable(...) executes Python model/layer hooks
          replay:  entry.aclgraph.replay() only
```

Evidence:

- upstream `kv_connector_model_runner_mixin.py` calls `start_load_kv` before
  yielding to the model;
- `model_runner_v1.py` enters that connector context around `_model_forward`;
- `ACLGraphWrapper` invokes `self.runnable` only while capturing a new graph;
  its steady-state branch calls only `entry.aclgraph.replay()`;
- `attention/utils.py` implements `wait_for_kv_layer_from_connector` as an
  ordinary Python connector call;
- `CPUOffloadingConnectorWorker.wait_for_layer_load` performs a host-side
  `load_stream.synchronize()`, increments `current_layer`, then calls
  `load_kv_layer` for the next layer;
- the cudagraph dispatch method contains no CPU-offload/KV-connector guard.
  Its caller passes the generic `model_config.enforce_eager` flag, not a flag
  derived from whether this step restores CPU KV.

Therefore a CPU-prefix restore step is **not automatically forced eager just
because it uses `CPUOffloadingConnector`**. Generic batch shape/backend rules
may still choose eager or piecewise execution, but there is no connector-
specific fallback in the audited path.

One additional suspicious detail is that MLA calls the connector wait hook only
under `if has_prefill`; SFA calls it unconditionally. This probe does not claim
which attention backend/request classification is selected by a complete
serving run, so that remains an end-to-end trace item.

## Device reproduction

[`tools/probe_aclgraph_kv_connector_hooks.py`](../../tools/probe_aclgraph_kv_connector_hooks.py)
reproduces the connector control flow with four layers:

1. stable pinned-host source per layer;
2. stable NPU destination per layer;
3. a dedicated load stream;
4. `start_load_kv` launches layer 0 outside the graph;
5. each Python layer hook synchronizes the current load and launches the next;
6. graph compute consumes every layer destination.

After capture, an initial graph replay without changing destinations validates
the captured compute graph. For the measured replays, all destinations are
poisoned, host values are changed in place, and only the normal external
`start_load_kv` is called. Layer 0 is explicitly synchronized before replay,
which gives the current design a favorable control. If the captured graph had
preserved the remaining layer loads, all new values would still be observed.

The tool emits machine-readable JSONL events and has a `--static-only` mode.

## NPU result

Tested on physical Ascend 910B2 NPU 3, exposed as logical `npu:0`, with
Torch/Torch-NPU 2.10.0. Both task-queue settings were tested independently.

| Control | Baseline graph replay | Python waits during each later replay | Python next-layer loads during replay | Output after poisoning |
| --- | --- | ---: | ---: | --- |
| `TASK_QUEUE_ENABLE=0` | correct (`46`) | 0 | 0 | `-2900`, then `-2800` |
| `TASK_QUEUE_ENABLE=1` | correct (`46`) | 0 | 0 | `-2900`, then `-2800` |

For the first measured replay the new layers should sum to `406`. The observed
`-2900` is exactly:

```text
new layer 0 (100) + three poisoned, never-loaded layers (-1000 each)
```

The second replay similarly produced `200 - 3000 = -2800`, rather than the
expected `806`.

This establishes for the minimal faithful reproduction that:

1. Python attention hooks execute during capture but do not rerun on full graph
   replay;
2. H2D copies launched by those hooks on the external load stream were not
   incorporated into the replayed graph;
3. only layer 0 is launched by the per-step `start_load_kv` call;
4. changing `TASK_QUEUE_ENABLE` does not rescue this connector-style pipeline.

## Conclusion and scope

The current Python layerwise connector protocol cannot be assumed to work in a
steady-state **full ACLGraph** replay. A correct integration needs one of:

- force CPU-prefix restore steps to an eager/piecewise path where Python hooks
  actually execute;
- move all per-layer scheduling outside the full graph and establish explicit
  device event dependencies;
- capture a graph-owned restore implementation whose source/destination
  addresses and dynamic metadata obey graph replay constraints; or
- fuse/promotion designs that make restore part of a graph-captured device
  kernel rather than a Python callback.

This is not yet proof that a particular serving model returns incorrect
results: generic graph dispatch, request classification, attention backend,
and the MLA `has_prefill` guard can change the exercised path. A real-engine
trace must record the selected cudagraph mode and per-step hook/load counts.
It is, however, strong evidence that the hoped-for behavior—"Python hooks
automatically run again" or "their external load-stream copies automatically
become graph nodes"—does not occur.

Raw retained JSONL is under the ignored directory:

`branch_development_notes/work/aclgraph-connector-hooks-20260718/`

## Validation

- physical NPU 3 faithful reproduction, task queue disabled: passed baseline,
  reproduced missing replay pipeline;
- physical NPU 3 faithful reproduction, normal task queue: passed baseline,
  reproduced missing replay pipeline;
- static source audit: passed;
- pure helper/source-audit unit tests: `3 passed`.
