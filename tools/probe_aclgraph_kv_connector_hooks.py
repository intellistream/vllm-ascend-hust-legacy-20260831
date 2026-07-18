#!/usr/bin/env python3
"""Probe layerwise KV connector hooks across ACLGraph capture and replay.

The CPU offload connector starts layer 0 outside the model, then advances the
load pipeline from Python ``wait_for_layer_load`` hooks inside attention.  A
full ACLGraph replay, however, normally replays device tasks without calling
the Python model again.  This tool makes that boundary observable without
starting a complete vLLM engine.

The device experiment is a faithful control-flow reproduction, not a model
benchmark.  It uses stable pinned-host sources, a dedicated load stream, and
one destination tensor per layer.  Each layer hook synchronizes its load and
launches the next layer exactly like ``CPUOffloadingConnectorWorker``.  JSONL
events report Python call counts and whether replay picked up new host values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _line_of(text: str, needle: str) -> int | None:
    for line_number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return line_number
    return None


def _method_body(text: str, method_name: str) -> str:
    marker = f"    def {method_name}("
    start = text.find(marker)
    if start < 0:
        return ""
    end = text.find("\n    def ", start + len(marker))
    return text[start:] if end < 0 else text[start:end]


def audit_sources(repo_root: Path) -> dict[str, Any]:
    """Return static evidence for the production connector/graph boundary."""
    connector_path = repo_root / (
        "vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/"
        "cpu_offload_connector.py"
    )
    graph_path = repo_root / "vllm_ascend/compilation/acl_graph.py"
    runner_path = repo_root / "vllm_ascend/worker/model_runner_v1.py"
    attention_path = repo_root / "vllm_ascend/attention/utils.py"
    mla_path = repo_root / "vllm_ascend/attention/mla_v1.py"
    mixin_path = repo_root.parent / (
        "vllm-hust/vllm/v1/worker/kv_connector_model_runner_mixin.py"
    )

    connector = connector_path.read_text()
    graph = graph_path.read_text()
    runner = runner_path.read_text()
    attention = attention_path.read_text()
    mla = mla_path.read_text()
    mixin = mixin_path.read_text() if mixin_path.exists() else None

    start_call = "kv_connector.start_load_kv(get_forward_context())"
    replay_call = "entry.aclgraph.replay()"
    runnable_call = "output = self.runnable(*args, **kwargs)"
    wait_call = "connector.wait_for_layer_load(layer_name)"
    load_sync = "self.load_stream.synchronize()"
    next_layer = "self.load_kv_layer(self.current_layer)"
    model_forward_call = "hidden_states = self._model_forward("
    connector_context_call = "self.maybe_get_kv_connector_output("
    dispatch_body = _method_body(runner, "_determine_batch_execution_and_padding")

    start_line = _line_of(mixin, start_call) if mixin is not None else None
    connector_context_line = _line_of(runner, connector_context_call)
    model_forward_line = _line_of(runner, model_forward_call)

    return {
        "upstream_mixin_available": mixin is not None,
        "upstream_mixin_calls_start_load": start_line is not None,
        "connector_context_wraps_model_forward": (
            connector_context_line is not None
            and model_forward_line is not None
            and connector_context_line < model_forward_line
        ),
        "graph_capture_calls_python_runnable": runnable_call in graph,
        "graph_replay_calls_python_runnable": (
            runnable_call in graph[graph.find(replay_call) :] if replay_call in graph else None
        ),
        "graph_replay_uses_device_replay": replay_call in graph,
        "attention_python_hook_exists": wait_call in attention,
        "connector_wait_host_synchronizes_load_stream": load_sync in connector,
        "connector_wait_launches_next_layer": next_layer in connector,
        "cudagraph_dispatch_has_kv_connector_guard": (
            "kv_connector" in dispatch_body
            or "kv_transfer" in dispatch_body
            or "CPUOffloadingConnector" in dispatch_body
        ),
        "cudagraph_call_only_passes_model_enforce_eager": (
            "force_eager=self.model_config.enforce_eager" in runner
        ),
        "mla_wait_is_prefill_conditional": (
            "if has_prefill:\n            wait_for_kv_layer_from_connector(layer_name)" in mla
        ),
        "locations": {
            "graph_capture_runnable": [str(graph_path), _line_of(graph, runnable_call)],
            "graph_replay": [str(graph_path), _line_of(graph, replay_call)],
            "attention_hook": [str(attention_path), _line_of(attention, wait_call)],
            "connector_load_sync": [str(connector_path), _line_of(connector, load_sync)],
            "connector_next_layer": [str(connector_path), _line_of(connector, next_layer)],
            "model_full_graph_wrapper": [
                str(runner_path),
                _line_of(runner, "self.model = ACLGraphWrapper("),
            ],
            "connector_context": [str(runner_path), connector_context_line],
            "model_forward": [str(runner_path), model_forward_line],
            "cudagraph_dispatch": [
                str(runner_path),
                _line_of(runner, "def _determine_batch_execution_and_padding("),
            ],
            "upstream_start_load": [str(mixin_path), start_line],
            "mla_prefill_guard": [str(mla_path), _line_of(mla, "if has_prefill:")],
        },
    }


def classify_replay(
    *,
    layers: int,
    capture_wait_calls: int,
    replay_wait_calls: int,
    replay_matches_all_new_sources: bool,
) -> dict[str, Any]:
    """Turn raw controls into a conservative semantics finding."""
    python_hooks_rerun = replay_wait_calls > capture_wait_calls
    if python_hooks_rerun:
        outcome = "python_hooks_reran"
    elif replay_matches_all_new_sources:
        outcome = "python_hooks_skipped_but_load_tasks_were_captured"
    else:
        outcome = "python_hooks_skipped_and_layerwise_load_pipeline_was_not_replayed"
    return {
        "layers": layers,
        "python_hooks_rerun": python_hooks_rerun,
        "replay_matches_all_new_sources": replay_matches_all_new_sources,
        "outcome": outcome,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--elements", type=int, default=256)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--task-queue-enable", choices=("0", "1"), default="1")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


class JsonTrace:
    def __init__(self, output: Path | None):
        self.output = output
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, **fields}
        self.events.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def close(self) -> None:
        if self.output is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in self.events))


def run_device_probe(args: argparse.Namespace, trace: JsonTrace) -> dict[str, Any]:
    if args.layers < 2:
        raise ValueError("--layers must be at least 2")
    if args.elements <= 0:
        raise ValueError("--elements must be positive")

    # This changes capture/submission semantics and must be set before torch_npu.
    os.environ["TASK_QUEUE_ENABLE"] = args.task_queue_enable

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)

    class ConnectorLikePipeline:
        def __init__(self):
            self.load_stream = torch.npu.Stream()
            self.sources = [
                torch.empty(args.elements, dtype=torch.float32, pin_memory=True)
                for _ in range(args.layers)
            ]
            self.destinations = [
                torch.empty(args.elements, dtype=torch.float32, device=args.device)
                for _ in range(args.layers)
            ]
            self.current_layer = 0
            self.start_calls = 0
            self.wait_calls = 0
            self.load_calls = 0

        def set_sources(self, base: float) -> None:
            for layer, source in enumerate(self.sources):
                source.fill_(base + layer)

        def poison_destinations(self, value: float) -> None:
            for destination in self.destinations:
                destination.fill_(value)
            torch.npu.synchronize()

        def load_layer(self, layer: int) -> None:
            if layer == args.layers:
                return
            self.load_calls += 1
            trace.emit(
                "python_load_layer",
                layer=layer,
                load_calls=self.load_calls,
                current_layer=self.current_layer,
            )
            with torch.npu.stream(self.load_stream):
                self.destinations[layer].copy_(self.sources[layer], non_blocking=True)

        def start_load_kv(self) -> None:
            self.start_calls += 1
            self.current_layer = 0
            trace.emit("python_start_load_kv", start_calls=self.start_calls)
            self.load_layer(0)

        def wait_for_layer_load(self, layer_name: str) -> None:
            self.wait_calls += 1
            trace.emit(
                "python_wait_for_layer_load",
                layer_name=layer_name,
                wait_calls=self.wait_calls,
                current_layer=self.current_layer,
            )
            self.load_stream.synchronize()
            self.current_layer += 1
            self.load_layer(self.current_layer)

    pipeline = ConnectorLikePipeline()
    model_input = torch.zeros(args.elements, dtype=torch.float32, device=args.device)

    def python_model(value: Any) -> Any:
        output = value
        for layer, destination in enumerate(pipeline.destinations):
            pipeline.wait_for_layer_load(f"layer.{layer}")
            output = output + destination
        return output

    capture_base = 10.0
    pipeline.set_sources(capture_base)
    pipeline.poison_destinations(-1.0)
    pipeline.start_load_kv()
    graph = torch.npu.NPUGraph()
    trace.emit("capture_begin")
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        graph_output = python_model(model_input)
    torch.npu.synchronize()
    capture_context_values = graph_output.cpu()
    capture_wait_calls = pipeline.wait_calls
    capture_load_calls = pipeline.load_calls
    expected_capture = sum(capture_base + layer for layer in range(args.layers))
    graph.replay()
    torch.npu.synchronize()
    baseline_values = graph_output.cpu()
    baseline_correct = bool(torch.allclose(baseline_values, torch.full_like(baseline_values, expected_capture)))
    trace.emit(
        "capture_end",
        capture_context_first_output_value=float(capture_context_values[0].item()),
        baseline_replay_correct=baseline_correct,
        baseline_replay_first_output_value=float(baseline_values[0].item()),
        expected_output_value=expected_capture,
        wait_calls=capture_wait_calls,
        load_calls=capture_load_calls,
    )

    replay_results: list[dict[str, Any]] = []
    poison = -1000.0
    for replay_index in range(args.replays):
        replay_base = 100.0 + replay_index * 100.0
        pipeline.set_sources(replay_base)
        pipeline.poison_destinations(poison)
        pipeline.start_load_kv()

        # Be deliberately generous to the production design: make layer 0
        # complete before replay.  A failure can then only come from layers
        # whose loads are normally launched by Python attention hooks.
        pipeline.load_stream.synchronize()
        wait_calls_before = pipeline.wait_calls
        load_calls_before = pipeline.load_calls
        trace.emit("replay_begin", replay_index=replay_index, replay_base=replay_base)
        graph.replay()
        torch.npu.synchronize()
        actual = graph_output.cpu()
        expected_value = sum(replay_base + layer for layer in range(args.layers))
        expected = torch.full_like(actual, expected_value)
        matches = bool(torch.allclose(actual, expected))
        first_value = float(actual[0].item())
        result = {
            "replay_index": replay_index,
            "matches_all_new_sources": matches,
            "first_output_value": first_value,
            "expected_output_value": expected_value,
            "wait_calls_delta": pipeline.wait_calls - wait_calls_before,
            "load_calls_delta": pipeline.load_calls - load_calls_before,
        }
        replay_results.append(result)
        trace.emit("replay_end", **result)

    finding = classify_replay(
        layers=args.layers,
        capture_wait_calls=capture_wait_calls,
        replay_wait_calls=pipeline.wait_calls,
        replay_matches_all_new_sources=all(row["matches_all_new_sources"] for row in replay_results),
    )
    result = {
        "device": args.device,
        "task_queue_enable": args.task_queue_enable,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "baseline_replay_correct": baseline_correct,
        "capture_wait_calls": capture_wait_calls,
        "capture_load_calls": capture_load_calls,
        "final_wait_calls": pipeline.wait_calls,
        "final_load_calls": pipeline.load_calls,
        "replays": replay_results,
        **finding,
    }
    trace.emit("device_finding", **result)
    return result


def main() -> int:
    args = parse_args()
    trace = JsonTrace(args.output)
    try:
        static = audit_sources(args.repo_root.resolve())
        trace.emit("static_audit", **static)
        if args.static_only:
            return 0
        result = run_device_probe(args, trace)
        return 0 if result["baseline_replay_correct"] else 2
    finally:
        trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
