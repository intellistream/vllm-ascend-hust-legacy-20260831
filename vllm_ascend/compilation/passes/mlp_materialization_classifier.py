#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Default-off MLP materialization boundary classifier.

This pass records whether the compiled graph contains the dense MLP boundary
pattern observed in Qwen-style decode:

    unquantized_gemm -> npu_swiglu -> unquantized_gemm

It does not rewrite the graph.  The output is evidence for rewrite readiness,
not a performance optimization.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.logger import logger

import vllm_ascend.ops.mlp_materialization  # noqa: F401
from vllm_ascend.ops.mlp_materialization import get_mlp_writeback_backend_status

_ENABLE_ENV = "VLLM_ASCEND_MLP_MATERIALIZATION_CLASSIFY"
_OUTPUT_ENV = "VLLM_ASCEND_MLP_MATERIALIZATION_CLASSIFY_FILE"
_REWRITE_ENV = "VLLM_ASCEND_MLP_MATERIALIZATION_REWRITE"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _target_name(target: object) -> str:
    return str(target)


def _is_unquantized_gemm(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "unquantized_gemm" in _target_name(node.target)


def _is_npu_swiglu(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "npu_swiglu" in _target_name(node.target)


def _shape_of(node: torch.fx.Node) -> list[str] | None:
    for key in ("val", "example_value"):
        value = node.meta.get(key)
        shape = getattr(value, "shape", None)
        if shape is not None:
            return [str(dim) for dim in shape]
    return None


def _dtype_of(node: torch.fx.Node) -> str | None:
    for key in ("val", "example_value"):
        value = node.meta.get(key)
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            return str(dtype)
    return None


def _dtype_bytes(dtype: str | None) -> int | None:
    if dtype in {"torch.bfloat16", "torch.float16", "torch.half"}:
        return 2
    if dtype in {"torch.float32", "torch.int32"}:
        return 4
    if dtype in {"torch.float64", "torch.int64"}:
        return 8
    return None


def _shape_bytes_expr(shape: list[str] | None, dtype: str | None) -> str | None:
    bytes_per_elem = _dtype_bytes(dtype)
    if shape is None or bytes_per_elem is None:
        return None
    return " * ".join([*shape, str(bytes_per_elem)])


def _tensor_bytes_proxy(
    name: str, shape: list[str] | None, dtype: str | None, why: str
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "bytes_expr": _shape_bytes_expr(shape, dtype),
        "why": why,
    }


def _boundary_materialization_proxy(
    previous_gemm: torch.fx.Node, swiglu: torch.fx.Node
) -> dict[str, Any]:
    gate_up = _tensor_bytes_proxy(
        previous_gemm.name,
        _shape_of(previous_gemm),
        _dtype_of(previous_gemm),
        "gate/up GEMM output is produced before SwiGLU consumes it",
    )
    activation = _tensor_bytes_proxy(
        swiglu.name,
        _shape_of(swiglu),
        _dtype_of(swiglu),
        "SwiGLU activation output is consumed only by the following down GEMM",
    )
    return {
        "proxy_kind": "static_fx_tensor_lifetime_proxy",
        "not_a_hardware_counter": True,
        "gate_up_tensor": gate_up,
        "activation_tensor": activation,
        "exposed_intermediate_events": [
            {
                "event": "gate_up_write_then_activation_read",
                "tensor": gate_up["name"],
                "bytes_expr_each_direction": gate_up["bytes_expr"],
                "why": "composite fallback exposes gate/up as an ordinary tensor",
            },
            {
                "event": "activation_write_then_down_read",
                "tensor": activation["name"],
                "bytes_expr_each_direction": activation["bytes_expr"],
                "why": "current fallback exposes SwiGLU activation before down projection",
            },
        ],
        "target_elision": (
            "a true lowering must keep the activation private to the carrier or "
            "stage it directly for down projection instead of exposing this "
            "standalone write/read boundary"
        ),
    }


def _numeric_dim(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _node_meta_keys(node: torch.fx.Node) -> list[str]:
    return sorted(str(key) for key in node.meta)


def _summarize_arg(value: object, *, depth: int = 0) -> Any:
    if depth > 2:
        return {"kind": "truncated", "repr": repr(value)[:160]}
    if isinstance(value, torch.fx.Node):
        return {
            "kind": "node",
            "name": value.name,
            "op": value.op,
            "target": _target_name(value.target),
            "shape": _shape_of(value),
            "dtype": _dtype_of(value),
            "user_count": len(value.users),
            "meta_keys": _node_meta_keys(value),
        }
    if isinstance(value, (tuple, list)):
        return [_summarize_arg(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _summarize_arg(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:160]}


def _summarize_node_call(node: torch.fx.Node) -> dict[str, Any]:
    return {
        "name": node.name,
        "op": node.op,
        "target": _target_name(node.target),
        "shape": _shape_of(node),
        "dtype": _dtype_of(node),
        "user_count": len(node.users),
        "users": sorted(user.name for user in node.users),
        "meta_keys": _node_meta_keys(node),
        "args": _summarize_arg(node.args),
        "kwargs": _summarize_arg(node.kwargs),
    }


def _shape_compatible_for_dense_mlp(
    previous_gemm: torch.fx.Node, swiglu: torch.fx.Node, next_gemm: torch.fx.Node
) -> bool:
    previous_shape = _shape_of(previous_gemm)
    swiglu_shape = _shape_of(swiglu)
    next_shape = _shape_of(next_gemm)
    if previous_shape is None or swiglu_shape is None or next_shape is None:
        return False
    if len(previous_shape) != 2 or len(swiglu_shape) != 2 or len(next_shape) != 2:
        return False
    if previous_shape[0] != swiglu_shape[0] or swiglu_shape[0] != next_shape[0]:
        return False
    previous_width = _numeric_dim(previous_shape[-1])
    swiglu_width = _numeric_dim(swiglu_shape[-1])
    return (
        previous_width is not None
        and swiglu_width is not None
        and previous_width == 2 * swiglu_width
    )


def _carrier_contract_for_boundary(
    previous_gemm: torch.fx.Node, swiglu: torch.fx.Node, next_gemm: torch.fx.Node
) -> dict[str, Any]:
    previous_shape = _shape_of(previous_gemm)
    swiglu_shape = _shape_of(swiglu)
    next_shape = _shape_of(next_gemm)
    dtype = _dtype_of(swiglu)
    guard_results = {
        "dtype_bf16": dtype == "torch.bfloat16",
        "single_swiglu_user": len(swiglu.users) == 1,
        "rank2_dense_mlp_shapes": all(
            shape is not None and len(shape) == 2
            for shape in (previous_shape, swiglu_shape, next_shape)
        ),
        "shape_compatible_gate_up_swiglu_down": _shape_compatible_for_dense_mlp(
            previous_gemm, swiglu, next_gemm
        ),
        "down_weight_visible": len(next_gemm.args) >= 2,
    }
    contract_ready = all(guard_results.values())
    return {
        "contract_ready_without_carrier": contract_ready,
        "guard_results": guard_results,
        "carrier_kind": "mlp_boundary_writeback_elision",
        "required_lowering": (
            "consume gate/up GEMM output, apply SwiGLU, and feed down GEMM "
            "without exposing the activation tensor as a standalone writeback"
        ),
        "semantic_boundary_preserved": (
            "equivalent to unquantized_gemm -> npu_swiglu -> unquantized_gemm"
        ),
        "candidate_saved_tensor": {
            **_tensor_bytes_proxy(
                swiglu.name,
                swiglu_shape,
                dtype,
                "activation output is consumed only by the following down GEMM",
            ),
        },
        "materialization_proxy": _boundary_materialization_proxy(previous_gemm, swiglu),
        "inputs": {
            "hidden_or_previous_input": _summarize_arg(previous_gemm.args[0])
            if previous_gemm.args
            else None,
            "gate_up_weight": _summarize_arg(previous_gemm.args[1])
            if len(previous_gemm.args) > 1
            else None,
            "down_weight": _summarize_arg(next_gemm.args[1])
            if len(next_gemm.args) > 1
            else None,
        },
        "output": _summarize_node_call(next_gemm),
    }


def _readiness_for_boundary(
    previous_gemm: torch.fx.Node, swiglu: torch.fx.Node, next_gemm: torch.fx.Node
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(swiglu.users) != 1:
        reasons.append("swiglu_has_multiple_users")
    if _shape_of(previous_gemm) is None or _shape_of(swiglu) is None or _shape_of(next_gemm) is None:
        reasons.append("missing_tensor_shape_meta")
    if _dtype_of(previous_gemm) is None or _dtype_of(swiglu) is None or _dtype_of(next_gemm) is None:
        reasons.append("missing_tensor_dtype_meta")
    if len(next_gemm.args) < 2:
        reasons.append("next_gemm_weight_arg_unavailable")
    carrier_contract = _carrier_contract_for_boundary(previous_gemm, swiglu, next_gemm)
    for guard_name, passed in carrier_contract["guard_results"].items():
        if not passed:
            reasons.append(f"carrier_guard_failed:{guard_name}")

    backend_status = get_mlp_writeback_backend_status()
    if backend_status["available"]:
        reasons.append("deployed_backend_is_composite_fallback_not_writeback_eliding")
    else:
        reasons.append("no_deployed_ascend_mlp_boundary_carrier_op")

    return {
        "rewrite_ready": False,
        "fallback_reasons": reasons,
        "safe_noop_probe": len(swiglu.users) == 1,
        "carrier_contract": carrier_contract,
        "required_runtime_carrier": (
            "single graph op or backend lowering for "
            "unquantized_gemm -> npu_swiglu -> unquantized_gemm"
        ),
    }


def _bias_arg(node: torch.fx.Node) -> Any:
    return node.args[2] if len(node.args) > 2 else None


def _carrier_shell_args(
    previous_gemm: torch.fx.Node, next_gemm: torch.fx.Node
) -> tuple[Any, Any, Any, Any, Any] | None:
    if len(previous_gemm.args) < 2 or len(next_gemm.args) < 2:
        return None
    return (
        previous_gemm.args[0],
        previous_gemm.args[1],
        _bias_arg(previous_gemm),
        next_gemm.args[1],
        _bias_arg(next_gemm),
    )


class MLPMaterializationClassifierPass(VllmInductorPass):
    """Classify dense MLP materialization boundaries without modifying graph."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        ascend_config = config.additional_config.get("ascend_compilation_config", {})
        self.enabled = _truthy(os.getenv(_ENABLE_ENV)) or _truthy(
            ascend_config.get("mlp_materialization_classify")
        )
        self.rewrite_mode = (
            os.getenv(_REWRITE_ENV)
            or str(ascend_config.get("mlp_materialization_rewrite", ""))
        ).strip()
        output_file = os.getenv(_OUTPUT_ENV) or ascend_config.get(
            "mlp_materialization_classify_file"
        )
        self.output_file = Path(output_file) if output_file else None
        self._emit(
            {
                "event": "constructed",
                "enabled": self.enabled,
                "rewrite_mode": self.rewrite_mode,
                "output_file": None if self.output_file is None else str(self.output_file),
            }
        )

    def _emit(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = {
            "timestamp_ns": time.time_ns(),
            "evidence_label": "real-compile-pass-probe",
            "pass": self.__class__.__name__,
            **payload,
        }
        logger.info("MLP materialization classifier event: %s", payload)
        if self.output_file is None:
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def is_applicable_for_range(self, compile_range: Any) -> bool:
        self._emit(
            {
                "event": "is_applicable_for_range",
                "compile_range": {
                    "start": getattr(compile_range, "start", None),
                    "end": getattr(compile_range, "end", None),
                },
            }
        )
        return True

    def __call__(self, graph: torch.fx.Graph) -> None:
        if not self.enabled:
            return

        fx_graph = graph.graph if hasattr(graph, "graph") else graph
        node_count_before = sum(1 for _ in fx_graph.nodes)
        records: list[dict[str, Any]] = []
        boundaries: list[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]] = []
        target_samples: list[dict[str, str]] = []
        for node in fx_graph.nodes:
            if node.op == "call_function" and len(target_samples) < 64:
                target_samples.append({"name": node.name, "target": _target_name(node.target)})
            if not _is_npu_swiglu(node):
                continue

            previous_gemm = node.args[0] if node.args else None
            if not isinstance(previous_gemm, torch.fx.Node) or not _is_unquantized_gemm(previous_gemm):
                continue

            next_gemms = [
                user
                for user in node.users
                if isinstance(user, torch.fx.Node) and _is_unquantized_gemm(user)
            ]
            for next_gemm in next_gemms:
                readiness = _readiness_for_boundary(previous_gemm, node, next_gemm)
                boundaries.append((previous_gemm, node, next_gemm))
                records.append(
                    {
                        "swiglu_node": node.name,
                        "previous_gemm_node": previous_gemm.name,
                        "next_gemm_node": next_gemm.name,
                        "swiglu_shape": _shape_of(node),
                        "swiglu_dtype": _dtype_of(node),
                        "previous_gemm_shape": _shape_of(previous_gemm),
                        "next_gemm_shape": _shape_of(next_gemm),
                        "swiglu_user_count": len(node.users),
                        "previous_gemm_user_count": len(previous_gemm.users),
                        "next_gemm_user_count": len(next_gemm.users),
                        "readiness": readiness,
                        "calls": {
                            "previous_gemm": _summarize_node_call(previous_gemm),
                            "swiglu": _summarize_node_call(node),
                            "next_gemm": _summarize_node_call(next_gemm),
                        },
                    }
                )

        fallback_reason_counts: dict[str, int] = {}
        for record in records:
            for reason in record["readiness"]["fallback_reasons"]:
                fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
        carrier_contract_ready = sum(
            1
            for record in records
            if record["readiness"]["carrier_contract"]["contract_ready_without_carrier"]
        )

        self._emit(
            {
                "event": "called",
                "node_count": node_count_before,
                "matched_dense_mlp_boundaries": len(records),
                "rewrite_ready_boundaries": sum(
                    1 for record in records if record["readiness"]["rewrite_ready"]
                ),
                "carrier_contract_ready_boundaries": carrier_contract_ready,
                "fallback_reason_counts": fallback_reason_counts,
                "records": records[:128],
                "target_samples": target_samples,
            }
        )
        if self.rewrite_mode in {
            "boundary_noop_probe",
            "materialization_noop_probe",
            "boundary_rewrite_readiness",
            "materialization_rewrite_readiness",
            "boundary_carrier_contract",
            "materialization_carrier_contract",
            "boundary_carrier_shell",
            "materialization_carrier_shell",
            "boundary_real_candidate",
            "materialization_real_candidate",
            "boundary_real_candidate_force_stub",
        }:
            self._emit(
                {
                    "event": "boundary_rewrite_readiness",
                    "rewrite_mode": self.rewrite_mode,
                    "matched_dense_mlp_boundaries": len(records),
                    "rewrite_ready_boundaries": sum(
                        1 for record in records if record["readiness"]["rewrite_ready"]
                    ),
                    "fallback_reason_counts": fallback_reason_counts,
                    "carrier_contract_ready_boundaries": carrier_contract_ready,
                    "candidate_boundary": "unquantized_gemm -> npu_swiglu -> unquantized_gemm",
                    "current_status": (
                        "classified; deployed composite fallback can validate "
                        "end-to-end carrier execution, but true writeback-eliding "
                        "lowering is still required for a performance claim"
                    ),
                    "records": records[:128],
                }
            )
            self._emit(
                {
                    "event": "boundary_carrier_contract",
                    "rewrite_mode": self.rewrite_mode,
                    "matched_dense_mlp_boundaries": len(records),
                    "carrier_contract_ready_boundaries": carrier_contract_ready,
                    "fallback_reason_counts": fallback_reason_counts,
                    "carrier_status": "contract_ready_no_backend_carrier"
                    if carrier_contract_ready and not get_mlp_writeback_backend_status()["available"]
                    else (
                        "contract_ready_deployed_composite_fallback"
                        if carrier_contract_ready
                        else "contract_blocked_before_carrier"
                    ),
                    "required_runtime_carrier": (
                        "MLP-level lowering for gate/up GEMM -> SwiGLU -> down GEMM "
                        "that elides or changes activation writeback"
                    ),
                    "records": records[:128],
                }
            )
        if self.rewrite_mode in {
            "boundary_real_candidate",
            "materialization_real_candidate",
            "boundary_real_candidate_force_stub",
        }:
            self._apply_real_candidate(
                fx_graph,
                records,
                boundaries,
                node_count_before,
                force_stub=self.rewrite_mode == "boundary_real_candidate_force_stub",
            )
        if self.rewrite_mode in {"boundary_carrier_shell", "materialization_carrier_shell"}:
            self._apply_carrier_shell(fx_graph, records, boundaries, node_count_before)
        if self.rewrite_mode in {"boundary_noop_probe", "materialization_noop_probe"}:
            self._emit(
                {
                    "event": "boundary_noop_probe",
                    "rewrite_mode": self.rewrite_mode,
                    "guard_result": "matched" if records else "no_match",
                    "matched_dense_mlp_boundaries": len(records),
                    "rewrite_ready_boundaries": sum(
                        1 for record in records if record["readiness"]["rewrite_ready"]
                    ),
                    "fallback_reason_counts": fallback_reason_counts,
                    "rewrite_semantics": "no-op; records the graph path only",
                    "next_hook_point": (
                        "replace or regroup the materialization between "
                        "previous_gemm, swiglu, and next_gemm records"
                    ),
                    "records": records[:128],
                }
            )

    def _apply_carrier_shell(
        self,
        fx_graph: torch.fx.Graph,
        records: list[dict[str, Any]],
        boundaries: list[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]],
        node_count_before: int,
    ) -> None:
        rewrite_hits: list[dict[str, Any]] = []
        fallback_counts: dict[str, int] = {}

        for record, (previous_gemm, swiglu, next_gemm) in zip(records, boundaries, strict=False):
            readiness = record["readiness"]
            contract = readiness["carrier_contract"]
            if not contract["contract_ready_without_carrier"]:
                for guard_name, passed in contract["guard_results"].items():
                    if not passed:
                        reason = f"carrier_guard_failed:{guard_name}"
                        fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                continue

            shell_args = _carrier_shell_args(previous_gemm, next_gemm)
            if shell_args is None:
                reason = "carrier_shell_args_unavailable"
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                continue

            try:
                with fx_graph.inserting_before(next_gemm):
                    shell = fx_graph.call_function(
                        torch.ops.vllm.mlp_boundary_carrier_shell,
                        args=shell_args,
                        kwargs={},
                    )
                shell.meta.update(next_gemm.meta)
                next_gemm.replace_all_uses_with(shell)
                fx_graph.erase_node(next_gemm)
                if len(swiglu.users) == 0:
                    fx_graph.erase_node(swiglu)
                if len(previous_gemm.users) == 0:
                    fx_graph.erase_node(previous_gemm)
                rewrite_hits.append(
                    {
                        "previous_gemm_node": record["previous_gemm_node"],
                        "swiglu_node": record["swiglu_node"],
                        "next_gemm_node": record["next_gemm_node"],
                        "carrier_node": shell.name,
                        "carrier_op": "torch.ops.vllm.mlp_boundary_carrier_shell",
                        "carrier_semantics": (
                            "shell only; internally lowers to "
                            "unquantized_gemm -> npu_swiglu -> unquantized_gemm"
                        ),
                        "candidate_saved_tensor": contract["candidate_saved_tensor"],
                        "guard_results": contract["guard_results"],
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive probe path
                reason = f"carrier_shell_rewrite_exception:{exc.__class__.__name__}"
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1

        node_count_after = sum(1 for _ in fx_graph.nodes)
        self._emit(
            {
                "event": "boundary_carrier_shell_rewrite",
                "rewrite_mode": self.rewrite_mode,
                "evidence_label": "real-compile-pass-carrier-shell",
                "matched_dense_mlp_boundaries": len(records),
                "carrier_contract_ready_boundaries": sum(
                    1
                    for record in records
                    if record["readiness"]["carrier_contract"]["contract_ready_without_carrier"]
                ),
                "carrier_shell_hit_count": len(rewrite_hits),
                "carrier_shell_fallback_counts": fallback_counts,
                "node_count_before": node_count_before,
                "node_count_after": node_count_after,
                "node_count_delta": node_count_after - node_count_before,
                "claim_boundary": (
                    "carrier shell proves graph replacement plumbing only; "
                    "it does not reduce activation writeback or support a service speedup"
                ),
                "rewrite_hits": rewrite_hits[:128],
            }
        )

    def _apply_real_candidate(
        self,
        fx_graph: torch.fx.Graph,
        records: list[dict[str, Any]],
        boundaries: list[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]],
        node_count_before: int,
        *,
        force_stub: bool,
    ) -> None:
        backend_status = get_mlp_writeback_backend_status()
        fallback_counts: dict[str, int] = {}
        candidate_records: list[dict[str, Any]] = []

        for record, (previous_gemm, swiglu, next_gemm) in zip(records, boundaries, strict=False):
            readiness = record["readiness"]
            contract = readiness["carrier_contract"]
            candidate_record = {
                "previous_gemm_node": record["previous_gemm_node"],
                "swiglu_node": record["swiglu_node"],
                "next_gemm_node": record["next_gemm_node"],
                "candidate_saved_tensor": contract["candidate_saved_tensor"],
                "guard_results": contract["guard_results"],
            }
            if not contract["contract_ready_without_carrier"]:
                for guard_name, passed in contract["guard_results"].items():
                    if not passed:
                        reason = f"carrier_guard_failed:{guard_name}"
                        fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                candidate_record["status"] = "blocked_before_backend"
                candidate_records.append(candidate_record)
                continue

            if not backend_status["available"] and not force_stub:
                reason = "missing_writeback_eliding_backend_op_schema"
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                candidate_record["status"] = "blocked_missing_backend_schema"
                candidate_records.append(candidate_record)
                continue

            candidate_args = _carrier_shell_args(previous_gemm, next_gemm)
            if candidate_args is None:
                reason = "real_candidate_args_unavailable"
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                candidate_record["status"] = "blocked_args_unavailable"
                candidate_records.append(candidate_record)
                continue

            try:
                with fx_graph.inserting_before(next_gemm):
                    candidate = fx_graph.call_function(
                        torch.ops.vllm.mlp_boundary_writeback_elision_candidate,
                        args=candidate_args,
                        kwargs={},
                    )
                candidate.meta.update(next_gemm.meta)
                next_gemm.replace_all_uses_with(candidate)
                fx_graph.erase_node(next_gemm)
                if len(swiglu.users) == 0:
                    fx_graph.erase_node(swiglu)
                if len(previous_gemm.users) == 0:
                    fx_graph.erase_node(previous_gemm)
                candidate_record.update(
                    {
                        "status": "rewritten_to_real_candidate_stub"
                        if force_stub and not backend_status["available"]
                        else "rewritten_to_backend_candidate",
                        "carrier_node": candidate.name,
                        "carrier_op": "torch.ops.vllm.mlp_boundary_writeback_elision_candidate",
                    }
                )
                candidate_records.append(candidate_record)
            except Exception as exc:  # pragma: no cover - defensive probe path
                reason = f"real_candidate_rewrite_exception:{exc.__class__.__name__}"
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                candidate_record["status"] = reason
                candidate_records.append(candidate_record)

        node_count_after = sum(1 for _ in fx_graph.nodes)
        rewrite_count = sum(
            1 for record in candidate_records if str(record.get("status", "")).startswith("rewritten")
        )
        event_name = (
            "boundary_real_candidate_force_stub_rewrite"
            if force_stub
            else (
                "boundary_real_candidate_rewrite"
                if backend_status["available"]
                else "boundary_real_candidate_blocked"
            )
        )
        self._emit(
            {
                "event": event_name,
                "rewrite_mode": self.rewrite_mode,
                "evidence_label": "real-compile-pass-real-candidate",
                "matched_dense_mlp_boundaries": len(records),
                "carrier_contract_ready_boundaries": sum(
                    1
                    for record in records
                    if record["readiness"]["carrier_contract"]["contract_ready_without_carrier"]
                ),
                "real_candidate_rewrite_count": rewrite_count,
                "real_candidate_fallback_counts": fallback_counts,
                "node_count_before": node_count_before,
                "node_count_after": node_count_after,
                "node_count_delta": node_count_after - node_count_before,
                "backend_status": backend_status,
                "writeback_proxy": {
                    "proxy_kind": "static_fx_tensor_lifetime_proxy",
                    "not_a_hardware_counter": True,
                    "candidate_tensor": "SwiGLU activation output",
                    "bytes_expr_source": (
                        "records[*].candidate_saved_tensor.bytes_expr and "
                        "records[*].materialization_proxy"
                    ),
                    "first_boundary_bytes_expr": (
                        candidate_records[0]["candidate_saved_tensor"].get("bytes_expr")
                        if candidate_records
                        else None
                    ),
                    "first_boundary_materialization_proxy": (
                        records[0]["readiness"]["carrier_contract"].get("materialization_proxy")
                        if records
                        else None
                    ),
                    "profiler_region_names": [
                        "mlp_boundary_writeback_elision: fused_activation_composite_fallback",
                        "mlp_boundary_writeback_elision: gate_up_linear",
                        "mlp_boundary_writeback_elision: fused_swiglu_fallback",
                        "mlp_boundary_writeback_elision: down_linear",
                    ],
                },
                "claim_boundary": (
                    "real candidate currently proves end-to-end backend carrier "
                    "execution through a composite fallback; a performance claim "
                    "still requires a lowering that does not expose the SwiGLU "
                    "activation as a standalone writeback"
                ),
                "records": candidate_records[:128],
            }
        )
