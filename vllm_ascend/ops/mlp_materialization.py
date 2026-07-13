#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Default-off MLP materialization carrier shell ops.

The op in this module intentionally preserves the existing three-step dense MLP
semantics.  It is a graph-carrier shell for compile-path validation, not a
writeback-eliding backend lowering.
"""

from __future__ import annotations

import os

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_BACKEND_OP_NAME = "mlp_boundary_writeback_elision"
_LOWERING_ENV = "VLLM_ASCEND_MLP_WRITEBACK_ELISION_LOWERING"
_BACKEND_OP_SCHEMA = (
    "_C_ascend::mlp_boundary_writeback_elision("
    "Tensor hidden_states, Tensor gate_up_weight, Tensor? gate_up_bias, "
    "Tensor down_weight, Tensor? down_bias) -> Tensor"
)


def _swiglu_fallback(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :half]) * x[..., half:]


def mlp_boundary_carrier_shell(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    gate_up = torch.ops.vllm.unquantized_gemm(
        hidden_states, gate_up_weight, gate_up_bias
    )
    if gate_up.device.type == "npu":
        import torch_npu

        activated = torch_npu.npu_swiglu(gate_up)
    else:
        activated = _swiglu_fallback(gate_up)
    return torch.ops.vllm.unquantized_gemm(activated, down_weight, down_bias)


def get_mlp_writeback_backend_status() -> dict[str, object]:
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    op = None if op_namespace is None else getattr(op_namespace, _BACKEND_OP_NAME, None)
    available = op is not None
    requested_lowering = os.environ.get(_LOWERING_ENV, "composite_fallback")
    true_lowering_requested = requested_lowering in {
        "true_dense_bf16",
        "true_writeback_elision",
    }
    return {
        "available": available,
        "namespace": "_C_ascend",
        "op_name": _BACKEND_OP_NAME,
        "expected_schema": _BACKEND_OP_SCHEMA,
        "requested_lowering": requested_lowering,
        "true_lowering_requested": true_lowering_requested,
        "true_lowering_available": False,
        "why_shell_is_not_enough": (
            "mlp_boundary_carrier_shell still materializes gate/up output and "
            "SwiGLU activation before down GEMM"
        ),
        "runtime_mode": (
            "true_dense_bf16_requested_public_api_no_go"
            if available and true_lowering_requested
            else "deployed_fused_activation_composite_fallback"
            if available
            else "missing_backend_schema"
        ),
        "true_lowering_no_go": (
            "public torch_npu/ACL operator schemas do not expose a dense BF16 "
            "matmul -> SwiGLU -> down-matmul carrier with private activation "
            "lifetime; npu_swiglu returns a standalone tensor, npu_fused_matmul "
            "only supports single-matmul epilogues, and grouped SwiGLU matmul "
            "operators are quantized/grouped contracts"
        ),
        "why_composite_fallback_is_not_enough": (
            "the deployed PrivateUse1 fallback proves end-to-end carrier "
            "execution by composing linear -> fused SwiGLU -> linear when "
            "torch_npu exposes npu::npu_swiglu, but still materializes the "
            "gate/up and activation intermediates and therefore is not a "
            "writeback-eliding lowering"
        ),
        "why_npu_fused_matmul_is_not_enough": (
            "torch_npu.npu_fused_matmul supports matmul epilogues such as "
            "gelu/add/mul, but it does not express dense BF16 "
            "GEMM -> SwiGLU -> down-GEMM with private activation lifetime"
        ),
        "why_grouped_quant_ops_are_not_equivalent": (
            "grouped_matmul_swiglu_quant* requires quantized scales/group_list "
            "and returns quantized output/scale metadata; it is not the dense "
            "BF16 Qwen2 MLP semantic contract"
        ),
    }


def mlp_boundary_writeback_elision_candidate(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    backend_op = None if op_namespace is None else getattr(op_namespace, _BACKEND_OP_NAME, None)
    if backend_op is None:
        raise RuntimeError(
            "Missing Ascend writeback-eliding MLP carrier backend op: "
            f"{_BACKEND_OP_SCHEMA}"
        )
    return backend_op(hidden_states, gate_up_weight, gate_up_bias, down_weight, down_bias)


def mlp_boundary_carrier_shell_fake(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    del gate_up_weight, gate_up_bias, down_bias
    output_shape = (*hidden_states.shape[:-1], down_weight.shape[0])
    return torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)


def mlp_boundary_writeback_elision_candidate_fake(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    return mlp_boundary_carrier_shell_fake(
        hidden_states, gate_up_weight, gate_up_bias, down_weight, down_bias
    )


direct_register_custom_op(
    op_name="mlp_boundary_carrier_shell",
    op_func=mlp_boundary_carrier_shell,
    fake_impl=mlp_boundary_carrier_shell_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="mlp_boundary_writeback_elision_candidate",
    op_func=mlp_boundary_writeback_elision_candidate,
    fake_impl=mlp_boundary_writeback_elision_candidate_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
