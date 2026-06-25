# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoERoutingParams,
    MoEWeights,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.moe_offload.runtime import (
    MoeOffloadDecisionPath,
    _is_current_graph_capturing,
    get_moe_offload_runtime,
)
from vllm_ascend.moe_offload.pipeline import get_moe_pipeline_profiler
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
    _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
    _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8]

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        # --- P0 pipeline profiling: npu.Event trace-only timing ---
        pipeline_profiler = get_moe_pipeline_profiler()
        do_pipe_profile = pipeline_profiler.enabled
        if do_pipe_profile:
            e0 = pipeline_profiler.record()  # before Stage T (offload plan / transfer)

        before_dispatch_evt = torch.npu.current_stream().record_event()

        # B2 wave-streamed prefill early branch. Only does extra work when the
        # b2_wave_prefill config is on AND offload is active; otherwise this is a
        # couple of cheap attribute checks and the normal path below is untouched.
        _b2_out = self._maybe_run_b2_wave_prefill(fused_experts_input, before_dispatch_evt)
        if _b2_out is not None:
            return _b2_out

        fused_experts_input = self._maybe_apply_moe_offload_plan(fused_experts_input)

        if do_pipe_profile:
            e1 = pipeline_profiler.record()  # after Stage T, before Stage R

        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)
        runtime = get_moe_offload_runtime()
        runtime.trace_grouped_active_experts(
            layer_id=fused_experts_input.trace_layer_id,
            group_list=token_dispatch_output.group_list,
            group_list_type=token_dispatch_output.group_list_type,
            physical_expert_count=fused_experts_input.routing.physical_expert_count,
        )
        compute_bucket_decision = runtime.classify_grouped_compute_bucket(
            layer_id=fused_experts_input.trace_layer_id,
            group_list=token_dispatch_output.group_list,
            group_list_type=token_dispatch_output.group_list_type,
            phase="decode" if fused_experts_input.hidden_states.shape[0] <= 1 else "prefill",
        )

        if do_pipe_profile:
            e2 = pipeline_profiler.record()  # after Stage R, before Stage C

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
            compute_bucket_decision=compute_bucket_decision,
        )

        # --- MVP-D.11: post-dispatch phase split (default off) ---
        _phase_split_enabled, _phase_plan, _phase_fail_reason = self._maybe_plan_phase_split(
            fused_experts_input=fused_experts_input,
            mlp_compute_input=mlp_compute_input,
            token_dispatch_output=token_dispatch_output,
        )
        if _phase_split_enabled and _phase_plan is not None:
            from vllm_ascend.moe_offload.phase_split import execute_phased_mlp

            mlp_output = execute_phased_mlp(
                mlp_compute_input=mlp_compute_input,
                phase_plan=_phase_plan,
            )
        elif _phase_split_enabled and _phase_fail_reason is not None:
            raise RuntimeError(
                "MoE phase split failed closed: " + _phase_fail_reason
            )
        else:
            mlp_output = self._apply_mlp(mlp_compute_input)

        if do_pipe_profile:
            e3 = pipeline_profiler.record()  # after Stage C, before Stage M

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        if do_pipe_profile:
            e4 = pipeline_profiler.record()  # after Stage M
            step_id = getattr(fused_experts_input, "_pipeline_step_id", 0)
            pipeline_profiler.commit(
                layer_id=getattr(fused_experts_input.offload, "layer_id", -1) if fused_experts_input.offload else -1,
                step_id=step_id,
                events=(e0, e1, e2, e3, e4),
            )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    def _maybe_run_b2_wave_prefill(self, fused_experts_input, before_dispatch_evt):
        """B2 wave-streamed prefill. Returns a FusedExpertsResult when the B2 gate
        fires, else None (caller continues the normal path).

        Splits the eager-prefill active expert union into capacity-bounded waves
        of <= num_slots; per wave it stages that wave's experts into the fixed slot
        bank, remaps topk_ids through the wave's log2phy (non-wave -> -1), zeroes
        non-wave weights (build_b2_wave_routing), runs the existing dispatch ->
        matmul -> combine on the staged slot weights, and accumulates the combined
        outputs. The accumulation reproduces the full single-pass MoE output
        (proven by the wave-accumulate keystone). Prefill+eager only; decode and
        every non-B2 layer return None here and keep their existing path.
        """
        offload = fused_experts_input.offload
        if offload is None or not offload.enabled:
            return None
        runtime = get_moe_offload_runtime()
        if not runtime.config.b2_wave_prefill:
            return None
        # Never run during graph capture (data-dependent host work); prefill is
        # eager so this is naturally satisfied, but guard explicitly.
        if _is_current_graph_capturing():
            return None

        hidden_states = fused_experts_input.hidden_states
        is_prefill = hidden_states.shape[0] > 1
        active_experts = tuple(
            int(e) for e in torch.unique(fused_experts_input.topk_ids.detach().cpu()).tolist()
        )
        if not runtime.should_use_b2_wave_prefill(
            layer_id=offload.layer_id,
            active_expert_count=len(set(active_experts)),
            is_prefill=is_prefill,
        ):
            return None

        return self._run_b2_wave_prefill(
            fused_experts_input=fused_experts_input,
            active_experts=active_experts,
            before_dispatch_evt=before_dispatch_evt,
        )

    def _run_b2_wave_prefill(self, *, fused_experts_input, active_experts, before_dispatch_evt):
        from vllm_ascend.moe_offload.phase_split import build_b2_wave_routing

        runtime = get_moe_offload_runtime()
        offload = fused_experts_input.offload
        num_slots = int(runtime.config.num_slots)
        device = fused_experts_input.topk_ids.device
        # Partition the active union into capacity-bounded waves of <= num_slots.
        uniq = sorted(set(int(e) for e in active_experts))
        waves = [tuple(uniq[s : s + num_slots]) for s in range(0, len(uniq), num_slots)]
        if os.environ.get("SEW_B2_PROBE") or os.environ.get("SEW_OFFLOAD_PROBE"):
            print(
                f"SEW_B2 branch=WAVE_RUN layer={offload.layer_id} "
                f"n_active={len(uniq)} num_slots={num_slots} n_waves={len(waves)}",
                flush=True,
            )

        accumulated = None
        last_group_list_type = None
        last_expert_tokens = None
        before_combine_evt = before_dispatch_evt
        for wave_index, wave in enumerate(waves):
            # Stage this wave's <= num_slots experts; prepared.log2phy maps these
            # logical experts to physical slot positions and all others to -1.
            prepared = runtime.prepare_fixed_slot_plan(
                layer_id=offload.layer_id,
                active_experts=wave,
                num_logical_experts=offload.num_logical_experts,
                device=device,
            )
            prepared.validate_backend_ready(expected_device_type=offload.expected_device_type)
            wave_result = self._run_b2_single_wave(
                fused_experts_input=fused_experts_input,
                prepared=prepared,
            )
            before_combine_evt = wave_result.before_combine_evt
            last_group_list_type = wave_result.group_list_type
            last_expert_tokens = wave_result.expert_tokens
            if accumulated is None:
                accumulated = wave_result.routed_out
            else:
                accumulated = accumulated + wave_result.routed_out

        if accumulated is None:  # no waves (shouldn't happen under the gate)
            return None
        return FusedExpertsResult(
            routed_out=accumulated,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=last_group_list_type,
            expert_tokens=last_expert_tokens,
        )

    def _run_b2_single_wave(self, *, fused_experts_input, prepared):
        """Run dispatch -> matmul -> combine for one B2 wave on staged slot weights.

        ``prepared`` is the slot plan for this wave: ``prepared.log2phy`` maps the
        wave's logical experts to physical slot positions and all others to -1.
        We remap topk_ids through it, then ``build_b2_wave_routing`` makes -1 safe
        (-> slot 0) and zeroes those weights so non-wave (token,expert) pairs add 0
        in combine. Everything else reuses the existing dispatch/matmul/combine.
        """
        from vllm_ascend.moe_offload.phase_split import build_b2_wave_routing

        physical_topk_ids = prepared.log2phy[fused_experts_input.topk_ids]
        safe_ids, masked_weights = build_b2_wave_routing(
            physical_topk_ids, fused_experts_input.topk_weights
        )

        wave_input = MoEFusedExpertsInput(
            hidden_states=fused_experts_input.hidden_states,
            topk_weights=masked_weights,
            topk_ids=safe_ids,
            weights=MoEWeights(
                w1=prepared.w1,
                w2=prepared.w2,
                w1_bias=fused_experts_input.weights.w1_bias,
                w2_bias=fused_experts_input.weights.w2_bias,
                w1_scale=fused_experts_input.weights.w1_scale,
                w2_scale=fused_experts_input.weights.w2_scale,
                w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                w1_offset=fused_experts_input.weights.w1_offset,
                w2_offset=fused_experts_input.weights.w2_offset,
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=fused_experts_input.routing.global_redundant_expert_num,
                mc2_mask=fused_experts_input.routing.mc2_mask,
                apply_router_weight_on_input=fused_experts_input.routing.apply_router_weight_on_input,
                log2phy=None,  # already remapped into safe_ids above
                physical_expert_count=prepared.physical_expert_count,
                pertoken_scale=fused_experts_input.routing.pertoken_scale,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            offload=fused_experts_input.offload,
            trace_layer_id=fused_experts_input.trace_layer_id,
            trace_num_logical_experts=fused_experts_input.trace_num_logical_experts,
        )

        token_dispatch_output = self.token_dispatcher.token_dispatch(
            token_dispatch_input=build_token_dispatch_input(fused_experts_input=wave_input)
        )
        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=wave_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
            compute_bucket_decision=None,
        )
        mlp_output = self._apply_mlp(mlp_compute_input)
        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_combine_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
        )

    def _maybe_plan_phase_split(
        self,
        *,
        fused_experts_input: MoEFusedExpertsInput,
        mlp_compute_input: MoEMlpComputeInput,
        token_dispatch_output,
    ) -> tuple[bool, object | None, str | None]:
        """MVP-D.11: decide whether to split MLP into phases.

        Returns ``(enabled, phase_plan | None, fail_reason | None)``.
        """
        runtime = get_moe_offload_runtime()
        if not runtime.config.phase_split_enabled:
            return False, None, None

        # --- Narrow-path gate ---
        # D.11 supports: AllGather + unquantized + no bias + no EP.
        from vllm_ascend.ascend_forward_context import MoECommType as _MoECommType

        if _EXTRA_CTX.moe_comm_type != _MoECommType.ALLGATHER:
            return True, None, "phase_split_requires_AllGather"
        if mlp_compute_input.quant.is_quant:
            return True, None, "phase_split_requires_unquantized"
        if fused_experts_input.weights.w1_bias is not None or fused_experts_input.weights.w2_bias is not None:
            return True, None, "phase_split_requires_no_bias"
        if fused_experts_input.routing.expert_map is not None:
            return True, None, "phase_split_requires_no_expert_map"

        # --- Build phase plan ---
        from vllm_ascend.moe_offload.phase_split import (
            MoEPhasePlan,
            PhaseSplitProfileEvent,
            _write_phase_split_profile_jsonl,
            compute_expert_token_slices,
            plan_hit_miss_phases,
        )

        offload = fused_experts_input.offload
        layer_id = offload.layer_id if offload is not None else -1

        try:
            group_list = mlp_compute_input.group_list
            group_list_type = mlp_compute_input.group_list_type

            expert_slices = compute_expert_token_slices(group_list, group_list_type)
            num_experts_in_group = len(expert_slices)

            # Active expert ids are 0..(num_experts_in_group-1) — the group_list
            # is already permuted to the local expert order.
            active_expert_ids = tuple(range(num_experts_in_group))

            # Build slot readiness map from the offload runtime.
            slot_readiness: dict[int, bool] = {}
            if offload is not None and offload.enabled and runtime.should_use_fixed_slot_plan_for_layer(layer_id):
                slot_bank = runtime._slot_banks.get(layer_id)
                if slot_bank is not None:
                    for expert_id in active_expert_ids:
                        from vllm_ascend.moe_offload.expert_key import ExpertKey
                        from vllm_ascend.moe_offload.slot_bank import SlotState

                        key = ExpertKey(layer_id, int(expert_id))
                        slot = slot_bank.lookup(key)
                        slot_readiness[int(expert_id)] = (
                            slot is not None and slot.state == SlotState.READY
                        )

            max_phases = runtime.config.max_phases
            phase_plan: MoEPhasePlan = plan_hit_miss_phases(
                expert_slices=expert_slices,
                active_expert_ids=active_expert_ids,
                slot_readiness=slot_readiness if slot_readiness else None,
                max_phases=max_phases,
            )

            # Observability
            event = PhaseSplitProfileEvent(
                name="phase_split_plan",
                layer_id=layer_id,
                seconds=0.0,  # plan time is negligible at this point
                phase_plan_jsonable=phase_plan.to_jsonable(),
            )
            _write_phase_split_profile_jsonl(event)

            return True, phase_plan, None

        except Exception as exc:
            fail_reason = f"phase_split_plan_failed: {exc}"
            event = PhaseSplitProfileEvent(
                name="phase_split_fail_closed",
                layer_id=layer_id,
                seconds=0.0,
                fail_reason=fail_reason,
            )
            _write_phase_split_profile_jsonl(event)
            return True, None, fail_reason

    def _maybe_apply_moe_offload_plan(self, fused_experts_input: MoEFusedExpertsInput) -> MoEFusedExpertsInput:
        offload = fused_experts_input.offload
        if offload is None or not offload.enabled:
            return fused_experts_input

        runtime = get_moe_offload_runtime()

        # Option 2 (graph-compatible offload): during graph capture, the
        # data-dependent host decision (torch.unique(...).cpu()) is FORBIDDEN on a
        # captured stream (Ascend: "synchronized memcpy not supported in capture
        # mode"). Use the capture-safe path: point routing at the fixed slot
        # tensors + the persistent (fixed-address) log2phy buffer with zero host
        # sync. The real decision + H2D staging is hoisted to the eager
        # stage_fixed_slot_plan() call before replay. Default offload behavior
        # (eager, no capture) is unchanged.
        if runtime.config.graph_compatible_offload and _is_current_graph_capturing():
            capture_weights = runtime.capture_safe_slot_weights(layer_id=offload.layer_id)
            if os.environ.get("SEW_OFFLOAD_PROBE"):
                buf = runtime.log2phy_buffer(offload.layer_id)
                # NOTE: no .item()/.sum() — device->host sync is forbidden during
                # capture (107027). Log shape/id only, never read tensor values.
                print(
                    f"SEW_PROBE branch=CAPTURE_SAFE layer={offload.layer_id} "
                    f"capture_weights_none={capture_weights is None} "
                    f"buf_numel={None if buf is None else buf.numel()} "
                    f"buf_id={None if buf is None else id(buf)}",
                    flush=True,
                )
            if capture_weights is not None:
                return self._with_prepared_slot_weights(fused_experts_input, capture_weights)
            return fused_experts_input

        active_experts = tuple(
            int(expert_id) for expert_id in torch.unique(fused_experts_input.topk_ids.detach().cpu()).tolist()
        )
        # B2 (wave-streamed prefill) detection — PROBE ONLY at this step. Mirrors
        # the prefill/decode classifier (hidden_states.shape[0] > 1 == prefill).
        # When the gate fires we only emit a marker; the path is unchanged (still
        # the existing fail-closed / slot-cache flow below). The real wave path is
        # wired in a following step. Keeps behavior identical while letting NPU
        # runs confirm the gate fires where expected (num_slots=8, fanout>8).
        is_prefill = fused_experts_input.hidden_states.shape[0] > 1
        if runtime.should_use_b2_wave_prefill(
            layer_id=offload.layer_id,
            active_expert_count=len(set(active_experts)),
            is_prefill=is_prefill,
        ):
            if os.environ.get("SEW_B2_PROBE") or os.environ.get("SEW_OFFLOAD_PROBE"):
                print(
                    f"SEW_B2 branch=GATE_FIRES layer={offload.layer_id} "
                    f"n_active={len(set(active_experts))} num_slots={runtime.config.num_slots} "
                    f"is_prefill={is_prefill} (probe-only, path unchanged)",
                    flush=True,
                )
        if os.environ.get("SEW_OFFLOAD_PROBE"):
            print(
                f"SEW_PROBE branch=EAGER layer={offload.layer_id} "
                f"capturing={_is_current_graph_capturing()} "
                f"graph_compat={runtime.config.graph_compatible_offload} "
                f"n_active={len(set(active_experts))}",
                flush=True,
            )
        use_slot_cache_path = True
        if runtime.should_use_layered_runtime:
            decision = runtime.decide_layered_path(
                layer_id=offload.layer_id,
                active_experts=active_experts,
            )
            if decision.path is MoeOffloadDecisionPath.FAIL_CLOSED:
                raise RuntimeError(
                    "MoE offload layered runtime failed closed: "
                    f"layer_id={offload.layer_id}, reason={decision.reason}"
                )
            use_slot_cache_path = decision.path is MoeOffloadDecisionPath.SLOT_CACHE_PATH

        if not use_slot_cache_path:
            return fused_experts_input

        prepared_weights = runtime.prepare_fixed_slot_plan(
            layer_id=offload.layer_id,
            active_experts=active_experts,
            num_logical_experts=offload.num_logical_experts,
            device=fused_experts_input.topk_ids.device,
        )
        prepared_weights.validate_backend_ready(expected_device_type=offload.expected_device_type)
        return self._with_prepared_slot_weights(fused_experts_input, prepared_weights)

    def _with_prepared_slot_weights(self, fused_experts_input, prepared_weights):
        """Build a MoEFusedExpertsInput that points w1/w2/log2phy at slot tensors.

        Shared by the eager plan path and the Option-2 capture-safe path so both
        wire the (fixed-address) slot weights + log2phy buffer identically.
        """
        return MoEFusedExpertsInput(
            hidden_states=fused_experts_input.hidden_states,
            topk_weights=fused_experts_input.topk_weights,
            topk_ids=fused_experts_input.topk_ids,
            weights=MoEWeights(
                w1=prepared_weights.w1,
                w2=prepared_weights.w2,
                w1_bias=fused_experts_input.weights.w1_bias,
                w2_bias=fused_experts_input.weights.w2_bias,
                w1_scale=fused_experts_input.weights.w1_scale,
                w2_scale=fused_experts_input.weights.w2_scale,
                w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                w1_offset=fused_experts_input.weights.w1_offset,
                w2_offset=fused_experts_input.weights.w2_offset,
            ),
            routing=MoERoutingParams(
                expert_map=fused_experts_input.routing.expert_map,
                global_redundant_expert_num=fused_experts_input.routing.global_redundant_expert_num,
                mc2_mask=fused_experts_input.routing.mc2_mask,
                apply_router_weight_on_input=fused_experts_input.routing.apply_router_weight_on_input,
                log2phy=prepared_weights.log2phy,
                physical_expert_count=prepared_weights.physical_expert_count,
                pertoken_scale=fused_experts_input.routing.pertoken_scale,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            offload=fused_experts_input.offload,
            trace_layer_id=fused_experts_input.trace_layer_id,
            trace_num_logical_experts=fused_experts_input.trace_num_logical_experts,
        )

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        )

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            out = torch.empty_like(fused_experts_input.hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                scale1=fused_experts_input.weights.w1_scale,
                scale2=fused_experts_input.weights.w2_scale,
                bias1=fused_experts_input.weights.w1_scale_bias,
                bias2=fused_experts_input.weights.w2_scale_bias,
                probs=fused_experts_input.topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=65536,
                x_active_mask=fused_experts_input.routing.mc2_mask,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
            expert_tokens = self.expert_token_nums
        elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
            assert fused_experts_input.routing.expert_map is not None, "expert_map cannot be None."
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=fused_experts_input.weights.w1,
                gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                gmm2_weight=fused_experts_input.weights.w2,
                gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                expert_smooth_scales=None,
                expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs,
            )
        else:
            raise ValueError(f"Wrong value of {envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2=}")
        return FusedExpertsResult(routed_out=out, expert_tokens=expert_tokens)
