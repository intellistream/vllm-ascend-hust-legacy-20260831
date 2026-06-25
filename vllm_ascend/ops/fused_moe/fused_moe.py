#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
import os

import torch
import torch.nn.functional as F
import torch_npu
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod, get_compressed_expert_map
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # type: ignore
try:
    from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
except ModuleNotFoundError:
    class SharedFusedMoE:
        """Compatibility mixin for vLLM versions where shared experts moved
        into FusedMoE/MoERunner instead of a separate SharedFusedMoE class."""

        pass

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime
# Registers vllm::moe_offload_stage (Regime B path ① splitting-op seam). Import
# for its registration side effect; the op is invoked via torch.ops below.
import vllm_ascend.ops.fused_moe.moe_offload_stage_op  # noqa: F401
# Registers vllm::moe_router / vllm::moe_router_indirect (Option B piece 1) and
# provides the B1 topk-injection registry used by the apply-path short-circuit.
import vllm_ascend.ops.fused_moe.moe_router_op  # noqa: F401
# Registers vllm::moe_mlp (Option B piece 3). Import for registration side effect.
import vllm_ascend.ops.fused_moe.moe_mlp_op  # noqa: F401
from vllm_ascend.ops.fused_moe import moe_seam_inject
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    enable_sp,
    maybe_trans_nz,
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)


def mock_false():
    return False


def mock_true():
    return True


def _fixed_slot_device_for_processed_weight(weight: torch.Tensor) -> torch.device:
    if weight.device.type == "cpu":
        return torch.device("npu", torch.npu.current_device())
    return weight.device


def _empty_npu_cache_if_available() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def _should_stage_processed_expert_weights_to_cpu(layer) -> bool:
    layer_id = int(getattr(layer, "layer_id", -1))
    if layer_id < 0:
        return False
    return get_moe_offload_runtime().should_use_fixed_slot_plan_for_layer(layer_id)


def _stage_processed_weight_to_cpu_if_needed(weight: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if enabled and weight.device.type != "cpu":
        weight = weight.to("cpu")
        _empty_npu_cache_if_available()
    return weight


def _build_fixed_slot_profile_topk_ids(
    *,
    num_tokens: int,
    top_k: int,
    num_slots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if top_k > num_slots:
        raise RuntimeError(f"fixed-slot profile run requires num_slots >= top_k, got {num_slots} < {top_k}")
    base_ids = torch.arange(top_k, device=device, dtype=dtype)
    return base_ids.unsqueeze(0).expand(num_tokens, top_k).contiguous()


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    @property
    def is_monolithic(self) -> bool:
        return False

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)
        stage_processed_weights_to_cpu = _should_stage_processed_expert_weights_to_cpu(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2:
            w13_data = torch_npu.npu_format_cast(w13_data, ACL_FORMAT_FRACTAL_NZ)
        else:
            w13_data = maybe_trans_nz(w13_data)
        w13_data = _stage_processed_weight_to_cpu_if_needed(
            w13_data,
            enabled=stage_processed_weights_to_cpu,
        )
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        del w13_data
        _empty_npu_cache_if_available()

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2:
            w2_data = torch_npu.npu_format_cast(w2_data, ACL_FORMAT_FRACTAL_NZ)
        else:
            w2_data = maybe_trans_nz(w2_data)
        w2_data = _stage_processed_weight_to_cpu_if_needed(
            w2_data,
            enabled=stage_processed_weights_to_cpu,
        )
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
        del w2_data
        _empty_npu_cache_if_available()
        moe_offload_runtime = get_moe_offload_runtime()
        layer_id = int(getattr(layer, "layer_id", -1))
        if moe_offload_runtime.should_use_fixed_slot_plan_for_layer(layer_id):
            moe_offload_runtime.register_layer_for_fixed_slots(
                layer,
                slot_device=_fixed_slot_device_for_processed_weight(layer.w13_weight),
            )
            # Option 2 (Regime A): one-time fill of fixed slots + persistent
            # log2phy buffer before ACLGraph capture, so the captured gather reads
            # the real mapping instead of the -1 init. No-op unless
            # graph_compatible_offload is on and num_slots >= num_logical_experts.
            # Staged from the host store's independent CPU copy, so it survives a
            # subsequent original-weight release. Eager-only (capture not active
            # here). Does not touch router / top-k / gate / combine semantics.
            #
            # Regime-gated, NOT seam-gated: full-residency staging is REQUIRED in
            # Regime A (num_slots >= n) even when offload_stage_seam is on, because
            # the static log2phy mapping is what the captured moe_mlp reads; the
            # per-step seam op is a no-op in Regime A (see moe_offload_stage_op).
            # It is skipped only in Regime B (num_slots < n), where it would trip
            # the working-set guard and per-step seam staging owns the mapping.
            _num_logical_experts = int(layer.w13_weight.shape[0])
            if moe_offload_runtime.is_static_residency_regime(_num_logical_experts):
                moe_offload_runtime.stage_full_residency_slot_plan(layer_id=layer_id)
            if os.environ.get("SEW_OFFLOAD_LEDGER"):
                # V2 verification probe (env-gated, inert by default): confirm
                # this offload layer's slot bank is filled and log2phy is no
                # longer the -1 sentinel. Eager (load-time), so reading the
                # buffer is sync-safe here. Machine-greppable single line.
                _buf = moe_offload_runtime.log2phy_buffer(layer_id)
                _ledger = moe_offload_runtime.memory_ledger()
                _staged = None if _buf is None else int((_buf >= 0).sum().item())
                print(
                    f"SEW_LEDGER layer={layer_id} "
                    f"log2phy_staged={_staged}/{None if _buf is None else _buf.numel()} "
                    f"registered_layers={_ledger.registered_layers} "
                    f"host_experts={_ledger.host_experts} "
                    f"slot_bank_bytes={_ledger.slot_bank_bytes} "
                    f"host_store_bytes={_ledger.host_store_bytes} "
                    f"original_weight_bytes={_ledger.original_expert_weight_bytes}",
                    flush=True,
                )
            if moe_offload_runtime.config.release_original_expert_weights:
                moe_offload_runtime.release_original_expert_weights_if_ready(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
        )
        # B1 seam short-circuit (Option B three-way split): when the seam path is
        # active, vllm::moe_router already computed (topk_weights, topk_ids) as a
        # top-level op and stashed them here; consume them instead of the value
        # just produced above. Faithful: the router op runs the IDENTICAL
        # select_experts call (moe_router_op.py), so on identity-prepare topology
        # this is byte-equivalent. Registry is empty when seam is off -> no-op.
        _seam_layer_id = int(getattr(layer, "layer_id", -1))
        if moe_seam_inject.has_injected_topk(_seam_layer_id):
            topk_weights, topk_ids = moe_seam_inject.peek_injected_topk(_seam_layer_id)
        moe_offload_runtime = get_moe_offload_runtime()
        topk_ids, topk_weights = moe_offload_runtime.trace_logical_active_experts(
            layer_id=getattr(layer, "layer_id", -1),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_logical_experts=num_logical_experts,
        )
        if layer.vllm_config.model_config is not None and layer.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.capture(
                    layer_id=layer.layer_id,
                    topk_ids=topk_ids,
                )

        if moe_offload_runtime.should_use_fixed_slots and zero_expert_num > 0 and zero_expert_type is not None:
            raise NotImplementedError("MoE offload fixed slots do not support zero expert path yet")

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_logical_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance and moe_offload_runtime.should_use_fixed_slots:
            topk_ids = _build_fixed_slot_profile_topk_ids(
                num_tokens=topk_ids.size(0),
                top_k=topk_ids.size(1),
                num_slots=moe_offload_runtime.config.num_slots,
                device=topk_ids.device,
                dtype=topk_ids.dtype,
            )
        elif enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
            w2_scale = None
        physical_expert_count = None
        offload_enabled = False
        offload_expected_device_type = x.device.type

        layer_id = int(getattr(layer, "layer_id", -1))
        if moe_offload_runtime.should_use_fixed_slot_plan_for_layer(layer_id):
            if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER:
                raise NotImplementedError("MoE offload fixed slots currently support AllGather only")
            if expert_map is not None:
                raise NotImplementedError("MoE offload fixed slots do not support expert_map yet")
            if global_redundant_expert_num != 0:
                raise NotImplementedError("MoE offload fixed slots do not support redundant experts yet")
            if self.moe.has_bias:
                raise NotImplementedError("MoE offload fixed slots do not support expert bias yet")

            if not moe_offload_runtime.is_layer_registered(layer_id):
                moe_offload_runtime.register_layer_for_fixed_slots(
                    layer,
                    slot_device=_fixed_slot_device_for_processed_weight(layer.w13_weight),
                )
                # Option 2 (Regime A): stage slots + log2phy on first eager touch
                # if this layer was registered lazily here rather than at load
                # time. No-op while capturing (must have staged eager already).
                # Regime-gated (NOT seam-gated): required in Regime A even with the
                # seam on; skipped only in Regime B where the per-step seam stages.
                _num_logical_experts_lazy = int(layer.w13_weight.shape[0])
                if moe_offload_runtime.is_static_residency_regime(_num_logical_experts_lazy):
                    moe_offload_runtime.stage_full_residency_slot_plan(layer_id=layer_id)
                if moe_offload_runtime.config.release_original_expert_weights:
                    moe_offload_runtime.release_original_expert_weights_if_ready(layer)
            offload_enabled = True

        # Regime B path ①: route topk_ids through the splitting-op seam so the
        # data-dependent active-set staging (D2H + H2D + log2phy write) runs eager
        # between the router piece and the grouped-MLP piece. The op is registered
        # in compilation_config.splitting_ops (platform.py), so the FX splitter
        # cuts the captured region here. Returns a clone of topk_ids to force the
        # grouped MLP's data dependency on the completed staging. No-op for resi-
        # dent / non-offload layers. Does not touch router / top-k / gate / combine.
        #
        # MUTUAL EXCLUSION with the P2c three-way seam: when topk was injected by
        # vllm::moe_mlp (has_injected_topk above), this apply() body is running
        # INSIDE the captured moe_mlp op, where a moe_offload_stage call would be
        # frozen at capture (the R3-NEGATIVE position). In that path the top-level
        # vllm::moe_offload_stage (run eager between the router/mlp pieces) already
        # owns staging, so the in-apply seam must NOT fire again. It fires only on
        # the monolithic-fallback path (no injection), where it is the sole seam.
        _seam_three_way = moe_seam_inject.has_injected_topk(_seam_layer_id)
        if (
            offload_enabled
            and moe_offload_runtime.config.offload_stage_seam
            and not _seam_three_way
        ):
            # Side-effect-only (mutates log2phy/slots in place, returns None). On
            # the monolithic-fallback path topk_ids is a plain local, so address
            # stability is moot here -- but keep the no-reassign call contract
            # uniform with the three-way seam path.
            torch.ops.vllm.moe_offload_stage(
                topk_ids,
                layer_id,
                num_logical_experts,
            )

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                physical_expert_count=physical_expert_count,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                offload_enabled=offload_enabled,
                offload_layer_id=layer_id,
                offload_num_logical_experts=num_logical_experts,
                offload_expected_device_type=offload_expected_device_type,
                trace_layer_id=layer_id,
                trace_num_logical_experts=num_logical_experts,
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


class AscendMoERunner(MoERunner):
    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    @property
    def _fused_output_is_reduced(self) -> bool:
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return shared_output

    # TODO: Remove this after drop v0.19.1 support
    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Override the default forward_impl to use Ascend-specific implementation.
        This delegates to the layer's forward_impl method which contains the
        Ascend-specific MoE computation logic.
        """
        if self.shared_experts is None:
            result = layer.forward_impl(hidden_states, router_logits)
        else:
            result = layer.shared_forward_impl(hidden_states, router_logits)
        # If the layer has shared experts, forward_impl returns a tuple (shared_out, routed_out)
        # Otherwise, it returns just routed_out
        # The torch op expects the same return type based on whether it's moe_forward or moe_forward_shared
        return result

    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )

    # ----------------------------------------------------------------------
    # Option B three-way seam (SEW-Offload, design doc 13). DEFAULT-OFF.
    #
    # When VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=1 *and* the config-level guards
    # pass, _select_forward returns _seam_forward_entry instead of the opaque
    # torch.ops.vllm.moe_forward. The entry calls three TOP-LEVEL ops --
    # moe_router_indirect | moe_offload_stage | moe_mlp -- so moe_offload_stage
    # becomes a real FX split point (it is in splitting_ops). The router + mlp
    # pieces are captured; staging runs eager between replays. seam=0 (or any
    # guard failing) -> base MoERunner._select_forward (monolithic moe_forward),
    # byte-for-byte the current path.
    #
    # All three ops receive the REAL layer name (self.layer_name) -> direct
    # no_compile_layers lookup with NO moe_layer_index increment. The only
    # runtime reader of moe_layer_index is the static-kernel wrap, which a guard
    # turns off on the seam path; so uniform real-name use is index-safe.
    # ----------------------------------------------------------------------
    def _seam_config_guards_pass(self) -> bool:
        """Config-level guards checkable at runner __init__ time."""
        import os as _os

        from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime

        def _probe(reason):
            if _os.environ.get("SEW_SEAM_PROBE"):
                print(
                    f"SEW_SEAM_SELECT layer={getattr(self, 'layer_name', '?')} "
                    f"config_guard={reason}",
                    flush=True,
                )

        runtime = get_moe_offload_runtime()
        if not runtime.config.offload_stage_seam:
            _probe("FAIL:offload_stage_seam_off")
            return False
        if self._shared_experts is not None:
            _probe("FAIL:shared_experts")
            return False
        # NOTE: a runner-held gate (is_internal_router; Qwen3-MoE always sets it)
        # is SUPPORTED -- moe_router_indirect applies the same gate to
        # hidden_states before select_experts, faithfully relocating the matmul
        # _forward_impl:710 would otherwise run. No gate guard here.
        mc = self.moe_config
        if mc.dp_size > 1 or mc.ep_size > 1 or mc.tp_size > 1 or mc.pcp_size > 1:
            _probe(
                f"FAIL:multicard dp={mc.dp_size} ep={mc.ep_size} "
                f"tp={mc.tp_size} pcp={mc.pcp_size}"
            )
            return False
        _probe("PASS")
        return True


    def _select_forward(self):
        if self._seam_config_guards_pass():
            return self._seam_forward_entry
        return super()._select_forward()

    def _seam_forward_entry(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        layer_name,
    ) -> torch.Tensor:
        """Drop-in replacement for the moe_forward custom op, decomposed into
        three top-level ops. Per-layer guards are resolved on first call and the
        decision is cached; any failure permanently delegates to moe_forward."""
        decision = getattr(self, "_seam_active", None)
        if decision is None:
            decision = self._resolve_seam_per_layer_guards()
            self._seam_active = decision

        if not decision:
            # Permanent fallback: the opaque monolithic op (same as base path).
            return torch.ops.vllm.moe_forward(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                layer_name,
            )

        real_name = self.layer_name
        topk_weights, topk_ids = torch.ops.vllm.moe_router_indirect(
            hidden_states,
            router_logits,
            real_name,
        )
        # Splitting op (in splitting_ops): eager D2H + stage + write log2phy buf.
        # Side-effect-only (returns None, declares mutates_args=["topk_ids"]): we
        # thread the SAME router-piece topk_ids tensor straight into moe_mlp so the
        # captured MLP reads it at the FIXED address recorded at capture. Returning
        # a clone here would land at a different address on each eager replay and
        # make the captured gather read a stale buffer -> MTE DDR out-of-range. The
        # declared mutation gives moe_mlp a real data dependency on the staging
        # side effect (prevents DCE / reorder) -- same contract as
        # unified_attention_with_output.
        torch.ops.vllm.moe_offload_stage(
            topk_ids,
            self._seam_layer_id,
            self._seam_num_logical_experts,
        )
        return torch.ops.vllm.moe_mlp(
            hidden_states,
            router_logits,
            topk_weights,
            topk_ids,
            shared_experts_input,
            input_ids,
            real_name,
        )

    def _resolve_seam_per_layer_guards(self) -> bool:
        """Resolve the layer once, check per-layer guards, cache layer_id +
        num_logical_experts for the seam ops. Returns False (-> fallback) on any
        unsupported configuration."""
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            get_layer_from_name,
        )

        from vllm_ascend.quantization.methods.base import (
            get_moe_num_logical_experts,
        )

        try:
            layer = get_layer_from_name(self.layer_name)
        except Exception:
            return False

        # Callable cannot cross an op boundary (decision ②/B1 constraint).
        if getattr(layer, "custom_routing_function", None) is not None:
            return False
        # Decision ④: first version mutually exclusive with multistream gate.
        if getattr(layer, "multistream_overlap_gate", False):
            return False
        # moe_layer_index index-safety: real-name lookups don't advance the
        # index, so the static-kernel wrap (the only runtime reader) must be off.
        if getattr(layer, "enable_npugraph_ex_static_kernel", False):
            return False
        # zero-expert path is not supported under the fixed-slot seam yet.
        if getattr(layer, "zero_expert_num", 0) and getattr(
            layer, "zero_expert_type", None
        ) is not None:
            return False

        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        self._seam_layer_id = int(getattr(layer, "layer_id", -1))
        self._seam_num_logical_experts = get_moe_num_logical_experts(
            layer,
            layer.moe_config.num_experts,
            global_redundant_expert_num=getattr(
                layer, "global_redundant_expert_num", 0
            ),
            num_shared_experts=num_shared_experts,
        )
        return True


class AscendFusedMoE(FusedMoE):
    moe_counter = -1
    gate_stream: torch.npu.Stream | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]
        num_shared_experts = kwargs.get("n_shared_experts", 0)

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self._expert_map = None
        self.log2phy = None

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()
        # flashcommon3 gate stream
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
            AscendFusedMoE.gate_stream = torch.npu.Stream()
        if self.custom_routing_function is None and self.e_score_correction_bias is not None:
            vllm_config = get_current_vllm_config()
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )

        # init moe
        eplb_config = ascend_config.eplb_config
        self.mix_placement = getattr(ascend_config, "mix_placement", False)
        self.n_shared_experts = num_shared_experts
        num_experts += num_shared_experts if self.mix_placement else 0
        self.moe_config.num_experts = num_experts
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config, self.mix_placement, num_shared_experts
        )
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.local_num_experts = self.global_num_experts // self.ep_size
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        if self.dynamic_eplb:
            self.multi_stage = False
            self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()
            if eplb_config.eplb_policy_type == 3:
                self.multi_stage = True
                self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
                self.num_iter = eplb_config.expert_heat_collection_interval
                self.moe_load = torch.zeros((self.num_iter, self.local_num_experts), dtype=torch.int32, device="npu")

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod"):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel

        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()

        self.runner = AscendMoERunner(
            self.layer_name,
            self.moe_config,
            self.router,
            kwargs.get("routed_input_transform"),
            kwargs.pop("gate", None),
            kwargs.pop("shared_experts", None),
            self.quant_method,
            self.vllm_config.parallel_config.enable_dbo,
            routed_output_transform=kwargs.get("routed_output_transform"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0)
            if kwargs.get("apply_routed_scale_to_output", False)
            else 1.0,
        )

    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)

        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)

        return quant_type

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()
        if self.multi_stage:
            self.load_counter.zero_()

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.ensure_moe_quant_config_init()
        return self.runner.forward(
            hidden_states,
            router_logits,
        )

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
    ) -> torch.Tensor | FusedMoEResult:
        assert self.quant_method is not None

        forward_context = get_forward_context()
        # When static kernels are enabled, the forward pass runs twice (compilation + capture),
        # causing moe_layer_index to overflow. Wrap the index to prevent out-of-bounds errors.
        if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
            moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
            forward_context.moe_layer_index = moe_layer_index

        # Load balancing for token distribution among experts in dummy_run
        # TODO: The community only considers load balancing when DP > 1.
        # This approach may overlook some extreme scenarios.
        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        forward_context = get_forward_context()
        if self.multistream_overlap_gate:
            assert AscendFusedMoE.gate_stream is not None
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            AscendFusedMoE.gate_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
                # share_expert
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                moe_comm_type = _EXTRA_CTX.moe_comm_type
                if (
                    moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                    and not shared_expert_dp_enabled()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                set_flash_common3_context(shared_out=shared_out)

                topk_weights, topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    use_grouped_topk=self.use_grouped_topk,
                    renormalize=self.renormalize,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    num_experts=self.moe_config.num_experts,
                )

                if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)

                set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        mc2_mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape
        pertoken_scale = prepare_output.pertoken_scale

        # Make sure the default stream waits for the gate stream to finish.
        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendFusedMoE.gate_stream)

        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_experts=self.moe_config.num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        if self.dynamic_eplb:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
            )
            local_load = (
                expert_tokens
                if group_list_type == 1
                else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            )
            if self.multi_stage:
                cur_iter = torch.remainder(self.load_counter, self.num_iter)
                self.moe_load.index_add_(
                    dim=0, index=cur_iter, source=local_load.to(torch.int32, non_blocking=True).view(1, -1)
                )
                self.load_counter.add_(1)
            else:
                self.moe_load.add_(local_load)
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=getattr(self, "reduce_results", False),
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
            )
        else:
            # The vLLM FusedMoE forward_impl does not return events.
            return routed_out


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):
    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        ascend_config = get_ascend_config()
        # TODO: Enabling the mix placement in deepseek_v2.py
        # remove this part after the mix placement merged into vllm
        # https://github.com/vllm-project/vllm/pull/31256
        if ascend_config.mix_placement:
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled = mock_false
            rocm_aiter_ops.is_fused_moe_enabled = mock_false
        AscendFusedMoE.__init__(self, **kwargs)
        if ascend_config.mix_placement:
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled = mock_true
            rocm_aiter_ops.is_fused_moe_enabled = mock_true

        self._routed_input_transform = routed_input_transform
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        has_shared_experts = shared_experts is not None
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert and has_shared_experts
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate and has_shared_experts
        if enable_sp():
            logger.info_once("Sequence parallelism is enabled, shared experts are replicated for best performance.")

        self._gate = gate
        # Recreate the runner with the correct shared_experts parameter.
        # The parent class created the runner before self._shared_experts was set.
        # NOTE: must use self._shared_experts here, not self.shared_experts —
        # FusedMoE.shared_experts is a property that reads self.runner.shared_experts,
        # which at this point is still the stale runner built with shared_experts=None.
        self.runner = AscendMoERunner(
            self.layer_name,
            self.moe_config,
            self.router,
            self._routed_input_transform,
            self.gate,
            self._shared_experts,
            self.quant_method,
            self.vllm_config.parallel_config.enable_dbo,
            routed_output_transform=getattr(self, "_routed_output_transform", None),
        )

        if self.multistream_overlap_shared_expert:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = self.quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self._validate_shared_expert_consistency()
                return result

            self.quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _validate_shared_expert_consistency(self):
        """Validate that split shared expert computation matches integrated
        computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self._shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error("SharedFusedMoE shared experts split computation does not match the integrated computation.")
            logger.error("Max absolute difference: %s", diff.max().item())
            logger.error(
                "Integrated output - sum: %s, norm: %s", integrated_out.sum().item(), integrated_out.norm().item()
            )
            logger.error("Split output - sum: %s, norm: %s", split_out.sum().item(), split_out.norm().item())
            raise ValueError(
                "SharedFusedMoE shared experts split computation does not match the integrated computation."
            )
        logger.info_once("SharedFusedMoE shared experts split computation matches the integrated computation.")

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return False

    @property
    def use_dp_chunking(self) -> bool:
        """This func routes to the chunked forward path using the FlashInfer Cutlass kernel
        only when data parallelism (DP) is enabled. Thus just returning False in vllm-ascend
        """
        return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = AscendFusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # When shared experts are absent, the parent returns only fused_out;
        # otherwise it returns a (shared_out, fused_out) tuple.
        if self._shared_experts is None:
            return None, result
        return result

    def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
            # Ensure the shared experts wait for hidden_states to be ready.
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            # Execute the gate projection and activation concurrently with the
            # dispatch communication.
            maybe_wait_event(fused_moe_evts.before_dispatch)
            part1_out = self._shared_experts_part1(hidden_states)
            # Execute the down projection concurrently with the combine
            # communication.
            maybe_wait_event(fused_moe_evts.before_combine)
            shared_out = self._shared_experts_part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        # NOTE: This is exactly the opposite of
        # `maybe_all_reduce_tensor_model_parallel`
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        if self.multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        before_routed_experts = torch.npu.current_stream().record_event()
        fused_moe_results = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out

        if self._shared_experts is None:
            return routed_out

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out
        else:
            shared_out = self._forward_shared_experts(
                hidden_states,
                FusedMoEEvents(
                    before_routed_experts=before_routed_experts,
                    before_dispatch=fused_moe_results.before_dispatch_evt,
                    before_combine=fused_moe_results.before_combine_evt,
                ),
            )

        return shared_out, routed_out
