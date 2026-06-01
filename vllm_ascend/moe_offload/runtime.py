#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from dataclasses import dataclass
from itertools import count
from pathlib import Path

import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.expert_key import ExpertKey
from vllm_ascend.moe_offload.host_store import HostExpertStore
from vllm_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState
from vllm_ascend.moe_offload.slot_mapping import ExpertSlotMapping, PreparedSlotWeights
from vllm_ascend.moe_offload.trace_collector import TraceCollector
from vllm_ascend.moe_offload.transfer_engine import TransferEngine


@dataclass(frozen=True)
class MoeOffloadMemoryLedger:
    registered_layers: int
    host_experts: int
    original_expert_weight_bytes: int
    host_store_bytes: int
    slot_bank_bytes: int

    @property
    def original_expert_weights_retained(self) -> bool:
        return self.original_expert_weight_bytes > 0

    @property
    def total_managed_bytes(self) -> int:
        return self.original_expert_weight_bytes + self.host_store_bytes + self.slot_bank_bytes


@dataclass(frozen=True)
class MoeExpertReleasePlan:
    ready: bool
    layers_ready: tuple[int, ...]
    blockers: tuple[str, ...]


class MoeOffloadRuntime:
    def __init__(self, config: MoeOffloadConfig | None = None) -> None:
        self.config = config if config is not None else MoeOffloadConfig.from_env()
        self.trace_collector = TraceCollector(max_records=self.config.trace_max_records)
        self._step_counter = count()
        self._host_store = HostExpertStore()
        self._slot_banks: dict[int, ExpertSlotBank] = {}
        self._original_expert_weight_bytes_by_layer: dict[int, int] = {}
        self._transfer_engine = TransferEngine()

    def trace_routing(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        mode: str = "unknown",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.should_trace:
            self.trace_collector.record(
                layer_id=layer_id,
                step_id=next(self._step_counter),
                topk_ids=topk_ids,
                num_experts=num_experts,
                mode=mode,
            )
        return topk_ids, topk_weights

    def export_trace(self, path: str | Path) -> int:
        return self.trace_collector.write_jsonl(path)

    @property
    def should_use_fixed_slots(self) -> bool:
        return self.config.enabled and not self.config.trace_only and self.config.num_slots > 0

    def register_layer_for_fixed_slots(
        self,
        layer: torch.nn.Module,
        *,
        slot_device: torch.device | None = None,
    ) -> None:
        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            raise ValueError("layer.layer_id is required for fixed-slot registration")

        self._host_store.register_layer(layer)
        w13_weight = getattr(layer, "w13_weight")
        w2_weight = getattr(layer, "w2_weight")
        self._original_expert_weight_bytes_by_layer[layer_id] = _tensor_nbytes(w13_weight) + _tensor_nbytes(w2_weight)
        device = slot_device if slot_device is not None else w13_weight.device
        self._slot_banks[layer_id] = ExpertSlotBank(
            self.config.num_slots,
            tuple(int(dim) for dim in w13_weight.shape[1:]),
            tuple(int(dim) for dim in w2_weight.shape[1:]),
            dtype=w13_weight.dtype,
            device=device,
        )

    def is_layer_registered(self, layer_id: int) -> bool:
        return int(layer_id) in self._slot_banks

    def memory_ledger(self) -> MoeOffloadMemoryLedger:
        return MoeOffloadMemoryLedger(
            registered_layers=len(self._slot_banks),
            host_experts=len(self._host_store),
            original_expert_weight_bytes=sum(self._original_expert_weight_bytes_by_layer.values()),
            host_store_bytes=self._host_store.total_bytes,
            slot_bank_bytes=sum(slot_bank.total_bytes for slot_bank in self._slot_banks.values()),
        )

    def plan_original_weight_release(
        self,
        *,
        expected_layer_ids: tuple[int, ...],
        default_path_preserved: bool,
        host_store_is_complete: bool | None = None,
        allow_retained_original_weights: bool = False,
    ) -> MoeExpertReleasePlan:
        normalized_layer_ids = tuple(int(layer_id) for layer_id in expected_layer_ids)
        blockers: list[str] = []
        if not normalized_layer_ids:
            blockers.append("no_expected_layers")

        missing_layers = tuple(layer_id for layer_id in normalized_layer_ids if layer_id not in self._slot_banks)
        if missing_layers:
            blockers.append(f"layers_not_registered:{list(missing_layers)}")

        if not default_path_preserved:
            blockers.append("default_path_not_preserved")
        if host_store_is_complete is False:
            blockers.append("host_store_not_marked_complete")

        host_store_report = self._host_store.validate_complete_layers(normalized_layer_ids)
        blockers.extend(host_store_report.blockers)
        if self.memory_ledger().original_expert_weights_retained and not allow_retained_original_weights:
            blockers.append("original_expert_weights_still_retained")

        layers_ready = () if blockers else normalized_layer_ids
        return MoeExpertReleasePlan(
            ready=not blockers,
            layers_ready=layers_ready,
            blockers=tuple(blockers),
        )

    def prepare_fixed_slot_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
    ) -> PreparedSlotWeights:
        if not self.should_use_fixed_slots:
            raise RuntimeError("fixed-slot plan requested while moe offload fixed slots are disabled")

        layer_id = int(layer_id)
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"active expert working set size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")

        step_id = next(self._step_counter)
        for expert_id in unique_active_experts:
            key = ExpertKey(layer_id, int(expert_id))
            slot = slot_bank.lookup(key)
            if slot is not None and slot.state == SlotState.READY:
                slot.last_used_step = int(step_id)
                continue

            slot = slot_bank.allocate_for(key, step_id=step_id)
            bundle = self._host_store.get(layer_id, int(expert_id))
            self._transfer_engine.load_sync(bundle, slot)

        mapping = ExpertSlotMapping.from_slot_bank(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
            slot_bank=slot_bank,
            device=device,
        )
        return PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)

    def prepare_weights_for_execution(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
    ) -> None:
        del layer_id, active_experts
        if not self.should_use_fixed_slots:
            return None
        raise NotImplementedError(
            "fixed-slot execution requires num_logical_experts and backend wiring; "
            "use prepare_fixed_slot_plan() for the current safe planning path"
        )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


_runtime: MoeOffloadRuntime | None = None


def get_moe_offload_runtime() -> MoeOffloadRuntime:
    global _runtime
    if _runtime is None:
        _runtime = MoeOffloadRuntime()
    return _runtime


def reset_moe_offload_runtime() -> None:
    global _runtime
    _runtime = None


def _dedupe_preserve_order(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            deduped.append(int_value)
    return tuple(deduped)


def _validate_active_expert_ids(
    *,
    layer_id: int,
    active_experts: tuple[int, ...],
    num_logical_experts: int,
) -> None:
    invalid_expert_ids = [
        int(expert_id)
        for expert_id in active_experts
        if int(expert_id) < 0 or int(expert_id) >= int(num_logical_experts)
    ]
    if invalid_expert_ids:
        raise ValueError(
            "fixed-slot active expert id out of range: "
            f"layer_id={int(layer_id)}, "
            f"num_logical_experts={int(num_logical_experts)}, "
            f"expert_ids={invalid_expert_ids}"
        )
