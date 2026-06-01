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

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TraceRecord:
    layer_id: int
    step_id: int
    mode: str
    num_tokens: int
    top_k: int
    num_experts: int
    active_experts: tuple[int, ...]
    expert_token_counts: dict[int, int]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "step_id": self.step_id,
            "mode": self.mode,
            "num_tokens": self.num_tokens,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "active_experts": list(self.active_experts),
            "expert_token_counts": {str(k): v for k, v in self.expert_token_counts.items()},
        }


class TraceCollector:
    def __init__(self, max_records: int = 4096) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be greater than 0")
        self._records: deque[TraceRecord] = deque(maxlen=max_records)
        self._latest_by_layer: dict[int, TraceRecord] = {}

    def record(
        self,
        *,
        layer_id: int,
        step_id: int,
        topk_ids: torch.Tensor,
        num_experts: int,
        mode: str = "unknown",
    ) -> TraceRecord:
        if topk_ids.ndim == 0:
            raise ValueError("topk_ids must have at least one dimension")

        detached_ids = topk_ids.detach()
        if detached_ids.device.type != "cpu":
            detached_ids = detached_ids.to("cpu")
        flattened_ids = detached_ids.reshape(-1).to(torch.int64)

        counts: dict[int, int] = {}
        for expert_id in flattened_ids.tolist():
            expert = int(expert_id)
            if expert < 0:
                continue
            counts[expert] = counts.get(expert, 0) + 1

        record = TraceRecord(
            layer_id=int(layer_id),
            step_id=int(step_id),
            mode=mode,
            num_tokens=int(topk_ids.shape[0]),
            top_k=int(topk_ids.shape[1]) if topk_ids.ndim > 1 else 1,
            num_experts=int(num_experts),
            active_experts=tuple(sorted(counts)),
            expert_token_counts=dict(sorted(counts.items())),
        )
        self._records.append(record)
        self._latest_by_layer[record.layer_id] = record
        return record

    def latest_for_layer(self, layer_id: int) -> TraceRecord | None:
        return self._latest_by_layer.get(int(layer_id))

    def records(self) -> list[TraceRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._latest_by_layer.clear()

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [record.to_jsonable() for record in self._records]

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(record.to_jsonable(), sort_keys=True) + "\n" for record in self._records
        )

    def write_jsonl(self, path: str | Path) -> int:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_jsonl(), encoding="utf-8")
        return len(self._records)
