# SEW-Offload Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vLLM Ascend 中以默认关闭、低侵入方式实现 SEW-Offload 的 trace、slot、prefetch 和 phased execution 原型。

**Architecture:** 新增 `vllm_ascend/moe_offload/` 独立包，围绕 `HostExpertStore`、`ExpertSlotBank`、`PrefetchPlanner`、`TransferEngine`、`PhaseScheduler` 和 `CostModel` 组织。首个可运行版本从 trace-only 和 simulator 开始，再接入 fixed slot 与同步加载，最后加入异步 prefetch 与 hit-first phased execution。

**Tech Stack:** Python, PyTorch/torch-npu, vLLM Ascend fused MoE, pytest, Ascend 910B3, CANN 8.5.1.

---

## 1. 文件结构

计划新增：

```text
vllm_ascend/moe_offload/
  __init__.py
  config.py
  trace_collector.py
  cost_model.py
  host_store.py
  slot_bank.py
  prefetch_planner.py
  transfer_engine.py
  phase_scheduler.py
  runtime.py

tests/ut/moe_offload/
  test_config.py
  test_trace_collector.py
  test_cost_model.py
  test_slot_bank.py
  test_prefetch_planner.py
  test_phase_scheduler.py
  test_runtime_trace_only.py

benchmarks/sew_offload/
  collect_moe_trace.py
  simulate_slot_policy.py
  run_single_card_offload.sh
```

计划修改：

```text
vllm_ascend/envs.py
vllm_ascend/ops/fused_moe/fused_moe.py
vllm_ascend/ops/fused_moe/moe_mlp.py
```

修改原则：

- `envs.py` 只增加默认关闭的环境变量。
- `fused_moe.py` 只加最小 hook。
- `moe_mlp.py` 只在进入 grouped MLP 前允许 runtime 替换 slot weights 或 phase plan。
- 不修改 router。
- 不修改 top-k。
- 不修改现有 token dispatcher 语义。

## 2. 实施阶段总览

| 阶段 | 目标 | 是否改变执行 |
| --- | --- | --- |
| MVP-0 | trace-only，记录 expert 工作集 | 否 |
| MVP-1 | offload simulator，离线评估 slot/prefetch 策略 | 否 |
| MVP-2 | fixed slot + sync load，跑通 correctness | 是 |
| MVP-3 | async prefetch，降低 blocking miss | 是 |
| MVP-4 | hit-first phased execution，隐藏 miss load | 是 |
| MVP-5 | graph/static window 与 split-weight prefetch | 是 |

## Task 1: 新增配置入口

**Files:**

- Create: `vllm_ascend/moe_offload/config.py`
- Modify: `vllm_ascend/envs.py`
- Test: `tests/ut/moe_offload/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_config.py
import os

from vllm_ascend.moe_offload.config import MoeOffloadConfig


def test_default_config_is_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", raising=False)
    cfg = MoeOffloadConfig.from_env()
    assert cfg.enabled is False
    assert cfg.trace_only is False
    assert cfg.num_slots == 0
    assert cfg.max_phases == 1


def test_env_config_parses_values(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "8")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", "2")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY", "deadline")

    cfg = MoeOffloadConfig.from_env()

    assert cfg.enabled is True
    assert cfg.trace_only is True
    assert cfg.num_slots == 8
    assert cfg.max_phases == 2
    assert cfg.prefetch_policy == "deadline"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/ut/moe_offload/test_config.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'vllm_ascend.moe_offload'
```

- [ ] **Step 3: 实现配置类**

```python
# vllm_ascend/moe_offload/config.py
from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class MoeOffloadConfig:
    enabled: bool = False
    trace_only: bool = False
    num_slots: int = 0
    max_phases: int = 1
    prefetch_policy: str = "none"

    @classmethod
    def from_env(cls) -> "MoeOffloadConfig":
        return cls(
            enabled=_env_bool("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", False),
            trace_only=_env_bool("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", False),
            num_slots=_env_int("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", 0),
            max_phases=_env_int("VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", 1),
            prefetch_policy=os.getenv(
                "VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY", "none"
            ),
        )
```

```python
# vllm_ascend/moe_offload/__init__.py
from vllm_ascend.moe_offload.config import MoeOffloadConfig

__all__ = ["MoeOffloadConfig"]
```

- [ ] **Step 4: 在 envs.py 注册环境变量**

在 `vllm_ascend/envs.py` 的 `env_variables` 字典中新增：

```python
"VLLM_ASCEND_MOE_OFFLOAD_ENABLED": lambda: bool(
    int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "0"))
),
"VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": lambda: bool(
    int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "0"))
),
"VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": lambda: int(
    os.getenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "0")
),
"VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES": lambda: int(
    os.getenv("VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", "1")
),
"VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY": lambda: os.getenv(
    "VLLM_ASCEND_MOE_OFFLOAD_PREFETCH_POLICY", "none"
),
```

- [ ] **Step 5: 运行测试确认通过**

Run:

```bash
pytest tests/ut/moe_offload/test_config.py -q
```

Expected:

```text
2 passed
```

## Task 2: TraceCollector

**Files:**

- Create: `vllm_ascend/moe_offload/trace_collector.py`
- Test: `tests/ut/moe_offload/test_trace_collector.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_trace_collector.py
import torch

from vllm_ascend.moe_offload.trace_collector import TraceCollector


def test_trace_collector_records_active_experts():
    collector = TraceCollector(max_records=4)
    topk_ids = torch.tensor([[0, 2], [2, 3], [3, 3]])

    record = collector.record(layer_id=7, step_id=11, topk_ids=topk_ids)

    assert record.layer_id == 7
    assert record.step_id == 11
    assert record.expert_token_counts == {0: 1, 2: 2, 3: 3}
    assert collector.latest_for_layer(7) == record
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/ut/moe_offload/test_trace_collector.py -q
```

Expected:

```text
ImportError
```

- [ ] **Step 3: 实现 TraceCollector**

```python
# vllm_ascend/moe_offload/trace_collector.py
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertTraceRecord:
    layer_id: int
    step_id: int
    expert_token_counts: dict[int, int]


class TraceCollector:
    def __init__(self, max_records: int = 4096) -> None:
        self._records: deque[ExpertTraceRecord] = deque(maxlen=max_records)
        self._latest_by_layer: dict[int, ExpertTraceRecord] = {}

    def record(
        self, layer_id: int, step_id: int, topk_ids: torch.Tensor
    ) -> ExpertTraceRecord:
        ids = topk_ids.detach().cpu().reshape(-1).tolist()
        counts: dict[int, int] = {}
        for expert_id in ids:
            expert_id = int(expert_id)
            counts[expert_id] = counts.get(expert_id, 0) + 1

        record = ExpertTraceRecord(
            layer_id=layer_id,
            step_id=step_id,
            expert_token_counts=counts,
        )
        self._records.append(record)
        self._latest_by_layer[layer_id] = record
        return record

    def latest_for_layer(self, layer_id: int) -> ExpertTraceRecord | None:
        return self._latest_by_layer.get(layer_id)

    def records(self) -> list[ExpertTraceRecord]:
        return list(self._records)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/ut/moe_offload/test_trace_collector.py -q
```

Expected:

```text
1 passed
```

## Task 3: ExpertSlotBank

**Files:**

- Create: `vllm_ascend/moe_offload/slot_bank.py`
- Test: `tests/ut/moe_offload/test_slot_bank.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_slot_bank.py
from vllm_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState


def test_slot_bank_tracks_expert_mapping():
    bank = ExpertSlotBank(num_slots=2)

    slot = bank.assign(expert_key=(3, 17))

    assert slot.slot_id == 0
    assert slot.expert_key == (3, 17)
    assert slot.state == SlotState.READY
    assert bank.lookup((3, 17)).slot_id == 0


def test_slot_bank_refuses_to_evict_computing_slot():
    bank = ExpertSlotBank(num_slots=1)
    slot = bank.assign(expert_key=(0, 1))
    bank.mark_computing(slot.slot_id)

    victim = bank.choose_victim(protected_experts=set())

    assert victim is None
```

- [ ] **Step 2: 实现 slot 状态机**

```python
# vllm_ascend/moe_offload/slot_bank.py
from dataclasses import dataclass
from enum import Enum


class SlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    COMPUTING = "computing"
    EVICTABLE = "evictable"


ExpertKey = tuple[int, int]


@dataclass
class ExpertSlot:
    slot_id: int
    expert_key: ExpertKey | None = None
    state: SlotState = SlotState.EMPTY


class ExpertSlotBank:
    def __init__(self, num_slots: int) -> None:
        self.slots = [ExpertSlot(slot_id=i) for i in range(num_slots)]
        self._expert_to_slot: dict[ExpertKey, int] = {}

    def lookup(self, expert_key: ExpertKey) -> ExpertSlot | None:
        slot_id = self._expert_to_slot.get(expert_key)
        if slot_id is None:
            return None
        return self.slots[slot_id]

    def assign(self, expert_key: ExpertKey) -> ExpertSlot:
        existing = self.lookup(expert_key)
        if existing is not None:
            return existing

        slot = self._find_empty_slot()
        if slot is None:
            slot = self.choose_victim(protected_experts=set())
        if slot is None:
            raise RuntimeError("No available expert slot")

        if slot.expert_key is not None:
            self._expert_to_slot.pop(slot.expert_key, None)

        slot.expert_key = expert_key
        slot.state = SlotState.READY
        self._expert_to_slot[expert_key] = slot.slot_id
        return slot

    def mark_loading(self, slot_id: int) -> None:
        self.slots[slot_id].state = SlotState.LOADING

    def mark_ready(self, slot_id: int) -> None:
        self.slots[slot_id].state = SlotState.READY

    def mark_computing(self, slot_id: int) -> None:
        self.slots[slot_id].state = SlotState.COMPUTING

    def mark_evictable(self, slot_id: int) -> None:
        self.slots[slot_id].state = SlotState.EVICTABLE

    def choose_victim(
        self, protected_experts: set[ExpertKey]
    ) -> ExpertSlot | None:
        for slot in self.slots:
            if slot.expert_key in protected_experts:
                continue
            if slot.state in {SlotState.READY, SlotState.EVICTABLE}:
                return slot
        return None

    def _find_empty_slot(self) -> ExpertSlot | None:
        for slot in self.slots:
            if slot.state == SlotState.EMPTY:
                return slot
        return None
```

- [ ] **Step 3: 运行测试**

Run:

```bash
pytest tests/ut/moe_offload/test_slot_bank.py -q
```

Expected:

```text
2 passed
```

## Task 4: PrefetchPlanner

**Files:**

- Create: `vllm_ascend/moe_offload/prefetch_planner.py`
- Test: `tests/ut/moe_offload/test_prefetch_planner.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_prefetch_planner.py
from vllm_ascend.moe_offload.prefetch_planner import (
    ExpertCandidate,
    PrefetchPlanner,
)


def test_planner_prioritizes_near_deadline_and_high_token_count():
    planner = PrefetchPlanner()
    candidates = [
        ExpertCandidate((0, 1), p_use=0.9, token_count=8, load_penalty_ms=4, deadline_ms=10),
        ExpertCandidate((0, 2), p_use=0.8, token_count=64, load_penalty_ms=4, deadline_ms=10),
        ExpertCandidate((0, 3), p_use=0.8, token_count=64, load_penalty_ms=4, deadline_ms=100),
    ]

    plan = planner.plan(candidates=candidates, available_slots=2)

    assert [item.expert_key for item in plan] == [(0, 2), (0, 1)]
```

- [ ] **Step 2: 实现 planner**

```python
# vllm_ascend/moe_offload/prefetch_planner.py
from dataclasses import dataclass


ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class ExpertCandidate:
    expert_key: ExpertKey
    p_use: float
    token_count: int
    load_penalty_ms: float
    deadline_ms: float


@dataclass(frozen=True)
class PrefetchPlanItem:
    expert_key: ExpertKey
    priority: float


class PrefetchPlanner:
    def plan(
        self,
        candidates: list[ExpertCandidate],
        available_slots: int,
    ) -> list[PrefetchPlanItem]:
        scored = [
            PrefetchPlanItem(
                expert_key=c.expert_key,
                priority=self._score(c),
            )
            for c in candidates
        ]
        scored.sort(key=lambda item: item.priority, reverse=True)
        return scored[:available_slots]

    def _score(self, candidate: ExpertCandidate) -> float:
        deadline = max(candidate.deadline_ms, 1e-6)
        return (
            candidate.p_use
            * candidate.token_count
            * candidate.load_penalty_ms
            / deadline
        )
```

- [ ] **Step 3: 运行测试**

Run:

```bash
pytest tests/ut/moe_offload/test_prefetch_planner.py -q
```

Expected:

```text
1 passed
```

## Task 5: CostModel

**Files:**

- Create: `vllm_ascend/moe_offload/cost_model.py`
- Test: `tests/ut/moe_offload/test_cost_model.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_cost_model.py
from vllm_ascend.moe_offload.cost_model import CostModel


def test_cost_model_decides_when_phase_split_is_profitable():
    model = CostModel(split_overhead_ms=0.2, useful_overlap_threshold_ms=0.1)

    assert model.should_split(
        predicted_load_ms=3.0,
        predicted_hit_compute_ms=2.0,
    )
    assert not model.should_split(
        predicted_load_ms=0.1,
        predicted_hit_compute_ms=2.0,
    )
```

- [ ] **Step 2: 实现 CostModel**

```python
# vllm_ascend/moe_offload/cost_model.py
from dataclasses import dataclass


@dataclass
class CostModel:
    split_overhead_ms: float = 0.2
    useful_overlap_threshold_ms: float = 0.1

    def should_split(
        self,
        predicted_load_ms: float,
        predicted_hit_compute_ms: float,
    ) -> bool:
        overlap = min(predicted_load_ms, predicted_hit_compute_ms)
        benefit = overlap - self.split_overhead_ms
        return benefit > self.useful_overlap_threshold_ms
```

- [ ] **Step 3: 运行测试**

Run:

```bash
pytest tests/ut/moe_offload/test_cost_model.py -q
```

Expected:

```text
1 passed
```

## Task 6: PhaseScheduler

**Files:**

- Create: `vllm_ascend/moe_offload/phase_scheduler.py`
- Test: `tests/ut/moe_offload/test_phase_scheduler.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/ut/moe_offload/test_phase_scheduler.py
from vllm_ascend.moe_offload.cost_model import CostModel
from vllm_ascend.moe_offload.phase_scheduler import PhaseScheduler


def test_phase_scheduler_creates_hit_and_miss_phases():
    scheduler = PhaseScheduler(
        cost_model=CostModel(split_overhead_ms=0.1),
        max_phases=2,
    )

    plan = scheduler.plan(
        hit_experts={(0, 1), (0, 2)},
        miss_experts={(0, 3)},
        predicted_load_ms=2.0,
        predicted_hit_compute_ms=1.5,
    )

    assert len(plan.phases) == 2
    assert plan.phases[0].kind == "hit"
    assert plan.phases[0].experts == {(0, 1), (0, 2)}
    assert plan.phases[1].kind == "miss"
    assert plan.phases[1].experts == {(0, 3)}
```

- [ ] **Step 2: 实现 scheduler**

```python
# vllm_ascend/moe_offload/phase_scheduler.py
from dataclasses import dataclass

from vllm_ascend.moe_offload.cost_model import CostModel

ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class ExecutionPhase:
    kind: str
    experts: set[ExpertKey]


@dataclass(frozen=True)
class PhasePlan:
    phases: list[ExecutionPhase]


class PhaseScheduler:
    def __init__(self, cost_model: CostModel, max_phases: int = 1) -> None:
        self.cost_model = cost_model
        self.max_phases = max_phases

    def plan(
        self,
        hit_experts: set[ExpertKey],
        miss_experts: set[ExpertKey],
        predicted_load_ms: float,
        predicted_hit_compute_ms: float,
    ) -> PhasePlan:
        if not miss_experts:
            return PhasePlan([ExecutionPhase(kind="all", experts=hit_experts)])

        should_split = (
            self.max_phases >= 2
            and bool(hit_experts)
            and self.cost_model.should_split(
                predicted_load_ms=predicted_load_ms,
                predicted_hit_compute_ms=predicted_hit_compute_ms,
            )
        )
        if should_split:
            return PhasePlan(
                [
                    ExecutionPhase(kind="hit", experts=hit_experts),
                    ExecutionPhase(kind="miss", experts=miss_experts),
                ]
            )

        return PhasePlan(
            [
                ExecutionPhase(
                    kind="all",
                    experts=set(hit_experts) | set(miss_experts),
                )
            ]
        )
```

- [ ] **Step 3: 运行测试**

Run:

```bash
pytest tests/ut/moe_offload/test_phase_scheduler.py -q
```

Expected:

```text
1 passed
```

## Task 7: Runtime trace-only hook

**Files:**

- Create: `vllm_ascend/moe_offload/runtime.py`
- Modify: `vllm_ascend/ops/fused_moe/fused_moe.py`
- Test: `tests/ut/moe_offload/test_runtime_trace_only.py`

- [ ] **Step 1: 写 runtime 单测**

```python
# tests/ut/moe_offload/test_runtime_trace_only.py
import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime


def test_trace_only_runtime_records_and_returns_none():
    runtime = MoeOffloadRuntime(
        config=MoeOffloadConfig(enabled=True, trace_only=True)
    )
    topk_ids = torch.tensor([[1, 2], [2, 2]])

    result = runtime.observe_layer(
        layer_id=0,
        step_id=0,
        topk_ids=topk_ids,
    )

    assert result is None
    assert runtime.trace_collector.latest_for_layer(0).expert_token_counts == {
        1: 1,
        2: 3,
    }
```

- [ ] **Step 2: 实现 runtime**

```python
# vllm_ascend/moe_offload/runtime.py
import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.trace_collector import TraceCollector


class MoeOffloadRuntime:
    def __init__(self, config: MoeOffloadConfig) -> None:
        self.config = config
        self.trace_collector = TraceCollector()

    def observe_layer(
        self,
        layer_id: int,
        step_id: int,
        topk_ids: torch.Tensor,
    ) -> None:
        if not self.config.enabled:
            return None
        self.trace_collector.record(
            layer_id=layer_id,
            step_id=step_id,
            topk_ids=topk_ids,
        )
        return None
```

- [ ] **Step 3: 在 fused_moe.py 加最小 hook**

要求：

- hook 必须被 `config.enabled` 保护。
- trace-only 模式不得改变任何 tensor。
- 如果 runtime 未初始化，必须走原路径。

建议伪代码：

```python
runtime = get_moe_offload_runtime_if_enabled()
if runtime is not None:
    runtime.observe_layer(
        layer_id=layer_idx,
        step_id=current_step_id,
        topk_ids=topk_ids,
    )
```

`layer_idx` 和 `step_id` 的来源需优先使用 vLLM Ascend 现有 forward context；如果拿不到 `step_id`，MVP 可以使用 runtime 内部递增计数。

- [ ] **Step 4: 运行相关单测**

Run:

```bash
pytest tests/ut/moe_offload/test_runtime_trace_only.py -q
```

Expected:

```text
1 passed
```

## Task 8: Offline simulator

**Files:**

- Create: `benchmarks/sew_offload/simulate_slot_policy.py`
- Test: `tests/ut/moe_offload/test_prefetch_planner.py`, `tests/ut/moe_offload/test_slot_bank.py`

- [ ] **Step 1: 定义 trace JSONL 格式**

每行一个 layer record：

```json
{"step_id": 0, "layer_id": 12, "expert_token_counts": {"3": 18, "7": 4}}
```

- [ ] **Step 2: 实现 simulator 输入输出**

脚本参数：

```bash
python benchmarks/sew_offload/simulate_slot_policy.py \
  --trace-jsonl path/to/trace.jsonl \
  --num-slots 8 \
  --policy deadline
```

输出字段：

```text
slot_hit_rate
miss_count
predicted_load_ms
predicted_overlap_ms
predicted_exposed_stall_ms
```

- [ ] **Step 3: 使用 synthetic trace 验证**

Run:

```bash
python benchmarks/sew_offload/simulate_slot_policy.py \
  --trace-jsonl tests/fixtures/moe_offload/synthetic_trace.jsonl \
  --num-slots 2 \
  --policy deadline
```

Expected:

```text
slot_hit_rate:
predicted_exposed_stall_ms:
```

## Task 9: Sync fixed-slot execution

**Files:**

- Create: `vllm_ascend/moe_offload/host_store.py`
- Create: `vllm_ascend/moe_offload/transfer_engine.py`
- Modify: `vllm_ascend/moe_offload/runtime.py`
- Test: `tests/ut/moe_offload/test_slot_bank.py`

- [ ] **Step 1: HostExpertStore 先只保存 metadata**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertWeightRef:
    layer_id: int
    expert_id: int
    gate_up_name: str
    down_name: str


class HostExpertStore:
    def __init__(self) -> None:
        self._weights: dict[tuple[int, int], ExpertWeightRef] = {}

    def register(self, ref: ExpertWeightRef) -> None:
        self._weights[(ref.layer_id, ref.expert_id)] = ref

    def get(self, layer_id: int, expert_id: int) -> ExpertWeightRef:
        return self._weights[(layer_id, expert_id)]
```

- [ ] **Step 2: TransferEngine MVP 使用同步 copy**

首版只要求接口成立：

```python
class TransferEngine:
    def load_sync(self, expert_key, target_slot) -> None:
        # Copy host expert weights into target HBM slot.
        # Exact tensor copy is implemented after HostExpertStore binds real tensors.
        target_slot.state = SlotState.READY
```

- [ ] **Step 3: correctness 验证**

在单层或 mock MoE 上验证：

- slot miss 时加载正确 expert。
- slot hit 时不重复加载。
- replacement 不驱逐当前计算 expert。

## Task 10: Async prefetch and phased execution

**Files:**

- Modify: `vllm_ascend/moe_offload/transfer_engine.py`
- Modify: `vllm_ascend/moe_offload/phase_scheduler.py`
- Modify: `vllm_ascend/moe_offload/runtime.py`

- [ ] **Step 1: TransferEngine 增加 async load 状态**

需要记录：

```python
LoadTicket(
    expert_key=(layer_id, expert_id),
    slot_id=slot_id,
    start_time_ns=...,
    done_event=...,
)
```

- [ ] **Step 2: PhaseScheduler 接入 runtime**

runtime 在每层执行前生成：

```text
hit_experts
miss_experts
phase_plan
```

- [ ] **Step 3: 实现 fallback**

如果 async load 没有按 deadline 完成：

```text
wait -> run miss phase -> update CostModel
```

不允许 drop expert。

## 3. 验证命令

基础单测：

```bash
pytest tests/ut/moe_offload -q
```

MoE 相关单测：

```bash
pytest tests/ut/ops/test_fused_moe.py -q
pytest tests/ut/ops/test_moe_runtime_args.py -q
```

trace-only 手工验证：

```bash
VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1 \
pytest tests/e2e/multicard/2-cards/test_qwen3_moe_routing_replay.py -q
```

如果当前环境没有 `torch/torch_npu/vllm`，上述运行命令应在 vLLM Ascend 正确开发容器或 conda 环境中执行。

## 4. Commit 切分建议

1. `docs: add sew offload project docs`
2. `feat(moe-offload): add config and trace collector`
3. `feat(moe-offload): add slot bank and prefetch planner`
4. `feat(moe-offload): add phase scheduler and cost model`
5. `feat(moe-offload): add trace-only fused moe hook`
6. `feat(moe-offload): add slot policy simulator`
7. `feat(moe-offload): add sync fixed-slot prototype`
8. `feat(moe-offload): add async prefetch prototype`

## 5. Plan self-review

- Spec coverage：覆盖 `01-system-design.md` 中的配置、trace、slot、prefetch、phase、cost model、runtime hook、实验 simulator。
- Placeholder scan：本文没有遗留空白实现项；后续 tensor copy 位置明确标记为进入 HostExpertStore 绑定真实 tensor 后实现。
- Type consistency：统一使用 `ExpertKey = tuple[int, int]` 表示 `(layer_id, expert_id)`。
- Scope check：本计划只覆盖 runtime 原型，不覆盖论文撰写和 slide 修改。
