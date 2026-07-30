import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_ascend import envs


SCHEMA_VERSION = "adm-runtime-observation/v1"
PATHS = frozenset({"dp1", "skip", "cpu", "npu"})
SCOPES = frozenset({"local", "global"})
INT32_MAX = 2**31 - 1

_OBSERVATION_FIELDS = frozenset(
    {
        "event_index",
        "path",
        "snapshot_scope",
        "num_tokens",
        "cudagraph_mode",
        "collective_enter_ns",
        "pack_ns",
        "collective_ns",
        "copy_to_host_ns",
        "total_ns",
    }
)


class ObservationViolation(ValueError):
    """A runtime observation violates the fail-closed schema."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeObservation:
    host_id: str
    pid: int
    rank: int
    dp_size: int
    event_index: int
    path: str
    snapshot_scope: str
    num_tokens: tuple[int, ...]
    cudagraph_mode: tuple[int, ...]
    collective_enter_ns: int | None
    pack_ns: int
    collective_ns: int | None
    copy_to_host_ns: int | None
    total_ns: int

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "observation",
            "host_id": self.host_id,
            "pid": self.pid,
            "rank": self.rank,
            "dp_size": self.dp_size,
            "event_index": self.event_index,
            "path": self.path,
            "snapshot_scope": self.snapshot_scope,
            "num_tokens": list(self.num_tokens),
            "cudagraph_mode": list(self.cudagraph_mode),
            "clock_domain": "host_monotonic",
            "collective_enter_ns": self.collective_enter_ns,
            "pack_ns": self.pack_ns,
            "collective_ns": self.collective_ns,
            "copy_to_host_ns": self.copy_to_host_ns,
            "total_ns": self.total_ns,
        }


def _is_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_optional_timing(
    value: Any,
    field: str,
) -> int | None:
    if value is None:
        return None
    if not _is_non_negative_int(value):
        raise ObservationViolation(
            "invalid_timing",
            f"{field} must be non-negative integer nanoseconds or null",
        )
    return value


def _validate_values(
    value: Any,
    *,
    field: str,
    expected_size: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != expected_size:
        raise ObservationViolation(
            "rank_shape_mismatch",
            f"{field} must contain {expected_size} value(s)",
        )

    if field == "num_tokens":
        valid = all(
            type(item) is int and 0 <= item <= INT32_MAX
            for item in value
        )
    else:
        valid = all(
            type(item) is int and item in {0, 1, 2}
            for item in value
        )

    if not valid:
        raise ObservationViolation(
            "invalid_field_value",
            f"{field} contains an unsupported value",
        )
    return tuple(value)


class ADMRuntimeObserver:
    """Bounded synchronous recorder for raw ADM runtime observations."""

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        rank: int,
        dp_size: int,
        max_samples: int = 4096,
        host_id: str,
        pid: int,
    ) -> None:
        if (
            type(dp_size) is not int
            or dp_size <= 0
            or type(rank) is not int
            or not 0 <= rank < dp_size
            or type(max_samples) is not int
            or max_samples <= 0
            or not isinstance(host_id, str)
            or not host_id
            or type(pid) is not int
            or pid <= 0
        ):
            raise ObservationViolation(
                "invalid_configuration",
                "rank, DP size, sample bound, host and PID must be valid",
            )

        self._enabled = True
        self._trace_dir = Path(trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.dp_size = dp_size
        self.max_samples = max_samples
        self.host_id = host_id
        self.pid = pid
        self._samples: list[RuntimeObservation] = []
        self._dropped_samples = 0
        self._trace_error_count = 0
        self._last_event_index: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        rank: int,
        dp_size: int,
    ) -> "ADMRuntimeObserver | None":
        trace_dir = envs.VLLM_ASCEND_ADM_TRACE_DIR
        if not trace_dir:
            return None
        return cls(
            trace_dir=trace_dir,
            rank=rank,
            dp_size=dp_size,
            max_samples=envs.VLLM_ASCEND_ADM_TRACE_MAX_SAMPLES,
            host_id=socket.gethostname(),
            pid=os.getpid(),
        )

    @classmethod
    def disabled(cls) -> "ADMRuntimeObserver":
        observer = cls.__new__(cls)
        observer._enabled = False
        observer._trace_dir = None
        observer.rank = -1
        observer.dp_size = 0
        observer.max_samples = 0
        observer.host_id = ""
        observer.pid = 0
        observer._samples = []
        observer._dropped_samples = 0
        observer._trace_error_count = 0
        observer._last_event_index = None
        return observer

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    @property
    def trace_error_count(self) -> int:
        return self._trace_error_count

    def record(self, **fields: Any) -> dict[str, object] | None:
        if not self._enabled:
            return None

        provided = set(fields)
        unknown = provided - _OBSERVATION_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ObservationViolation(
                "unknown_field",
                f"observation contains unsupported fields: {names}",
            )

        missing = _OBSERVATION_FIELDS - provided
        if missing:
            names = ", ".join(sorted(missing))
            raise ObservationViolation(
                "missing_field",
                f"observation is missing required fields: {names}",
            )

        event_index = fields["event_index"]
        if not _is_non_negative_int(event_index):
            raise ObservationViolation(
                "invalid_event",
                "event_index must be a non-negative integer",
            )
        if (
            self._last_event_index is not None
            and event_index <= self._last_event_index
        ):
            raise ObservationViolation(
                "non_monotonic_event",
                "event_index must increase for every observation",
            )

        path = fields["path"]
        if path not in PATHS:
            raise ObservationViolation(
                "invalid_path",
                "path must be dp1, skip, cpu or npu",
            )

        snapshot_scope = fields["snapshot_scope"]
        if snapshot_scope not in SCOPES:
            raise ObservationViolation(
                "invalid_scope",
                "snapshot_scope must be local or global",
            )

        global_path = path in {"cpu", "npu"}
        expected_scope = "global" if global_path else "local"
        if snapshot_scope != expected_scope:
            raise ObservationViolation(
                "invalid_scope",
                f"{path} observations require {expected_scope} scope",
            )

        expected_size = self.dp_size if global_path else 1
        num_tokens = _validate_values(
            fields["num_tokens"],
            field="num_tokens",
            expected_size=expected_size,
        )
        cudagraph_mode = _validate_values(
            fields["cudagraph_mode"],
            field="cudagraph_mode",
            expected_size=expected_size,
        )

        pack_ns = fields["pack_ns"]
        total_ns = fields["total_ns"]
        if (
            not _is_non_negative_int(pack_ns)
            or not _is_non_negative_int(total_ns)
        ):
            raise ObservationViolation(
                "invalid_timing",
                "pack_ns and total_ns must be non-negative integers",
            )

        collective_enter_ns = _validate_optional_timing(
            fields["collective_enter_ns"],
            "collective_enter_ns",
        )
        collective_ns = _validate_optional_timing(
            fields["collective_ns"],
            "collective_ns",
        )
        copy_to_host_ns = _validate_optional_timing(
            fields["copy_to_host_ns"],
            "copy_to_host_ns",
        )

        if global_path:
            if collective_enter_ns is None or collective_ns is None:
                raise ObservationViolation(
                    "invalid_path_timing",
                    f"{path} requires collective timings",
                )
        elif (
            collective_enter_ns is not None
            or collective_ns is not None
            or copy_to_host_ns is not None
        ):
            raise ObservationViolation(
                "invalid_path_timing",
                f"{path} must not report collective or copy timings",
            )

        if path == "npu" and copy_to_host_ns is None:
            raise ObservationViolation(
                "invalid_path_timing",
                "npu requires copy_to_host_ns",
            )
        if path != "npu" and copy_to_host_ns is not None:
            raise ObservationViolation(
                "invalid_path_timing",
                f"{path} must not report copy_to_host_ns",
            )

        observation = RuntimeObservation(
            host_id=self.host_id,
            pid=self.pid,
            rank=self.rank,
            dp_size=self.dp_size,
            event_index=event_index,
            path=path,
            snapshot_scope=snapshot_scope,
            num_tokens=num_tokens,
            cudagraph_mode=cudagraph_mode,
            collective_enter_ns=collective_enter_ns,
            pack_ns=pack_ns,
            collective_ns=collective_ns,
            copy_to_host_ns=copy_to_host_ns,
            total_ns=total_ns,
        )
        self._last_event_index = event_index

        if len(self._samples) >= self.max_samples:
            self._dropped_samples += 1
            return None

        self._samples.append(observation)
        return observation.to_record()

    def note_error(self) -> None:
        if self._enabled:
            self._trace_error_count += 1

    def flush(self, reason: str) -> bool:
        if not self._enabled:
            return False
        if not isinstance(reason, str) or not reason:
            raise ObservationViolation(
                "invalid_flush_reason",
                "flush reason must be a non-empty string",
            )

        records = [sample.to_record() for sample in self._samples]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "receipt",
                "rank": self.rank,
                "pid": self.pid,
                "sample_count": self.sample_count,
                "dropped_samples": self.dropped_samples,
                "trace_error_count": self.trace_error_count,
                "flush_reason": reason,
            }
        )
        payload = "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )

        assert self._trace_dir is not None
        output = self._trace_dir / (
            f"rank-{self.rank}-pid-{self.pid}.jsonl"
        )
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            payload,
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
        return True
