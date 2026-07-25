# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

try:
    from vllm.v1.worker.pp_opt_profile import (
        mark_t2,
        mark_t3,
        profile_model_runner_execute,
        profile_worker_execute,
        profile_worker_sample_tokens,
        record_active,
        set_microbatch_stats,
    )
except ModuleNotFoundError as exc:
    if exc.name != "vllm.v1.worker.pp_opt_profile":
        raise

    def _identity(func: Callable[_P, _R]) -> Callable[_P, _R]:
        return func

    profile_model_runner_execute = _identity
    profile_worker_execute = _identity
    profile_worker_sample_tokens = _identity

    def mark_t2() -> None:
        pass

    def mark_t3() -> None:
        pass

    def record_active() -> bool:
        return False

    def set_microbatch_stats(*args: object, **kwargs: object) -> None:
        pass
