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
"""Persistent worker threads for dual-stream split replay.

Per decode step, ``run_inplace_parallel`` / ``run_dual_pad`` used to create and
join one fresh ``threading.Thread`` per split (2 starts + 2 joins every step).
Thread creation/join shows up as measurable host time in profiler traces of
small-batch steady-state decode, where the whole step budget is only tens of
microseconds of device work.

This module keeps two long-lived worker threads alive for the lifetime of the
engine process.  Each step dispatches one zero-arg closure per split to its
dedicated worker and waits for completion; semantics are identical to the
previous per-step threads (same calling thread per stream slot, same
error funnel ordered by split index).
"""

import threading
from collections.abc import Callable

from vllm.logger import logger


class _WorkerSlot:
    """One persistent thread consuming zero-arg closures from an inbox."""

    def __init__(self, name: str):
        self._inbox: list[Callable[[], None]] = []
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], None]) -> None:
        """Queue zero-arg ``fn``; raises RuntimeError if the worker died."""
        with self._cv:
            if self._stop or not self._thread.is_alive():
                raise RuntimeError("split replay worker is not alive")
            self._inbox.append(fn)
            self._cv.notify()

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _loop(self) -> None:  # pragma: no cover - thread body
        while True:
            with self._cv:
                while not self._inbox and not self._stop:
                    self._cv.wait()
                if not self._inbox:
                    return  # stopped
                fn = self._inbox.pop(0)
            # Execute outside the condition lock so submit() during execution
            # only queues behind the next round.
            fn()


class SplitReplayWorkerPool:
    """Fixed-size pool of persistent split-replay workers.

    ``dispatch`` runs one callable per split in parallel on distinct
    persistent threads and blocks until all have finished.  Exceptions are
    re-raised on the caller thread ordered by split index, matching the
    previous per-step-thread behaviour.
    """

    def __init__(self, num_workers: int = 2):
        self._slots = [
            _WorkerSlot(f"split-replay-{i}") for i in range(num_workers)
        ]
        self._done_cv = threading.Condition()
        logger.debug("SplitReplayWorkerPool started with %d workers",
                     num_workers)

    def dispatch(self, fns: list[Callable[[], None]]) -> None:
        """Run ``fns[i]`` on worker ``i`` concurrently; block until all done."""
        n = len(fns)
        if n == 0:
            return
        if n > len(self._slots):
            raise ValueError(
                f"requested {n} splits but pool has {len(self._slots)} workers")

        remaining = n
        errors: dict[int, BaseException] = {}
        cv = self._done_cv

        def _run_one(idx: int) -> None:
            nonlocal remaining
            try:
                fns[idx]()
            except BaseException as exc:  # noqa: BLE001 - funneled to caller
                with cv:
                    errors[idx] = exc
            finally:
                with cv:
                    remaining -= 1
                    if remaining == 0:
                        cv.notify_all()

        for i in range(n):
            # Bind i via default arg: submitted closures must be zero-arg.
            self._slots[i].submit(lambda i=i: _run_one(i))

        with cv:
            while remaining > 0:
                cv.wait()

        if errors:
            first = min(errors)
            raise RuntimeError(
                "split replay worker failed at "
                f"slice_idx={first}: {errors[first]}") from errors[first]

    def shutdown(self) -> None:
        for slot in self._slots:
            slot.shutdown()
