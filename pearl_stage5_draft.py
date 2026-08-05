# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V3
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V2
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V1
# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_DRAFT_V1
# PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1
# PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REBASE_V3
# PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REBASE_V2
# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_DRAFT_V1
# PEARL_STAGE5_PIPELINE_V2
# PEARL_STAGE5_PIPELINE_V1
# PEARL_STAGE5_BATCH_GT1_V3
# PEARL_STAGE5_BATCH_GT1_V2
# PEARL_STAGE5_BATCH_GT1_V1
#!/usr/bin/env python3
"""Persistent one-request Draft engine for the nano-PEARL Stage-5 test.

This deliberately uses vLLM V1's in-process ``LLMEngine``/``EngineCore``.
The engine is kept in the Draft worker, while the Target worker asks for
``gamma`` token IDs through a Unix socket.  The Target's prefix is the source
of truth: after every verification, ``sync_prefix`` trims or extends the
Draft request before the next decode step.

This is an integration bridge, not a public vLLM API.  It is intentionally
limited to one request, one sequence, greedy sampling, and eager execution.
Those constraints make KV rollback observable before adding batching or
overlap.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Any


class PersistentDraftEngine:
    """Keep one Draft request alive and generate tokens one step at a time."""

    def __init__(
        self, model: str, max_model_len: int, max_num_seqs: int = 1
    ) -> None:
        # The Draft engine must stay in this process.  The Target may still
        # use vLLM's normal worker process topology.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        from vllm import EngineArgs, SamplingParams
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.llm_engine import LLMEngine

        self._lock = threading.RLock()
        self.max_model_len = int(max_model_len)
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        self.max_num_seqs = int(max_num_seqs)
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")

        engine_args = EngineArgs(
            model=model,
            runner="generate",
            tensor_parallel_size=1,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=(
                self.max_model_len * self.max_num_seqs
            ),
            # PEARL_STAGE5_DRAFT_MEMORY_RESERVE_REMOVE_V1
            enforce_eager=True,
# PEARL_STAGE5_DISABLE_ASYNC_SCHEDULING_DRAFT_V2
            async_scheduling=False,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.engine = LLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
            enable_multiprocessing=False,
        )
        self.core_client = self.engine.engine_core
        if not hasattr(self.core_client, "engine_core"):
            raise RuntimeError(
                "PersistentDraftEngine requires an in-process vLLM V1 "
                "EngineCore; check VLLM_ENABLE_V1_MULTIPROCESSING"
            )
        self.core = self.core_client.engine_core

        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_model_len,
            ignore_eos=True,
        )
        self._states: dict[str, dict[str, Any]] = {}
        self._active_key: str | None = None
        self._pending_tokens: dict[str, deque[int]] = {}
        self._pipeline_candidates: dict[str, dict[str, Any]] = {}
        self._pipeline_thread: threading.Thread | None = None
        self._pipeline_error: BaseException | None = None
        self._request_counter = 0

    def _current_state(self) -> dict[str, Any]:
        if self._active_key is None:
            raise RuntimeError("no active external Draft request")
        return self._states.setdefault(
            self._active_key,
            {
                "request_id": None,
                "prompt_token_ids": None,
                "committed_token_ids": [],
            },
        )

    def _activate_request(self, external_request_id: str) -> None:
        external_request_id = str(external_request_id)
        if not external_request_id:
            raise ValueError("external Draft request ID must not be empty")
        self._active_key = external_request_id
        self._states.setdefault(
            external_request_id,
            {
                "request_id": None,
                "prompt_token_ids": None,
                "committed_token_ids": [],
            },
        )

    @property
    def request_id(self) -> str | None:
        return self._current_state()["request_id"]

    @request_id.setter
    def request_id(self, value: str | None) -> None:
        self._current_state()["request_id"] = value

    @property
    def prompt_token_ids(self) -> list[int] | None:
        return self._current_state()["prompt_token_ids"]

    @prompt_token_ids.setter
    def prompt_token_ids(self, value: list[int] | None) -> None:
        self._current_state()["prompt_token_ids"] = value

    @property
    def committed_token_ids(self) -> list[int]:
        return self._current_state()["committed_token_ids"]

    @committed_token_ids.setter
    def committed_token_ids(self, value: list[int]) -> None:
        self._current_state()["committed_token_ids"] = value

    def _request(self) -> Any:
        if self.request_id is None:
            raise RuntimeError("Draft request has not been created")
        request = self.core.scheduler.requests.get(self.request_id)
        if request is None:
            raise RuntimeError(
                f"Draft request {self.request_id!r} is no longer in the scheduler"
            )
        return request

    # PEARL_STAGE5_RUNTIME_STATE_TRACE_V2
    def _trace_state(
        self,
        label: str,
        request: Any | None = None,
    ) -> None:
        """Print compact scheduler/runner state when explicitly enabled."""

        if os.environ.get("PEARL_DRAFT_STATE_TRACE") != "1":
            return

        def _safe_len(value: Any) -> int | str:
            if value is None:
                return 0
            try:
                return len(value)
            except Exception:
                return "?"

        def _short(value: Any, limit: int = 16) -> Any:
            if value is None:
                return None
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
            except Exception as exc:
                return f"<{type(value).__name__}: {type(exc).__name__}: {exc}>"
            try:
                if isinstance(value, (list, tuple)):
                    if len(value) > limit:
                        return list(value[:limit]) + [
                            f"...(+{len(value) - limit})"
                        ]
                    return list(value)
                return value
            except Exception as exc:
                return f"<{type(value).__name__}: {type(exc).__name__}: {exc}>"

        def _get(obj: Any, name: str, default: Any = None) -> Any:
            if obj is None:
                return default
            try:
                return getattr(obj, name, default)
            except Exception as exc:
                return f"<{name}: {type(exc).__name__}: {exc}>"

        def _ids(value: Any, limit: int = 12) -> Any:
            if value is None:
                return []
            if isinstance(value, dict):
                value = value.keys()
            try:
                items = list(value)
            except Exception:
                return _short(value, limit)
            result = []
            for item in items[:limit]:
                result.append(_get(item, "request_id", item))
            if len(items) > limit:
                result.append(f"...(+{len(items) - limit})")
            return result

        def _block_summary(value: Any) -> Any:
            if value is None:
                return None
            try:
                groups = list(value)
            except Exception:
                return _short(value)
            lengths = []
            heads = []
            for group in groups[:8]:
                try:
                    block_ids = list(group)
                    lengths.append(len(block_ids))
                    heads.append(block_ids[:8])
                except Exception:
                    lengths.append("?")
                    heads.append(_short(group, 8))
            return {
                "groups": len(groups),
                "lengths": lengths,
                "heads": heads,
            }

        def _batch_value(batch: Any, field: str, index: Any) -> Any:
            if batch is None or index is None:
                return None
            try:
                values = getattr(batch, field)
                return _short(values[index])
            except Exception as exc:
                return f"<{field}: {type(exc).__name__}: {exc}>"

        if request is None:
            try:
                request = self._request()
            except Exception:
                request = None

        core = _get(self, "core")
        scheduler = _get(core, "scheduler")
        scheduler_requests = _get(scheduler, "requests", {})
        request_id = _get(self, "request_id")

        executor = _get(core, "model_executor")
        driver_worker = _get(executor, "driver_worker")
        runner = _get(driver_worker, "model_runner")
        if runner is None:
            workers = _get(executor, "workers", [])
            try:
                if workers:
                    runner = _get(workers[0], "model_runner")
            except Exception:
                pass

        runner_requests = _get(runner, "requests", {})
        req_state = None
        try:
            req_state = runner_requests.get(request_id)
        except Exception:
            pass

        input_batch = _get(runner, "input_batch")
        req_index = None
        try:
            req_index = _get(input_batch, "req_id_to_index", {}).get(request_id)
        except Exception:
            pass

        request_all_ids = _get(request, "_all_token_ids", [])
        request_output_ids = _get(request, "_output_token_ids", [])
        req_state_output_ids = _get(req_state, "output_token_ids", [])
        payload = {
            "label": label,
            "request_id": request_id,
            "committed_len": _safe_len(_get(self, "committed_token_ids", [])),
            "committed_head": _short(
                _get(self, "committed_token_ids", []), 12
            ),
            "request": {
                "status": _short(_get(request, "status")),
                "num_prompt_tokens": _short(
                    _get(request, "num_prompt_tokens")
                ),
                "num_tokens": _short(_get(request, "num_tokens")),
                "num_computed_tokens": _short(
                    _get(request, "num_computed_tokens")
                ),
                "all_len": _safe_len(request_all_ids),
                "all_head": _short(request_all_ids, 12),
                "prompt_len": _safe_len(_get(request, "prompt_token_ids", [])),
                "output_len": _safe_len(request_output_ids),
                "output_head": _short(request_output_ids, 12),
                "spec_len": _safe_len(_get(request, "spec_token_ids", [])),
                "num_output_placeholders": _short(
                    _get(request, "num_output_placeholders")
                ),
                "skip_reading_prefix_cache": _short(
                    _get(request, "skip_reading_prefix_cache")
                ),
                "block_hashes_len": _safe_len(
                    _get(request, "block_hashes", [])
                ),
            },
            "req_state": {
                "num_prompt_tokens": _short(
                    _get(req_state, "num_prompt_tokens")
                ),
                "num_tokens": _short(_get(req_state, "num_tokens")),
                "num_computed_tokens": _short(
                    _get(req_state, "num_computed_tokens")
                ),
                "prompt_len": _safe_len(
                    _get(req_state, "prompt_token_ids", [])
                ),
                "output_len": _safe_len(req_state_output_ids),
                "output_head": _short(req_state_output_ids, 12),
                "block_ids": _block_summary(_get(req_state, "block_ids")),
            },
            "input_batch": {
                "req_index": _short(req_index),
                "num_prompt_tokens": _batch_value(
                    input_batch, "num_prompt_tokens", req_index
                ),
                "num_tokens_no_spec": _batch_value(
                    input_batch, "num_tokens_no_spec", req_index
                ),
                "num_tokens": _batch_value(
                    input_batch, "num_tokens", req_index
                ),
                "num_computed_tokens_cpu": _batch_value(
                    input_batch, "num_computed_tokens_cpu", req_index
                ),
                "token_ids_head": _batch_value(
                    input_batch, "token_ids_cpu", req_index
                ),
                "is_token_ids_head": _batch_value(
                    input_batch, "is_token_ids", req_index
                ),
                "spec_token_ids": _batch_value(
                    input_batch, "spec_token_ids", req_index
                ),
            },
            "scheduler": {
                "request_ids": _ids(scheduler_requests),
                "running": _ids(_get(scheduler, "running")),
                "waiting": _ids(_get(scheduler, "waiting")),
                "prev_step_scheduled_req_ids": _short(
                    _get(scheduler, "prev_step_scheduled_req_ids")
                ),
                "finished_req_ids": _short(
                    _get(scheduler, "finished_req_ids")
                ),
            },
        }
        print(
            "[PEARL_STAGE5_RUNTIME_STATE_TRACE_V2] "
            + repr(payload),
            flush=True,
        )

    # PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1
    def _trace_full_block_state(
        self,
        label: str,
        request: Any | None = None,
    ) -> None:
        """Print compact request/runner/KV state when explicitly enabled."""

        if os.environ.get("PEARL_STAGE5_FULL_BLOCK_TRACE", "0") != "1":
            return

        def _get(obj: Any, name: str, default: Any = None) -> Any:
            if obj is None:
                return default
            try:
                return getattr(obj, name, default)
            except Exception as exc:
                return f"<{name}:{type(exc).__name__}:{exc}>"

        def _len(value: Any) -> int | str:
            if value is None:
                return 0
            try:
                return len(value)
            except Exception:
                return "?"

        def _short(value: Any, limit: int = 12) -> Any:
            if value is None:
                return None
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
            except Exception as exc:
                return f"<{type(value).__name__}:{type(exc).__name__}:{exc}>"
            if isinstance(value, (list, tuple)):
                value = list(value)
                if len(value) > limit:
                    return value[:limit] + [f"...(+{len(value) - limit})"]
                return value
            return value

        def _request_ids(value: Any) -> Any:
            if value is None:
                return []
            try:
                items = list(value.keys()) if isinstance(value, dict) else list(value)
            except Exception:
                return "?"
            result = []
            for item in items[:8]:
                result.append(_get(item, "request_id", item))
            if len(items) > 8:
                result.append(f"...(+{len(items) - 8})")
            return result

        def _blocks(value: Any) -> Any:
            if value is None:
                return None
            try:
                value = getattr(value, "blocks", value)
            except Exception:
                pass
            try:
                groups = list(value)
            except Exception:
                return _short(value)
            summary = []
            for group in groups[:8]:
                try:
                    block_ids = list(group)
                    summary.append({
                        "len": len(block_ids),
                        "head": block_ids[:8],
                    })
                except Exception:
                    summary.append(_short(group, 8))
            if len(groups) > 8:
                summary.append({"groups_more": len(groups) - 8})
            return summary

        def _batch_value(batch: Any, field: str, index: Any) -> Any:
            if batch is None or index is None:
                return None
            try:
                return _short(getattr(batch, field)[index])
            except Exception as exc:
                return f"<{field}:{type(exc).__name__}:{exc}>"

        if request is None:
            try:
                request = self._request()
            except Exception:
                request = None

        request_id = _get(self, "request_id")
        core = _get(self, "core")
        scheduler = _get(core, "scheduler")
        executor = _get(core, "model_executor")
        runner = _get(_get(executor, "driver_worker"), "model_runner")
        if runner is None:
            workers = _get(executor, "workers", [])
            try:
                if workers:
                    runner = _get(workers[0], "model_runner")
            except Exception:
                pass

        runner_requests = _get(runner, "requests", {})
        try:
            req_state = runner_requests.get(request_id)
        except Exception:
            req_state = None
        input_batch = _get(runner, "input_batch")
        try:
            req_index = _get(input_batch, "req_id_to_index", {}).get(request_id)
        except Exception:
            req_index = None

        kv_cache_manager = _get(scheduler, "kv_cache_manager")
        try:
            owned_blocks = kv_cache_manager.get_blocks(request_id)
        except Exception:
            owned_blocks = None

        request_all = _get(request, "_all_token_ids", [])
        request_output = _get(request, "_output_token_ids", [])
        req_state_output = _get(req_state, "output_token_ids", [])
        payload = {
            "label": label,
            "request_id": request_id,
            "request": {
                "status": _short(_get(request, "status")),
                "num_computed_tokens": _short(
                    _get(request, "num_computed_tokens")
                ),
                "num_tokens": _short(_get(request, "num_tokens")),
                "num_prompt_tokens": _short(
                    _get(request, "num_prompt_tokens")
                ),
                "all_len": _len(request_all),
                "output_len": _len(request_output),
            },
            "runner": {
                "num_computed_tokens": _short(
                    _get(req_state, "num_computed_tokens")
                ),
                "output_len": _len(req_state_output),
                "block_ids": _blocks(_get(req_state, "block_ids")),
            },
            "input_batch": {
                "req_index": _short(req_index),
                "num_computed_tokens_cpu": _batch_value(
                    input_batch, "num_computed_tokens_cpu", req_index
                ),
                "num_tokens_no_spec": _batch_value(
                    input_batch, "num_tokens_no_spec", req_index
                ),
                "num_tokens": _batch_value(input_batch, "num_tokens", req_index),
                "token_ids_head": _batch_value(
                    input_batch, "token_ids_cpu", req_index
                ),
            },
            "kv_manager_blocks": _blocks(owned_blocks),
            "scheduler": {
                "running": _request_ids(_get(scheduler, "running")),
                "waiting": _request_ids(_get(scheduler, "waiting")),
            },
        }
        print("[PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1] " + repr(payload), flush=True)

    def _add_request(self, prefix_token_ids: list[int]) -> None:
        from vllm.inputs import tokens_input

        if not prefix_token_ids:
            raise ValueError("Draft prefix must not be empty")

        self._request_counter += 1
        request_id = f"pearl-draft-{id(self)}-{self._request_counter}"
        core_request = self.engine.input_processor.process_inputs(
            request_id=request_id,
            prompt=tokens_input(prefix_token_ids),
            params=self.sampling_params,
            supported_tasks=self.engine.get_supported_tasks(),
        )
        self.core_client.add_request(core_request)

        # EngineCore may randomize request IDs unless explicitly disabled.  Use
        # the actual scheduler key so this also works on forks with that
        # default enabled.
        request_keys = list(self.core.scheduler.requests)
        matching = [key for key in request_keys if key.startswith(request_id)]
        if matching:
            self.request_id = matching[-1]
        elif request_id in self.core.scheduler.requests:
            self.request_id = request_id
        else:
            raise RuntimeError(
                "EngineCore accepted the Draft request but it is not visible "
                f"in the scheduler: {request_keys}"
            )

        self.prompt_token_ids = list(prefix_token_ids)
        self.committed_token_ids = list(prefix_token_ids)

    def _replace_tokens(self, prefix_token_ids: list[int]) -> None:
        """Rollback/extend the scheduler-side request to Target's prefix.

        vLLM's scheduler interprets ``num_computed_tokens`` as the number of
        tokens whose KV has been consumed.  The final committed token is left
        uncomputed so the next Draft step consumes it and samples the next
        token.  This is the same one-token lag as ordinary V1 decoding.
        """

        request = self._request()
        assert self.prompt_token_ids is not None
        self._trace_state("replace.before", request=request)
        if len(prefix_token_ids) < len(self.prompt_token_ids):
            raise ValueError("Target prefix is shorter than Draft prompt")
        if prefix_token_ids[: len(self.prompt_token_ids)] != self.prompt_token_ids:
            raise ValueError(
                "Target changed the prompt while the Draft request was alive"
            )

        del request._all_token_ids[:]
        request._all_token_ids.extend(prefix_token_ids)
        del request._output_token_ids[:]
        # PEARL_STAGE5_REQUEST_REBASE_V1
        # Rebase the live request to Target's current prefix.
        request.prompt_token_ids = list(prefix_token_ids)
        request.num_prompt_tokens = len(prefix_token_ids)
        request.spec_token_ids = []
        request.num_output_placeholders = 0
        # PEARL_STAGE5_KV_RECOMPUTE_V1
        # Diagnostic mode: rebuild the current prefix from token 0.
        request.num_computed_tokens = 0
        request.is_prefill_chunk = False

        # Prefix caching is disabled in the Stage-5 command.  Still refresh
        # hashes if a fork has a block hasher attached to Request.
        request.block_hashes.clear()
        request.update_block_hashes()

        # EngineCore and the V1 model runner each keep a small request state.
        # The scheduler-side edit above is enough for the next schedule, but
        # the model runner must also see the authoritative token list when a
        # Target bonus/replacement token is different from the last Draft
        # token.  Keep this best-effort and fail loudly if a fork exposes an
        # incompatible runner state instead of silently producing bad KV.
        self._sync_model_runner_state(prefix_token_ids, request)
        self._trace_state("replace.after", request=request)

        self.committed_token_ids = list(prefix_token_ids)

    def _sync_model_runner_state(self, prefix_token_ids: list[int], request: Any) -> None:
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            # Some executor variants expose the runner through a worker list.
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner for prefix sync"
            )

        req_state = getattr(runner, "requests", {}).get(self.request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for "
                f"request {self.request_id!r}"
            )

        # PEARL_STAGE5_REQUEST_REBASE_V1
        # Keep the model-runner prompt boundary identical to a fresh
        # request created from this Target prefix.
        req_state.prompt_token_ids = list(prefix_token_ids)
        req_state.num_prompt_tokens = len(prefix_token_ids)
        del req_state.output_token_ids[:]
        req_state.num_computed_tokens = request.num_computed_tokens

        req_index = input_batch.req_id_to_index.get(self.request_id)
        if req_index is None:
            raise RuntimeError(
                "Draft model runner input batch has no index for "
                f"request {self.request_id!r}"
            )

        # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_FIX_V1
        # A correctness diagnostic may explicitly request fresh KV blocks even
        # while the persistent-requeue path remains enabled.
        persistent_requeue = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE", "0"
        ) == "1"
        force_fresh_block = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK", "0"
        ) == "1"
        if not persistent_requeue or force_fresh_block:
            # PEARL_STAGE5_PERSISTENT_REQUEUE_KV_BYPASS_V2
            # PEARL_STAGE5_KV_RECOMPUTE_V1
            kv_cache_manager = getattr(
                self.core.scheduler, "kv_cache_manager", None
            )
            if kv_cache_manager is None:
                raise RuntimeError(
                    "Cannot locate the Draft scheduler KV cache manager"
                )
            # Drop all blocks owned by this live request.  The request itself
            # remains in the scheduler and will receive fresh blocks next step.
            kv_cache_manager.free(request)
            # Do not let prefix caching immediately reuse the blocks just freed;
            # this run is a correctness diagnostic for persistent KV state.
            request.skip_reading_prefix_cache = True
            block_ids = getattr(req_state, "block_ids", None)
            if block_ids is not None:
                req_state.block_ids = tuple([] for _ in block_ids)
            block_table = getattr(input_batch, "block_table", None)
            if block_table is None or not hasattr(block_table, "clear_row"):
                raise RuntimeError(
                    "Draft input batch has no block table clear_row API"
                )
            block_table.clear_row(req_index)
        if hasattr(input_batch, "num_prompt_tokens"):
            input_batch.num_prompt_tokens[req_index] = len(prefix_token_ids)
        input_batch.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        input_batch.num_tokens_no_spec[req_index] = len(prefix_token_ids)
        if hasattr(input_batch, "num_tokens"):
            input_batch.num_tokens[req_index] = len(prefix_token_ids)
        input_batch.token_ids_cpu[req_index, : len(prefix_token_ids)] = prefix_token_ids
        # PEARL_STAGE5_PERSISTENT_KV_FIX_V1
        is_token_ids = getattr(input_batch, "is_token_ids", None)
        if is_token_ids is not None:
            valid_mask = is_token_ids[req_index]
            if hasattr(valid_mask, "fill_"):
                valid_mask.fill_(False)
            else:
                valid_mask[...] = False
            valid_mask[: len(prefix_token_ids)] = True
        input_batch.spec_token_ids[req_index].clear()

    def _remove_request_from_model_runner_batch(self) -> None:
        """Remove the batch row but keep the cached request state."""
        if self.request_id is None:
            raise RuntimeError(
                "Cannot remove a missing Draft request from input batch"
            )

        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner while "
                "removing the persistent batch row"
            )

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None:
            raise RuntimeError(
                "Draft model runner has no input batch while removing "
                f"request {self.request_id!r}"
            )
        if self.request_id not in input_batch.req_id_to_index:
            raise RuntimeError(
                "Draft request is not present in the input batch: "
                f"{self.request_id!r}"
            )
        input_batch.remove_request(self.request_id)

    def _release_request_kv_for_fresh_recompute(self, request: Any) -> None:
        """Release old KV and clear runner-side block ownership."""
        scheduler = self.core.scheduler
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            raise RuntimeError(
                "Cannot locate the Draft scheduler KV cache manager"
            )
        kv_cache_manager.free(request)

        if self.request_id is None:
            raise RuntimeError(
                "Cannot clear KV state for a missing Draft request"
            )
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner while "
                "clearing fresh-recompute KV state"
            )

        req_state = getattr(runner, "requests", {}).get(self.request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for fresh "
                f"recompute request {self.request_id!r}"
            )

        block_ids = getattr(req_state, "block_ids", None)
        if block_ids is not None:
            req_state.block_ids = tuple([] for _ in block_ids)
        req_index = input_batch.req_id_to_index.get(self.request_id)
        block_table = getattr(input_batch, "block_table", None)
        if req_index is not None and block_table is not None:
            if not hasattr(block_table, "clear_row"):
                raise RuntimeError(
                    "Draft input batch block table has no clear_row API"
                )
            block_table.clear_row(req_index)

    def _drop_request_from_model_runner_cache(self) -> None:
        """Drop only the cached runner request state for this probe."""
        if self.request_id is None:
            raise RuntimeError(
                "Cannot drop cached state for a missing Draft request"
            )

        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner while "
                "dropping cached request state"
            )

        requests = getattr(runner, "requests", None)
        if requests is None or not hasattr(requests, "pop"):
            raise RuntimeError(
                "Draft model runner has no mutable request-state cache"
            )
        requests.pop(self.request_id, None)
        if os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0"
        ) == "1":
            print(
                "[PEARL_STAGE5_PERSISTENT_REQUEUE_CACHED_STATE_PROBE_V6] "
                f"dropped request-state cache for {self.request_id!r}",
                flush=True,
            )

    # PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1
    def _drop_partial_request_blocks(
        self,
        request: Any,
        reusable_tokens: int,
    ) -> None:
        """Keep aligned full blocks and return only stale tail blocks."""

        scheduler = self.core.scheduler
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        if not managers:
            raise RuntimeError(
                "Cannot locate single-type KV managers for selective tail release"
            )

        request_id = request.request_id
        kept_block_ids: list[list[int]] = []
        for manager_index, manager in enumerate(managers):
            req_to_blocks = getattr(manager, "req_to_blocks", None)
            cached_counts = getattr(manager, "num_cached_block", None)
            block_pool = getattr(manager, "block_pool", None)
            if req_to_blocks is None or cached_counts is None or block_pool is None:
                raise RuntimeError(
                    "KV manager %d lacks req_to_blocks, num_cached_block, or "
                    "block_pool" % manager_index
                )

            req_blocks = req_to_blocks.get(request_id)
            if req_blocks is None:
                raise RuntimeError(
                    "KV manager %d has no block list for request %r"
                    % (manager_index, request_id)
                )

            try:
                block_size = int(getattr(manager, "block_size"))
            except (TypeError, ValueError, AttributeError) as exc:
                raise RuntimeError(
                    "KV manager %d has no valid block_size" % manager_index
                ) from exc
            if block_size <= 0:
                raise RuntimeError(
                    "KV manager %d has invalid block_size=%r"
                    % (manager_index, block_size)
                )

            keep_count = min(len(req_blocks), reusable_tokens // block_size)
            stale_blocks = list(req_blocks[keep_count:])
            if stale_blocks:
                del req_blocks[keep_count:]

                free_blocks = getattr(block_pool, "free_blocks", None)
                if not callable(free_blocks):
                    raise RuntimeError(
                        "KV manager %d block pool has no free_blocks API"
                        % manager_index
                    )
                # The vLLM block pool expects tail blocks to be freed in
                # reverse allocation order.
                free_blocks(reversed(stale_blocks))

            cached_count = int(cached_counts.get(request_id, 0))
            if cached_count > keep_count:
                cached_counts[request_id] = keep_count

            kept_block_ids.append(
                [int(block.block_id) for block in req_blocks]
            )

        # Keep the in-process model-runner state aligned with the manager's
        # request bookkeeping before the persistent batch row is removed.
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner for tail release"
            )

        req_state = getattr(runner, "requests", {}).get(request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for selective tail release"
            )

        req_state.block_ids = tuple(list(group) for group in kept_block_ids)
        req_index = input_batch.req_id_to_index.get(request_id)
        block_table = getattr(input_batch, "block_table", None)
        if req_index is not None and block_table is not None:
            if not hasattr(block_table, "clear_row") or not hasattr(
                block_table, "append_row"
            ):
                raise RuntimeError(
                    "Draft input batch block table lacks clear_row/append_row"
                )
            block_table.clear_row(req_index)
            block_table.append_row(kept_block_ids, req_index)

        if os.environ.get("PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1] "
                f"request={request_id!r} reusable_tokens={reusable_tokens} "
                f"kept_block_ids={kept_block_ids}",
                flush=True,
            )

    # PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1
    def _nanoparl_kv_ownership_snapshot(self, request_id: str) -> dict:
        """Return non-mutating manager/runner block-ID snapshots."""

        snapshot = {
            "available": False,
            "reason": None,
            "manager_block_ids": [],
            "num_cached_blocks": [],
            "runner_block_ids": [],
        }

        def normalize_ids(values: Any) -> list[int]:
            if values is None:
                return []
            if not isinstance(values, (list, tuple)):
                values = [values]
            result: list[int] = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    result.extend(normalize_ids(value))
                    continue
                block_id = getattr(value, "block_id", value)
                try:
                    result.append(int(block_id))
                except (TypeError, ValueError):
                    result.append(-1)
            return result

        scheduler = self.core.scheduler
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        if managers is None and getattr(kv_cache_manager, "req_to_blocks", None) is not None:
            managers = [kv_cache_manager]
        if not managers:
            snapshot["reason"] = "single_type_managers_unavailable"
        else:
            manager_seen = False
            for manager in managers:
                req_to_blocks = getattr(manager, "req_to_blocks", None)
                cached_counts = getattr(manager, "num_cached_block", None)
                if req_to_blocks is None or cached_counts is None:
                    snapshot["reason"] = "manager_bookkeeping_unavailable"
                    continue
                blocks = req_to_blocks.get(request_id)
                if blocks is None:
                    snapshot["reason"] = "request_block_list_unavailable"
                    continue
                manager_seen = True
                snapshot["manager_block_ids"].append(normalize_ids(blocks))
                try:
                    cached_count = int(cached_counts.get(request_id, 0))
                except (TypeError, ValueError):
                    cached_count = -1
                snapshot["num_cached_blocks"].append(cached_count)
            snapshot["available"] = manager_seen

        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is not None:
            requests = getattr(runner, "requests", None)
            req_state = requests.get(request_id) if requests is not None else None
            if req_state is not None:
                snapshot["runner_block_ids"] = normalize_ids(
                    getattr(req_state, "block_ids", None)
                )

        return snapshot
    def _requeue_request_preserve_kv(
        self, prefix_token_ids: list[int]
    ) -> None:
        """Requeue the live Draft request without freeing its KV blocks."""
        from vllm.v1.request import RequestStatus

        if self.request_id is None:
            raise RuntimeError("Cannot requeue a missing Draft request")

        request = self._request()
        scheduler = self.core.scheduler
        if request.status != RequestStatus.RUNNING:
            raise RuntimeError(
                "Persistent requeue requires a RUNNING request, got "
                f"{request.status!s} for {request.request_id!r}"
            )
        if request not in scheduler.running:
            raise RuntimeError(
                "Draft request is not present in scheduler.running: "
                f"{request.request_id!r}"
            )

        old_prefix = self.committed_token_ids
        common_len = 0
        for old_token, new_token in zip(old_prefix, prefix_token_ids):
            if old_token != new_token:
                break
            common_len += 1

        prompt_len = len(self.prompt_token_ids or [])
        if common_len < prompt_len:
            raise RuntimeError(
                "Target prefix diverged inside the Draft prompt: "
                f"common_len={common_len} prompt_len={prompt_len}"
            )

        # PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1
        # PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_DEFAULT_V1
        # Do not attempt same-Request KV reuse through a partial block.
        # The reset path is the correctness reference established by the
        # fresh-request control experiment.
        # PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_BLOCK_REUSE_V1
        # PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE_V1
        partial_recompute = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE",
            "0",
        ) == "1"
        reusable_tokens = max(0, common_len - 1)
        # PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V1
        # PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V2
        # PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V3
        # Clean default-on expression; explicit env=0 still disables reuse.
        true_partial_reuse = (
            os.environ['PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE']
            if 'PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE' in os.environ
            else "1"
        ) == "1" and reusable_tokens > 0
        partial_recompute_block_size = 0
        if partial_recompute:
            try:
                partial_recompute_block_size = int(
                    getattr(scheduler, "block_size")
                )
            except (AttributeError, TypeError, ValueError):
                partial_recompute_block_size = 0
        partial_recompute_has_full_block = (
            partial_recompute
            and partial_recompute_block_size > 0
            and reusable_tokens >= partial_recompute_block_size
        )

        if os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK", "1"
        ) == "1" and not partial_recompute_has_full_block and not true_partial_reuse:
            kv_cache_manager = getattr(
                self.core.scheduler, "kv_cache_manager", None
            )
            # PEARL_STAGE5_PERSISTENT_REQUEUE_BLOCK_SIZE_LOOKUP_V1
            scheduler = self.core.scheduler
            block_size_candidates = (
                (
                    "kv_cache_manager.block_size",
                    getattr(kv_cache_manager, "block_size", None),
                ),
                (
                    "kv_cache_manager.block_size_tokens",
                    getattr(kv_cache_manager, "block_size_tokens", None),
                ),
                (
                    "kv_cache_manager.block_pool.block_size",
                    getattr(
                        getattr(kv_cache_manager, "block_pool", None),
                        "block_size",
                        None,
                    ),
                ),
                (
                    "kv_cache_manager.cache_config.block_size",
                    getattr(
                        getattr(kv_cache_manager, "cache_config", None),
                        "block_size",
                        None,
                    ),
                ),
                (
                    "scheduler.block_size",
                    getattr(scheduler, "block_size", None),
                ),
                (
                    "scheduler.cache_config.block_size",
                    getattr(
                        getattr(scheduler, "cache_config", None),
                        "block_size",
                        None,
                    ),
                ),
                (
                    "core.vllm_config.cache_config.block_size",
                    getattr(
                        getattr(
                            getattr(self.core, "vllm_config", None),
                            "cache_config",
                            None,
                        ),
                        "block_size",
                        None,
                    ),
                ),
            )
            block_size = None
            block_size_source = "unknown"
            for candidate_name, candidate_value in block_size_candidates:
                try:
                    candidate_int = int(candidate_value)
                except (TypeError, ValueError):
                    continue
                if candidate_int > 0:
                    block_size = candidate_int
                    block_size_source = candidate_name
                    break
            if not isinstance(block_size, int) or block_size <= 0:
                reusable_tokens = max(0, common_len - 1)
                reason = "unknown_block_size"
                can_reuse_full_block = False
            else:
                reusable_tokens = max(0, common_len - 1)
                reason = (
                    "less_than_one_block"
                    if reusable_tokens < block_size
                    else "partial_block"
                )
                can_reuse_full_block = (
                    reusable_tokens >= block_size
                    and reusable_tokens % block_size == 0
                )
            if not can_reuse_full_block:
                if os.environ.get(
                    "PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0"
                ) == "1":
                    print(
                        "[PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1] "
                        f"request={self.request_id!r} "
                        f"common_len={common_len} "
                        f"reusable_tokens={reusable_tokens} "
                        f"block_size={block_size!r} "
                        f"block_size_source={block_size_source} "
                        f"reason={reason} "
                        "action=reset_request",
                        flush=True,
                    )
                self._reset_request(prefix_token_ids)
                return

        # Synchronize the live Request's logical prompt with the complete
        # Target prefix before rebuilding runner-side token state.  A safe
        # abort/re-add naturally gets these fields from tokens_input().
        # PEARL_STAGE5_PERSISTENT_REQUEUE_PROMPT_RESET_PROBE_V7
        self.prompt_token_ids = list(prefix_token_ids)
        request.prompt_token_ids = list(prefix_token_ids)
        request.num_prompt_tokens = len(prefix_token_ids)
        if getattr(request, "prompt_is_token_ids", None) is not None:
            request.prompt_is_token_ids = [True] * len(prefix_token_ids)

        # Synchronize token IDs while the request is still in the persistent
        # model-runner batch. Then keep only the common-prefix KV usable.
        self._trace_full_block_state(
            "full_requeue.before_replace", request=request
        )
        self._replace_tokens(prefix_token_ids)
        if (
            os.environ.get(
                "PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK",
                "0",
            ) == "1"
            or partial_recompute_has_full_block
        ) and not true_partial_reuse:
            self._drop_partial_request_blocks(
                request, max(0, common_len - 1)
            )
        # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5
        # Release the old request-owned KV only for the explicit fresh-
        # block control.  The request object and scheduler lifecycle stay
        # alive, but the next schedule must allocate a new block.
        if os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK", "0"
        ) == "1":
            self._release_request_kv_for_fresh_recompute(request)
            request.num_computed_tokens = 0
        elif os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE", "0"
        ) == "1":
            request.num_computed_tokens = 0
        elif true_partial_reuse:
            request.num_computed_tokens = max(0, common_len - 1)
            if os.environ.get(
                "PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0"
            ) == "1":
                print(
                    "[PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_V1] "
                    f"request={request.request_id!r} "
                    f"common_len={common_len} "
                    f"reusable_tokens={reusable_tokens} "
                    f"num_computed_tokens={request.num_computed_tokens} "
                    "action=retain_partial_tail",
                    flush=True,
                )
        elif partial_recompute_has_full_block:
            request.num_computed_tokens = (
                reusable_tokens // partial_recompute_block_size
            ) * partial_recompute_block_size
            if os.environ.get(
                "PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0"
            ) == "1":
                print(
                    "[PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE_V1] "
                    f"request={request.request_id!r} "
                    f"common_len={common_len} "
                    f"reusable_tokens={reusable_tokens} "
                    f"recompute_from={request.num_computed_tokens} "
                    f"block_size={partial_recompute_block_size}",
                    flush=True,
                )
        else:
            request.num_computed_tokens = max(0, common_len - 1)
        # # PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1
        # The normal path below performs batch-row removal plus
        # RUNNING -> WAITING -> re-add. True nano-PEARL keeps the live row
        # and only moves the authoritative boundary backward. The current
        # vLLM V1 convention leaves the last committed token uncomputed, so
        # valid_len is represented by num_computed_tokens + 1.
        if os.environ.get("PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK", "0") == "1":
            kv_before = self._nanoparl_kv_ownership_snapshot(
                request.request_id
            )
            accepted_len = common_len
            valid_len = max(0, accepted_len - 1)
            request.num_computed_tokens = valid_len
            request.is_prefill_chunk = False
            request.spec_token_ids = []
            request.num_output_placeholders = 0

            # Keep the existing Request, model-runner row, and request-owned
            # block IDs. _sync_model_runner_state updates the CPU scheduling
            # view; the next normal engine step consumes only the suffix
            # beyond valid_len. Do not free blocks or rebuild the row.
            self._sync_model_runner_state(prefix_token_ids, request)
            kv_after = self._nanoparl_kv_ownership_snapshot(
                request.request_id
            )
            if (
                os.environ.get("PEARL_STAGE5_NANOPEARL_KV_TRACE", "0")
                == "1"
                or os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0")
                == "1"
            ):
                print(
                    "[PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1] "
                    + repr({
                        "request": request.request_id,
                        "accepted_len": accepted_len,
                        "valid_len": valid_len,
                        "before": kv_before,
                        "after": kv_after,
                        "same_manager_block_ids": (
                            kv_before["manager_block_ids"]
                            == kv_after["manager_block_ids"]
                        ),
                        "same_runner_block_ids": (
                            kv_before["runner_block_ids"]
                            == kv_after["runner_block_ids"]
                        ),
                    }),
                    flush=True,
                )
            inflight_prefills = getattr(scheduler, "_inflight_prefills", None)
            if inflight_prefills is not None:
                inflight_prefills.discard(request)

            print(
                "[PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1] "
                f"request={request.request_id!r} "
                f"accepted_len={accepted_len} "
                f"valid_len={valid_len} "
                f"num_computed_tokens={request.num_computed_tokens} "
                "action=retain_running_row",
                flush=True,
            )
            return

        self._sync_model_runner_state(prefix_token_ids, request)
        self._trace_full_block_state(
            "full_requeue.after_sync", request=request
        )
        # PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3
        # Remove only the persistent-batch row. Cached request state and
        # request-owned block IDs remain alive for the next add path.
        self._remove_request_from_model_runner_batch()
        # PEARL_STAGE5_PERSISTENT_REQUEUE_CACHED_STATE_PROBE_V6
        # The v5 control clears the old KV/block ownership.  This extra
        # opt-in step also drops the cached request state, so the next
        # WAITING schedule must build a fresh runner state for this ID.
        if os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_RESET_CACHED_STATE", "0"
        ) == "1":
            self._drop_request_from_model_runner_cache()

        # Do not call _preempt_request(), finish_requests(), or
        # kv_cache_manager.free(): all of those release request blocks.
        scheduler.running.remove(request)
        scheduler._inflight_prefills.discard(request)
        request.status = RequestStatus.WAITING
        scheduler.waiting.prepend_request(request)

        if os.environ.get("PEARL_STAGE5_FULL_BLOCK_TRACE", "0") == "1":
            self._pearl_full_block_trace_pending = True
            self._trace_full_block_state(
                "full_requeue.after_enqueue", request=request
            )

        if os.environ.get("PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_PERSISTENT_REQUEUE_V1] "
                "request=%s common_len=%d num_computed_tokens=%d "
                "status=%s running=%d waiting=%d"
                % (
                    request.request_id,
                    common_len,
                    request.num_computed_tokens,
                    request.status.name,
                    len(scheduler.running),
                    len(scheduler.waiting),
                ),
                flush=True,
            )

    def _reset_request(self, prefix_token_ids: list[int]) -> None:
        if self.request_id is not None:
            self.core_client.abort_requests([self.request_id])
        self.request_id = None
        self.prompt_token_ids = None
        self.committed_token_ids = []
        self._add_request(prefix_token_ids)

    def sync_prefix(self, prefix_token_ids: list[int]) -> None:
        prefix_token_ids = [int(token_id) for token_id in prefix_token_ids]
        if not prefix_token_ids:
            raise ValueError("Target prefix must not be empty")
        # PEARL_DRAFT_FORCE_RESET_V1
        if os.environ.get("PEARL_DRAFT_FORCE_RESET", "0") == "1":
            self._reset_request(prefix_token_ids)
            return

        if self.request_id is None:
            self._add_request(prefix_token_ids)
            return

        assert self.prompt_token_ids is not None
        if prefix_token_ids[: len(self.prompt_token_ids)] != self.prompt_token_ids:
            self._reset_request(prefix_token_ids)
            return

        request = self._request()
        if request.is_finished():
            self._reset_request(prefix_token_ids)
            return

        if prefix_token_ids != self.committed_token_ids:
            # Any queued output was generated from the old Target prefix.
            self._pending_tokens.pop(self.request_id, None)
            # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_REQUEST_CONTROL_V1
            # Isolate request-lifecycle state for the fresh-block correctness
            # control. The normal persistent path below remains unchanged.
            if os.environ.get(
                "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK", "0"
            ) == "1":
                self._reset_request(prefix_token_ids)
                return

            # PEARL_STAGE5_PERSISTENT_REQUEUE_V1
            if os.environ.get(
                "PEARL_STAGE5_PERSISTENT_REQUEUE", "0"
            ) == "1":
                self._requeue_request_preserve_kv(prefix_token_ids)
                return
            if os.environ.get(
                "PEARL_DRAFT_PERSISTENT_REUSE", "0"
            ) == "1":
                self._replace_tokens(prefix_token_ids)
                return
            self._reset_request(prefix_token_ids)
            return


    def _step_one(self) -> int:
        if self.request_id is None:
            raise RuntimeError("Draft request has not been created")

        # Do not consume a token buffered while another request was active.
        # The Target prefix may have changed since that token was produced.
        # Driving get_output() here forces the current request through the
        # scheduler after every persistent requeue, so it cannot remain in
        # WAITING state merely because another batch row produced output.
        while True:
            outputs = self.core_client.get_output()
            for output in outputs.outputs:
                output_request_id = str(output.request_id)
                if output_request_id != self.request_id:
                    continue
                new_token_ids = getattr(output, "new_token_ids", None) or []
                if new_token_ids:
                    return int(new_token_ids[-1])
                if output.finished:
                    raise RuntimeError(
                        "Draft request finished before returning a token: "
                        f"{output.finish_reason}"
                    )

            if self.request_id not in self.core.scheduler.requests:
                raise RuntimeError("Draft request disappeared while decoding")

    def _collect_batch_tokens(
        self,
        internal_to_external: dict[str, str],
        target_counts: dict[str, int],
    ) -> dict[str, list[int]]:
        """Advance all active Draft requests and collect their tokens.

        A call to ``get_output`` advances one EngineCore step.  All
        requests that are ready in the Draft scheduler are therefore
        verified from the same batched model forward.  Extra tokens from
        a single output are queued and consumed on later draft rounds.
        """
        collected = {external_id: [] for external_id in target_counts}
        pending = self._pending_tokens
        batch_step = 0

        while any(
            len(collected[external_id]) < count
            for external_id, count in target_counts.items()
        ):
            batch_step += 1
            outputs = self.core_client.get_output()
            if os.environ.get("PEARL_STAGE5_DRAFT_BATCH_TRACE", "0") == "1":
                print(
                    "[PEARL_STAGE5_DRAFT_BATCH_TRACE] "
                    f"step={batch_step} "
                    f"output_request_ids={[str(output.request_id) for output in outputs.outputs]} ",
                    flush=True,
                )
            for output in outputs.outputs:
                internal_id = str(output.request_id)
                external_id = internal_to_external.get(internal_id)
                if external_id is None:
                    # An output for a request outside this RPC is stale
                    # with respect to the current Target prefix.  Do not
                    # leak it into the next batch.
                    continue
                new_token_ids = getattr(output, "new_token_ids", None) or []
                if new_token_ids:
                    queue = pending.setdefault(internal_id, deque())
                    queue.extend(int(token_id) for token_id in new_token_ids)
                if output.finished and not new_token_ids:
                    if len(collected[external_id]) < target_counts[external_id]:
                        raise RuntimeError(
                            "Draft request finished before returning enough "
                            f"tokens: request={external_id!r} "
                            f"reason={output.finish_reason!r}"
                        )

            for internal_id, external_id in internal_to_external.items():
                count = target_counts[external_id]
                queue = pending.get(internal_id)
                while (
                    queue
                    and len(collected[external_id]) < count
                ):
                    collected[external_id].append(queue.popleft())

            for internal_id, external_id in internal_to_external.items():
                if len(collected[external_id]) >= target_counts[external_id]:
                    continue
                if internal_id not in self.core.scheduler.requests:
                    raise RuntimeError(
                        "Draft request disappeared while decoding: "
                        f"request={external_id!r} internal={internal_id!r}"
                    )

        return collected

    def _pipeline_enabled(self) -> bool:
        return os.environ.get("PEARL_STAGE5_PIPELINE", "0") == "1"

    def _pipeline_trace(self, message: str) -> None:
        if os.environ.get("PEARL_STAGE5_PIPELINE_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_PIPELINE_V1] " + message,
                flush=True,
            )

    def _pipeline_lookahead(self, gamma: int) -> int:
        raw = os.environ.get("PEARL_STAGE5_PIPELINE_LOOKAHEAD")
        if raw is None:
            # Need gamma+1 extra tokens to cover the common case where
            # Target accepts all gamma tokens and its bonus token equals
            # the first lookahead token.
            return max(1, int(gamma) + 1)
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "PEARL_STAGE5_PIPELINE_LOOKAHEAD must be an integer"
            ) from exc
        if value < 0:
            raise ValueError(
                "PEARL_STAGE5_PIPELINE_LOOKAHEAD must be non-negative"
            )
        return value

    def _join_pipeline(self) -> None:
        thread = self._pipeline_thread
        if thread is not None:
            # Join outside self._lock: the worker acquires the same lock
            # around EngineCore access.
            thread.join()
            self._pipeline_thread = None
        error = self._pipeline_error
        self._pipeline_error = None
        if error is not None:
            self._pipeline_candidates.clear()
            self._pipeline_trace(
                "prefetch_error=" + repr(error) + " action=fallback"
            )

    def _pipeline_prefetch_worker(
        self,
        internal_to_external: dict[str, str],
        target_counts: dict[str, int],
        base_generated: dict[str, list[int]],
    ) -> None:
        try:
            with self._lock:
                collected = self._collect_batch_tokens(
                    internal_to_external=internal_to_external,
                    target_counts=target_counts,
                )
                for internal_id, external_id in internal_to_external.items():
                    candidate = self._pipeline_candidates.get(external_id)
                    if candidate is None:
                        continue
                    if candidate["generated"] != base_generated[external_id]:
                        # A future code path changed the candidate while
                        # this worker was running.  Never append tokens to
                        # a different prefix.
                        continue
                    candidate["generated"].extend(
                        int(token_id)
                        for token_id in collected.get(external_id, [])
                    )
                    pending = self._pending_tokens.pop(internal_id, None)
                    if pending:
                        candidate["generated"].extend(int(x) for x in pending)
                self._pipeline_trace(
                    "prefetch_done "
                    f"batch={len(internal_to_external)} "
                    f"tokens={sum(len(value) for value in collected.values())}"
                )
        except BaseException as exc:
            self._pipeline_error = exc

    def _start_pipeline_prefetch(
        self,
        internal_to_external: dict[str, str],
        target_counts: dict[str, int],
        base_generated: dict[str, list[int]],
    ) -> None:
        positive_counts = {
            external_id: count
            for external_id, count in target_counts.items()
            if count > 0
        }
        if not positive_counts:
            self._pipeline_thread = None
            return
        filtered_mapping = {
            internal_id: external_id
            for internal_id, external_id in internal_to_external.items()
            if external_id in positive_counts
        }
        filtered_base = {
            external_id: list(base_generated[external_id])
            for external_id in positive_counts
        }
        self._pipeline_trace(
            "prefetch_start "
            f"batch={len(filtered_mapping)} "
            f"lookahead={sum(positive_counts.values())}"
        )
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_prefetch_worker,
            args=(filtered_mapping, positive_counts, filtered_base),
            name="pearl-stage5-draft-prefetch",
            daemon=True,
        )
        self._pipeline_thread.start()

    def _pipeline_candidate_can_serve(
        self,
        external_id: str,
        prefix: list[int],
        gamma: int,
    ) -> bool:
        candidate = self._pipeline_candidates.get(external_id)
        if candidate is None:
            return False
        internal_id = str(candidate["internal_id"])
        state = self._states.get(external_id)
        if state is None or str(state.get("request_id")) != internal_id:
            return False
        generated = candidate["generated"]
        if len(prefix) < len(candidate["base_prefix"]):
            return False
        if generated[: len(prefix)] != prefix:
            return False
        return len(generated) >= len(prefix) + int(gamma)

    def _sync_model_runner_lengths_only(
        self,
        request: Any,
        target_prefix_len: int,
    ) -> None:
        """Advance lengths and Draft-local bookkeeping without RPC token IDs."""
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner for "
                "length-only commit"
            )

        req_state = getattr(runner, "requests", {}).get(self.request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for length-only "
                f"request {self.request_id!r}"
            )
        req_index = input_batch.req_id_to_index.get(self.request_id)
        if req_index is None:
            raise RuntimeError(
                "Draft model runner input batch has no index for "
                f"length-only request {self.request_id!r}"
            )

        # The length-only RPC deliberately carries no token IDs.  When the
        # Target boundary advances into Draft's optimistic look-ahead, adopt
        # those IDs from the already resident Draft row before clearing its
        # speculative placeholders.  This also keeps committed_token_ids in
        # sync across consecutive length-only rounds.
        committed = [int(x) for x in self.committed_token_ids]
        adopted = 0
        if len(committed) < target_prefix_len:
            needed = target_prefix_len - len(committed)
            candidates: list[int] = []

            spec_rows = getattr(input_batch, "spec_token_ids", None)
            if spec_rows is not None:
                try:
                    values = spec_rows[req_index]
                    if hasattr(values, "tolist"):
                        values = values.tolist()
                    if isinstance(values, (list, tuple)):
                        candidates = [int(x) for x in values if int(x) >= 0]
                except (IndexError, TypeError, ValueError):
                    candidates = []

            if len(candidates) < needed:
                token_ids_cpu = getattr(input_batch, "token_ids_cpu", None)
                if token_ids_cpu is not None:
                    try:
                        values = token_ids_cpu[
                            req_index, len(committed):target_prefix_len
                        ]
                        if hasattr(values, "tolist"):
                            values = values.tolist()
                        if isinstance(values, (list, tuple)):
                            cpu_candidates = [
                                int(x) for x in values if int(x) >= 0
                            ]
                            if len(cpu_candidates) >= needed:
                                candidates = cpu_candidates
                    except (IndexError, TypeError, ValueError):
                        pass

            if len(candidates) >= needed:
                committed.extend(candidates[:needed])
                adopted = needed

        if len(committed) < target_prefix_len:
            raise RuntimeError(
                "length-only local Draft state is shorter than the requested "
                f"boundary for {self.request_id!r}: "
                f"have={len(committed)} need={target_prefix_len}"
            )

        self.committed_token_ids = committed[:target_prefix_len]
        self.prompt_token_ids = list(self.committed_token_ids)
        request.prompt_token_ids = list(self.committed_token_ids)
        request.num_prompt_tokens = target_prefix_len
        if getattr(request, "prompt_is_token_ids", None) is not None:
            request.prompt_is_token_ids = [True] * target_prefix_len

        request.num_computed_tokens = target_prefix_len - 1
        req_state.num_computed_tokens = request.num_computed_tokens
        if hasattr(input_batch, "num_computed_tokens_cpu"):
            input_batch.num_computed_tokens_cpu[req_index] = (
                request.num_computed_tokens
            )
        input_batch.num_tokens_no_spec[req_index] = target_prefix_len
        if hasattr(input_batch, "num_tokens"):
            input_batch.num_tokens[req_index] = target_prefix_len
        input_batch.spec_token_ids[req_index].clear()

        if os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_V3] "
                f"request_id={self.request_id!r} "
                f"target_prefix_len={target_prefix_len} "
                f"adopted_local_tokens={adopted} "
                "action=update_lengths_only_keep_kv",
                flush=True,
            )
    def commit_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Commit Target boundaries, with a strict length-only fast path."""
        if not isinstance(updates, list) or not updates:
            raise ValueError("commit_batch requires a non-empty updates list")
        if os.environ.get("PEARL_STAGE5_NANOPEARL_COMMIT_STATE", "0") != "1":
            rebase_batch = getattr(self, "rebase_batch", None)
            if callable(rebase_batch):
                return rebase_batch(updates)
            raise RuntimeError(
                "commit state is disabled and Draft has no rebase_batch fallback"
            )

        join_pipeline = getattr(self, "_join_pipeline", None)
        if callable(join_pipeline):
            join_pipeline()

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(updates):
            if not isinstance(item, dict):
                raise ValueError(f"commit update {index} must be an object")
            external_id = str(item.get("request_id", f"row-{index}"))
            length_only = bool(item.get("length_only", False))
            prefix = item.get("prefix_token_ids")
            target_prefix_len = int(
                item.get(
                    "target_prefix_len",
                    len(prefix) if isinstance(prefix, list) else -1,
                )
            )
            valid_len = int(item.get("valid_len", -1))
            accepted_len = int(item.get("accepted_len", 0))
            draft_len = int(item.get("draft_len", 0))
            replacement = item.get("replacement_token_id")
            finished = bool(item.get("finished", False))
            if length_only:
                if prefix is not None:
                    raise ValueError(
                        f"length-only commit carried token IDs for {external_id!r}"
                    )
                if target_prefix_len <= 0 or valid_len != target_prefix_len - 1:
                    raise ValueError(
                        f"invalid length-only boundary for {external_id!r}"
                    )
                if (
                    finished
                    or accepted_len != draft_len
                ):
                    raise ValueError(
                        f"unsafe length-only verifier result for {external_id!r}"
                    )
                normalized_prefix = None
            else:
                if not isinstance(prefix, list) or not prefix:
                    raise ValueError(
                        f"commit prefix_token_ids must be non-empty for {external_id!r}"
                    )
                normalized_prefix = [int(x) for x in prefix]
                if target_prefix_len != len(normalized_prefix):
                    raise ValueError(
                        f"target_prefix_len mismatch for {external_id!r}"
                    )
                if valid_len < 0 or valid_len > len(normalized_prefix) - 1:
                    raise ValueError(
                        f"invalid valid_len={valid_len} for {external_id!r}"
                    )
            if accepted_len < 0 or draft_len < 0 or accepted_len > draft_len:
                raise ValueError(
                    f"invalid accepted/draft lengths for {external_id!r}"
                )
            normalized.append(
                {
                    "request_id": external_id,
                    "prefix_token_ids": normalized_prefix,
                    "target_prefix_len": target_prefix_len,
                    "valid_len": valid_len,
                    "accepted_len": accepted_len,
                    "draft_len": draft_len,
                    "replacement_token_id": replacement,
                    "finished": finished,
                    "length_only": length_only,
                }
            )

        results: list[dict[str, Any]] = []
        with self._lock:
            for item in normalized:
                external_id = item["request_id"]
                valid_len = item["valid_len"]
                target_prefix_len = item["target_prefix_len"]
                self._activate_request(external_id)
                request = self._request()
                scheduler = self.core.scheduler

                if item["finished"]:
                    results.append(
                        {
                            "request_id": external_id,
                            "valid_len": valid_len,
                            "action": "skip_finished",
                        }
                    )
                    continue

                from vllm.v1.request import RequestStatus

                if not (
                    request.status == RequestStatus.RUNNING
                    and request in scheduler.running
                ):
                    raise RuntimeError(
                        "commit_batch requires a RUNNING persistent Request: "
                        f"{external_id!r} status={request.status!s}"
                    )

                if item["length_only"]:
                    self._sync_model_runner_lengths_only(
                        request,
                        target_prefix_len,
                    )
                    request.is_prefill_chunk = False
                    request.spec_token_ids = []
                    request.num_output_placeholders = 0
                    inflight_prefills = getattr(
                        scheduler, "_inflight_prefills", None
                    )
                    if inflight_prefills is not None:
                        inflight_prefills.discard(request)
                    results.append(
                        {
                            "request_id": external_id,
                            "accepted_len": item["accepted_len"],
                            "draft_len": item["draft_len"],
                            "valid_len": valid_len,
                            "common_len": target_prefix_len,
                            "action": "update_lengths_only_keep_kv",
                        }
                    )
                    if os.environ.get(
                        "PEARL_STAGE5_NANOPEARL_TRACE", "0"
                    ) == "1":
                        print(
                            "[PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_V1] "
                            f"request_id={external_id!r} "
                            f"accepted_len={item['accepted_len']} "
                            f"draft_len={item['draft_len']} "
                            f"valid_len={valid_len} "
                            f"target_prefix_len={target_prefix_len} "
                            "action=update_lengths_only_keep_kv",
                            flush=True,
                        )
                    continue

                prefix = item["prefix_token_ids"]
                old_prefix = list(self.committed_token_ids)
                common_len = 0
                for old_token, new_token in zip(old_prefix, prefix):
                    if old_token != new_token:
                        break
                    common_len += 1
                if common_len < valid_len:
                    raise RuntimeError(
                        "commit prefix diverged before valid_len for "
                        f"{external_id!r}: common_len={common_len} "
                        f"valid_len={valid_len}"
                    )

                self.prompt_token_ids = list(prefix)
                request.prompt_token_ids = list(prefix)
                request.num_prompt_tokens = len(prefix)
                if getattr(request, "prompt_is_token_ids", None) is not None:
                    request.prompt_is_token_ids = [True] * len(prefix)
                self._replace_tokens(prefix)
                request.num_computed_tokens = valid_len
                request.is_prefill_chunk = False
                request.spec_token_ids = []
                request.num_output_placeholders = 0
                self._sync_model_runner_state(prefix, request)
                inflight_prefills = getattr(scheduler, "_inflight_prefills", None)
                if inflight_prefills is not None:
                    inflight_prefills.discard(request)

                results.append(
                    {
                        "request_id": external_id,
                        "accepted_len": item["accepted_len"],
                        "draft_len": item["draft_len"],
                        "valid_len": valid_len,
                        "common_len": common_len,
                        "action": "update_lengths_keep_kv",
                    }
                )
                if os.environ.get(
                    "PEARL_STAGE5_NANOPEARL_TRACE", "0"
                ) == "1":
                    print(
                        "[PEARL_STAGE5_NANOPEARL_COMMIT_STATE_V1] "
                        f"request_id={external_id!r} "
                        f"accepted_len={item['accepted_len']} "
                        f"draft_len={item['draft_len']} "
                        f"valid_len={valid_len} "
                        f"common_len={common_len} "
                        "action=update_lengths_keep_kv",
                        flush=True,
                    )
        return results
    def rebase_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rebase Target prefixes without mistaking optimistic tails for prompts.

        During PRE/POST-VERIFY prefetch, ``self.prompt_token_ids`` is updated
        to the optimistic prefix so vLLM's Request and ModelRunner remain
        synchronized.  That value is *not* an immutable user prompt: it can
        contain draft tokens which Target is expected to roll back.

        Keep one first-authoritative-prefix-minus-one-token anchor per external
        Target request.  The minus-one accounts for the sampled Target token
        that is already present in the custom proposer prefix on the first
        call.  Later Target prefixes must preserve this anchor; only their
        suffix is eligible for same-Request partial-KV rebase.
        """
        if not isinstance(requests, list) or not requests:
            raise ValueError("rebase_batch requires a non-empty requests list")

        join_pipeline = getattr(self, "_join_pipeline", None)
        if callable(join_pipeline):
            join_pipeline()

        normalized: list[tuple[str, list[int]]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                raise ValueError(f"rebase request {index} must be an object")
            external_id = str(item.get("request_id", f"row-{index}"))
            prefix = item.get("prefix_token_ids")
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(
                    "rebase prefix_token_ids must be non-empty for "
                    f"{external_id!r}"
                )
            normalized.append((external_id, [int(x) for x in prefix]))

        true_partial_reuse = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE", "1"
        ) != "0"
        trace = os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1"
        anchor_by_external_id = getattr(
            self, "_nano_rebase_anchor_by_external_id", None
        )
        if anchor_by_external_id is None:
            anchor_by_external_id = {}
            self._nano_rebase_anchor_by_external_id = anchor_by_external_id

        results: list[dict[str, Any]] = []

        with self._lock:
            for external_id, prefix in normalized:
                self._activate_request(external_id)
                old_internal_id = self.request_id
                if old_internal_id is not None:
                    self._pending_tokens.pop(str(old_internal_id), None)

                old_prefix = list(self.committed_token_ids)
                common_len = 0
                for old_token, new_token in zip(old_prefix, prefix):
                    if old_token != new_token:
                        break
                    common_len += 1
                reusable_tokens = max(0, common_len - 1)

                # The first rebase prefix is authoritative Target state.  The
                # proposer includes one already-sampled Target token, so keep
                # the stable portion before that token as this slot's anchor.
                anchor = anchor_by_external_id.get(external_id)
                if anchor is None:
                    anchor = tuple(prefix[:-1] if len(prefix) > 1 else prefix)
                    anchor_by_external_id[external_id] = anchor
                anchor = tuple(int(x) for x in anchor)
                anchor_len = len(anchor)
                anchor_matches = (
                    len(prefix) >= anchor_len
                    and prefix[:anchor_len] == list(anchor)
                )

                reused = False
                action = "fresh_reset"
                reason = "true_partial_reuse_disabled"

                if true_partial_reuse and old_internal_id is not None:
                    from vllm.v1.request import RequestStatus

                    request = self._request()
                    scheduler = self.core.scheduler
                    request_is_running = (
                        request.status == RequestStatus.RUNNING
                        and request in scheduler.running
                    )

                    if not anchor_matches:
                        # An external ID must not silently reuse KV from a
                        # different prompt.  Reset the anchor only after the
                        # request has been sent through the fresh path below.
                        reason = "rebase_anchor_divergence"
                    elif common_len < anchor_len:
                        reason = "common_prefix_before_rebase_anchor"
                    elif reusable_tokens <= 0:
                        reason = "no_reusable_tokens"
                    elif not request_is_running:
                        reason = f"request_not_running:{request.status!s}"
                    else:
                        # The validated helper checks self.prompt_token_ids.
                        # At this point that field may contain optimistic draft
                        # tokens, so temporarily expose only the stable anchor.
                        # The helper then rewrites the Request/runner state to
                        # the complete authoritative Target prefix and retains
                        # the existing Request-owned KV blocks.
                        self.prompt_token_ids = list(anchor)
                        self._requeue_request_preserve_kv(prefix)
                        if self.prompt_token_ids != prefix:
                            self.prompt_token_ids = list(prefix)
                        reused = True
                        action = "retain_partial_tail"
                        reason = "eligible_running_request_anchor_safe"

                if not reused:
                    # Correctness fallback for a new/ineligible slot or a
                    # request whose stable anchor no longer matches.
                    self._reset_request(prefix)
                    anchor_by_external_id[external_id] = tuple(
                        prefix[:-1] if len(prefix) > 1 else prefix
                    )
                    anchor = anchor_by_external_id[external_id]
                    anchor_len = len(anchor)

                new_internal_id = self.request_id
                if new_internal_id is not None:
                    self._pending_tokens.pop(str(new_internal_id), None)

                results.append(
                    {
                        "request_id": external_id,
                        "prefix_len": len(prefix),
                        "common_len": common_len,
                        "reusable_tokens": reusable_tokens,
                        "anchor_len": anchor_len,
                        "action": action,
                    }
                )

                if trace:
                    if reused:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REUSE_V3] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
                            f"anchor_len={anchor_len} "
                            f"prefix_len={len(prefix)} "
                            f"old_internal={old_internal_id!r} "
                            f"new_internal={new_internal_id!r} "
                            "action=retain_partial_tail "
                            f"reason={reason}",
                            flush=True,
                        )
                    else:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_REBASE_V3] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
                            f"anchor_len={anchor_len} "
                            f"prefix_len={len(prefix)} "
                            f"old_internal={old_internal_id!r} "
                            f"new_internal={new_internal_id!r} "
                            "action=fresh_reset "
                            f"reason={reason}",
                            flush=True,
                        )

        return results

    def propose_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reuse compatible rows and rebase only mismatched rows."""
        if not requests:
            return []
        if len(requests) > self.max_num_seqs:
            raise ValueError(
                f"Draft batch={len(requests)} exceeds "
                f"max_num_seqs={self.max_num_seqs}"
            )

        normalized: list[tuple[str, list[int], int]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                raise ValueError(f"Draft request {index} must be an object")
            external_id = str(item.get("request_id", f"row-{index}"))
            if external_id in seen_ids:
                raise ValueError(
                    f"duplicate external Draft request ID: {external_id!r}"
                )
            seen_ids.add(external_id)
            prefix = item.get("prefix_token_ids")
            gamma = int(item.get("gamma", 0))
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(
                    f"prefix_token_ids must be non-empty for {external_id!r}"
                )
            if gamma < 0:
                raise ValueError(f"gamma must be non-negative for {external_id!r}")
            normalized.append(
                (external_id, [int(token_id) for token_id in prefix], gamma)
            )

        pipeline = self._pipeline_enabled()
        self._join_pipeline()
        if not pipeline:
            self._pipeline_candidates.clear()

        with self._lock:
            fast_rows: dict[str, dict[str, Any]] = {}
            if pipeline:
                for external_id, prefix, gamma in normalized:
                    if self._pipeline_candidate_can_serve(
                        external_id, prefix, gamma
                    ):
                        fast_rows[external_id] = self._pipeline_candidates[
                            external_id
                        ]

            fallback_count = len(normalized) - len(fast_rows)
            if pipeline:
                self._pipeline_trace(
                    "row_partition "
                    f"batch={len(normalized)} "
                    f"reuse_rows={len(fast_rows)} "
                    f"fallback_rows={fallback_count}"
                )

            internal_to_external: dict[str, str] = {}
            target_counts: dict[str, int] = {}
            draft_by_id: dict[str, list[int]] = {}

            for external_id, prefix, gamma in normalized:
                self._activate_request(external_id)
                candidate = fast_rows.get(external_id)
                if candidate is not None:
                    generated = candidate["generated"]
                    draft_by_id[external_id] = [
                        int(token_id)
                        for token_id in generated[
                            len(prefix) : len(prefix) + gamma
                        ]
                    ]
                    internal_id = str(candidate["internal_id"])
                    internal_to_external[internal_id] = external_id
                    # A compatible row already has its current proposal.
                    # Count zero here: if EngineCore advances this row while
                    # serving fallback rows, its output is captured into
                    # _pending_tokens and appended below.
                    target_counts[external_id] = 0
                    continue

                if pipeline:
                    self._pipeline_candidates.pop(external_id, None)
                self.sync_prefix(prefix)
                internal_id = self.request_id
                if internal_id is None:
                    raise RuntimeError(
                        f"Draft request was not created for {external_id!r}"
                    )
                internal_id = str(internal_id)
                # Discard output that belongs to the old Target prefix.
                self._pending_tokens.pop(internal_id, None)
                internal_to_external[internal_id] = external_id
                target_counts[external_id] = gamma

            if any(count > 0 for count in target_counts.values()):
                collected = self._collect_batch_tokens(
                    internal_to_external=internal_to_external,
                    target_counts=target_counts,
                )
            else:
                collected = {
                    external_id: [] for external_id, _prefix, _gamma in normalized
                }

            base_generated: dict[str, list[int]] = {}
            for external_id, prefix, gamma in normalized:
                internal_id = next(
                    key
                    for key, value in internal_to_external.items()
                    if value == external_id
                )
                candidate = fast_rows.get(external_id)
                if candidate is not None:
                    generated = list(candidate["generated"])
                else:
                    generated = list(prefix) + list(
                        collected.get(external_id, [])
                    )
                # Capture any token produced for a fast row while the
                # fallback rows were being advanced, and any output beyond
                # the requested gamma for a fallback row.
                pending = self._pending_tokens.pop(internal_id, None)
                if pending:
                    generated.extend(int(token_id) for token_id in pending)

                # Fallback rows were synchronously advanced above.  Their
                # current proposal must be returned even when the
                # row-wise pipeline is enabled; otherwise a mixed batch
                # would silently return an empty proposal for every
                # rebased row.
                if candidate is None:
                    draft_by_id[external_id] = [
                        int(token_id)
                        for token_id in collected.get(external_id, [])[:gamma]
                    ]

                if pipeline:
                    if candidate is None:
                        candidate = {
                            "internal_id": internal_id,
                            "base_prefix": list(prefix),
                            "generated": generated,
                        }
                        self._pipeline_candidates[external_id] = candidate
                    else:
                        candidate["generated"] = generated
                    base_generated[external_id] = list(generated)
                elif candidate is None:
                    draft_by_id[external_id] = [
                        int(token_id)
                        for token_id in collected.get(external_id, [])[:gamma]
                    ]

            results = [
                {
                    "request_id": external_id,
                    "draft_token_ids": draft_by_id.get(
                        external_id,
                        [],
                    )[:gamma],
                }
                for external_id, _prefix, gamma in normalized
            ]
            if not pipeline:
                return results

            # Start a fresh lookahead for every current row.  Compatible
            # rows continue from their already-generated stream; rebased
            # rows continue from the new Target prefix.
            lookahead = {
                external_id: self._pipeline_lookahead(gamma)
                for external_id, _prefix, gamma in normalized
                if gamma > 0
            }
            self._start_pipeline_prefetch(
                internal_to_external,
                lookahead,
                base_generated,
            )
            return results

    def propose(
        self,
        request_id: str | list[int],
        prefix_token_ids: list[int] | int,
        gamma: int | None = None,
    ) -> list[int]:
        """Compatibility wrapper for the legacy single-request RPC."""
        if gamma is None:
            external_id = "target-0"
            legacy_prefix = request_id
            legacy_gamma = prefix_token_ids
            if not isinstance(legacy_prefix, list):
                raise TypeError("legacy Draft prefix must be a list")
            prefix_token_ids = legacy_prefix
            gamma = int(legacy_gamma)
        else:
            external_id = str(request_id)
            if not isinstance(prefix_token_ids, list):
                raise TypeError("Draft prefix must be a list")

        result = self.propose_batch(
            [
                {
                    "request_id": external_id,
                    "prefix_token_ids": prefix_token_ids,
                    "gamma": int(gamma),
                }
            ]
        )
        return result[0]["draft_token_ids"] if result else []

    def shutdown(self) -> None:
        thread = self._pipeline_thread
        if thread is not None:
            thread.join()
            self._pipeline_thread = None
        with self._lock:
            request_ids = [
                state["request_id"]
                for state in self._states.values()
                if state.get("request_id") is not None
            ]
            if request_ids:
                try:
                    self.core_client.abort_requests(request_ids)
                except Exception:
                    pass
            self._pending_tokens.clear()
            self._states.clear()
            self._active_key = None
            shutdown = getattr(self.engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except AttributeError as exc:
                    # Some EngineCore cleanup paths remove model_executor
                    # before the compatibility shutdown hook is called.
                    print(
                        "[stage5] draft shutdown compatibility fallback: "
                        f"ignored {exc}",
                        flush=True,
                    )
