#!/usr/bin/env python3
"""Add correctness-first batch>1 support to the Stage-5 nano-PEARL bridge.

The first Stage-5 bridge used one Target row and one persistent Draft state.
This patch keeps the Draft requests isolated by Target request ID, adds a
batch proposal RPC, passes Target request IDs into the custom proposer, and
lets the coordinator submit a real Target batch.

The Draft proposals are intentionally serialized inside one worker for this
first batch implementation.  Target verification is batched; Draft-side
parallel scheduling/overlap is a later optimization.

The patch is source-only and opt-in through the command-line batch settings:

    --max-num-seqs 2 --batch-size 2

Default behavior remains batch size 1.  Every real modification creates one
new backup directory; dry-run creates neither a backup nor a source change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent


MARKER = "# PEARL_STAGE5_BATCH_GT1_V1"

TARGETS = (
    Path("pearl_stage5_proposer.py"),
    Path("pearl_stage5_worker.py"),
    Path("pearl_stage5_draft.py"),
    Path("pearl_stage5_coordinator.py"),
    Path("pearl_stage5_gsm8k_ac_benchmark_v1.py"),
    Path("vllm_ascend/worker/model_runner_v1.py"),
)


def add_marker(source: str) -> str:
    return source if MARKER in source else MARKER + "\n" + source


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(
    source: str,
    method_name: str,
    next_method_name: str,
    replacement: str,
) -> str:
    start_anchor = f"    def {method_name}"
    end_anchor = f"    def {next_method_name}"
    start = source.find(start_anchor)
    end = source.find(end_anchor, start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method_name}: method boundary not found; no files were changed"
        )
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def replace_function(
    source: str,
    function_name: str,
    next_function_name: str,
    replacement: str,
) -> str:
    start_anchor = f"def {function_name}"
    end_anchor = f"def {next_function_name}"
    start = source.find(start_anchor)
    end = source.find(end_anchor, start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{function_name}: function boundary not found; no files were changed"
        )
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def class_block(block: str) -> str:
    """Indent a dedented method block into a class body."""
    lines = block.strip("\n").splitlines()
    return "\n".join(("    " + line if line else "") for line in lines)


def indent_block(block: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(
        (prefix + line if line else "")
        for line in block.strip("\n").splitlines()
    )


PROPOSER_METHOD = dedent(
    '''
        def propose(
            self,
            sampled_token_ids: list[list[int]],
            num_tokens_no_spec: Any,
            token_ids_cpu: Any,
            slot_mappings: Any = None,
            request_ids: Any = None,
        ) -> list[list[int]]:
            """Return one Draft proposal list for every Target batch row.

            The custom proposer API does not carry request IDs by default, so
            the Ascend model runner passes ``input_batch.req_ids`` explicitly.
            The external Draft worker keeps one persistent state per ID.
            """
            batch_size = len(sampled_token_ids)
            if batch_size == 0:
                return []

            if request_ids is None:
                row_request_ids = [f"row-{index}" for index in range(batch_size)]
            else:
                raw_ids = request_ids.tolist() if hasattr(request_ids, "tolist") else list(request_ids)
                row_request_ids = [str(value) for value in raw_ids[:batch_size]]
                if len(row_request_ids) < batch_size:
                    row_request_ids.extend(
                        f"row-{index}" for index in range(len(row_request_ids), batch_size)
                    )

            def scalar_at(values: Any, row: int) -> int:
                value = values[row] if hasattr(values, "__getitem__") else values
                if hasattr(value, "item"):
                    value = value.item()
                return int(value)

            def row_ids(row: int, count: int) -> list[int]:
                values = token_ids_cpu[row, :count]
                if hasattr(values, "tolist"):
                    values = values.tolist()
                return [int(token_id) for token_id in values]

            proposals = [[] for _ in range(batch_size)]
            requests: list[dict[str, Any]] = []
            for row, sampled_row in enumerate(sampled_token_ids):
                sampled = [
                    int(token_id)
                    for token_id in sampled_row
                    if int(token_id) >= 0
                ]
                # During initial prefill there is no just-sampled token for
                # this row.  It must still occupy a result slot, but does not
                # need a Draft request yet.
                if not sampled:
                    continue

                count = scalar_at(num_tokens_no_spec, row)
                prefix = row_ids(row, count)
                if prefix[-len(sampled) :] != sampled:
                    prefix.extend(sampled)
                requests.append(
                    {
                        "request_id": row_request_ids[row],
                        "prefix_token_ids": prefix,
                        "gamma": self.gamma,
                    }
                )

            if not requests:
                return proposals

            print(
                "[target proposer] "
                f"batch_size={batch_size} request_count={len(requests)} "
                f"request_ids={[item['request_id'] for item in requests]} "
                f"gamma={self.gamma}",
                flush=True,
            )
            with self._lock:
                response = self._request(
                    {
                        "cmd": "draft_batch",
                        "requests": requests,
                    }
                )

            results = response.get("results")
            if not isinstance(results, list):
                raise RuntimeError(
                    "Draft batch response is missing a list-valued 'results'"
                )
            result_by_id = {
                str(item.get("request_id")): item
                for item in results
                if isinstance(item, dict) and item.get("request_id") is not None
            }
            for row, request in enumerate(requests):
                item = result_by_id.get(str(request["request_id"]))
                if item is None:
                    raise RuntimeError(
                        "Draft batch response omitted request "
                        f"{request['request_id']!r}"
                    )
                draft_ids = item.get("draft_token_ids", [])
                if not isinstance(draft_ids, list):
                    raise RuntimeError(
                        "Draft batch response has non-list draft_token_ids for "
                        f"{request['request_id']!r}"
                    )
                target_row = row_request_ids.index(str(request["request_id"]))
                proposals[target_row] = [
                    int(token_id) for token_id in draft_ids[: self.gamma]
                ]
            return proposals
    '''
)


def patch_proposer(source: str) -> str:
    source = add_marker(source)
    start = source.find("    def propose(")
    if start < 0:
        raise RuntimeError(
            "proposer propose method not found; no files were changed"
        )
    suffix = ""
    top_level_main = source.find("\nif __name__", start)
    if top_level_main >= 0:
        suffix = source[top_level_main:]
    return source[:start] + class_block(PROPOSER_METHOD).rstrip() + "\n" + suffix


TARGET_HANDLER = dedent(
    '''
    def _handle_target_generate(
        llm: Any,
        message: dict[str, Any],
        default_max_tokens: int,
        max_num_seqs: int,
    ):
        from vllm import SamplingParams

        prompt_token_ids_batch = message.get("prompt_token_ids_batch")
        if isinstance(prompt_token_ids_batch, list):
            if not prompt_token_ids_batch:
                raise ValueError("prompt_token_ids_batch must not be empty")
            if len(prompt_token_ids_batch) > max_num_seqs:
                raise ValueError(
                    f"batch={len(prompt_token_ids_batch)} exceeds "
                    f"max_num_seqs={max_num_seqs}"
                )
            prompts = []
            for row in prompt_token_ids_batch:
                if not isinstance(row, list) or not row:
                    raise ValueError(
                        "each prompt_token_ids_batch row must be a non-empty list"
                    )
                prompts.append({"prompt_token_ids": [int(x) for x in row]})
        else:
            prompt_token_ids = message.get("prompt_token_ids")
            prompt = message.get("prompt")
            if isinstance(prompt_token_ids, list) and prompt_token_ids:
                prompts = [{"prompt_token_ids": [int(x) for x in prompt_token_ids]}]
            elif isinstance(prompt, str) and prompt:
                prompts = [prompt]
            else:
                raise ValueError("request requires prompt or prompt_token_ids")

        max_tokens = int(message.get("max_tokens", default_max_tokens))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                {
                    "token_ids": [int(x) for x in completion.token_ids],
                    "text": completion.text,
                }
            )
        response: dict[str, Any] = {
            "status": "result",
            "results": results,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        # Keep the old single-result socket shape usable for old callers.
        if len(results) == 1:
            response.update(results[0])
        return response


    def _draft_proposal_loop(
        server: socket.socket,
        engine: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                server.settimeout(0.5)
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                with conn.makefile("r", encoding="utf-8") as reader:
                    while not stop_event.is_set():
                        message = receive_message(reader)
                        if message is None:
                            break
                        try:
                            command = message.get("cmd")
                            legacy_single = command == "draft"
                            if legacy_single:
                                raw_requests = [message]
                            elif command == "draft_batch":
                                raw_requests = message.get("requests")
                                if not isinstance(raw_requests, list) or not raw_requests:
                                    raise ValueError(
                                        "draft_batch requires a non-empty requests list"
                                    )
                            else:
                                raise ValueError(f"unknown proposal command: {command!r}")

                            results = []
                            for index, request in enumerate(raw_requests):
                                if not isinstance(request, dict):
                                    raise ValueError(
                                        f"draft request {index} must be an object"
                                    )
                                request_id = str(request.get("request_id", f"row-{index}"))
                                prefix = request.get("prefix_token_ids")
                                gamma = int(request.get("gamma", 0))
                                if not isinstance(prefix, list) or not prefix:
                                    raise ValueError(
                                        f"prefix_token_ids must be non-empty for {request_id!r}"
                                    )
                                draft_ids = engine.propose(
                                    request_id,
                                    [int(x) for x in prefix],
                                    gamma,
                                )
                                print(
                                    "[draft] proposal "
                                    f"request_id={request_id!r} "
                                    f"prefix_len={len(prefix)} gamma={gamma} "
                                    f"draft_ids={draft_ids}",
                                    flush=True,
                                )
                                results.append(
                                    {
                                        "request_id": request_id,
                                        "draft_token_ids": draft_ids,
                                        "prefix_len": len(prefix),
                                    }
                                )

                            if legacy_single:
                                response = {
                                    "status": "result",
                                    "draft_token_ids": results[0]["draft_token_ids"],
                                    "prefix_len": results[0]["prefix_len"],
                                    "request_id": results[0]["request_id"],
                                }
                            else:
                                response = {"status": "result", "results": results}
                            send_message(conn, response)
                        except Exception as exc:
                            traceback.print_exc()
                            send_message(conn, {"status": "error", "error": repr(exc)})


    def _run_draft(args: argparse.Namespace, control_server: socket.socket) -> None:
    '''
)


def patch_worker(source: str) -> str:
    source = add_marker(source)
    source = replace_once(
        source,
        '    parser.add_argument("--default-max-tokens", type=int, default=8)\n',
        '    parser.add_argument("--default-max-tokens", type=int, default=8)\n'
        '    parser.add_argument("--max-num-seqs", type=int, default=1)\n',
        "worker parser batch anchor",
    )
    source = replace_once(
        source,
        "        max_num_seqs=1,\n",
        "        max_num_seqs=args.max_num_seqs,\n",
        "target max_num_seqs anchor",
    )
    handler_start = TARGET_HANDLER.find("def _handle_target_generate(")
    loop_start = TARGET_HANDLER.find("def _draft_proposal_loop(")
    run_draft_start = TARGET_HANDLER.find("def _run_draft(")
    if handler_start < 0 or loop_start < 0 or run_draft_start < 0:
        raise RuntimeError(
            "worker batch method boundaries are malformed; no files were changed"
        )
    handler_replacement = TARGET_HANDLER[handler_start:loop_start]
    loop_replacement = TARGET_HANDLER[loop_start:run_draft_start]
    source = replace_function(
        source,
        "_handle_target_generate(",
        "_draft_proposal_loop(",
        handler_replacement,
    )
    source = replace_function(
        source,
        "_draft_proposal_loop(",
        "_run_draft(",
        loop_replacement,
    )
    source = replace_once(
        source,
        "        engine = PersistentDraftEngine(args.model, args.max_model_len)\n",
        "        engine = PersistentDraftEngine(\n"
        "            args.model, args.max_model_len, args.max_num_seqs\n"
        "        )\n",
        "draft engine batch anchor",
    )
    source = replace_once(
        source,
        "llm, message, args.default_max_tokens\n",
        "llm, message, args.default_max_tokens, args.max_num_seqs\n",
        "target handler call anchor",
    )
    return source


DRAFT_PROPERTIES = dedent(
    '''
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
    '''
)


STEP_METHOD = dedent(
    '''
        def _step_one(self) -> int:
            if self.request_id is None:
                raise RuntimeError("Draft request has not been created")

            pending = self._pending_tokens.get(self.request_id)
            if pending:
                return pending.popleft()

            while True:
                outputs = self.core_client.get_output()
                for output in outputs.outputs:
                    output_request_id = str(output.request_id)
                    new_token_ids = getattr(output, "new_token_ids", None) or []
                    if new_token_ids:
                        queue = self._pending_tokens.setdefault(
                            output_request_id, deque()
                        )
                        queue.extend(int(token_id) for token_id in new_token_ids)
                    if (
                        output_request_id == self.request_id
                        and not new_token_ids
                        and output.finished
                    ):
                        raise RuntimeError(
                            "Draft request finished before returning a token: "
                            f"{output.finish_reason}"
                        )

                pending = self._pending_tokens.get(self.request_id)
                if pending:
                    return pending.popleft()
                if self.request_id not in self.core.scheduler.requests:
                    raise RuntimeError("Draft request disappeared while decoding")
    '''
)


PROPOSE_METHOD = dedent(
    '''
        def propose(
            self,
            request_id: str | list[int],
            prefix_token_ids: list[int] | int,
            gamma: int | None = None,
        ) -> list[int]:
            # Keep the old two-argument call usable while the worker rolls out
            # the request-aware three-argument protocol.
            if gamma is None:
                external_request_id = "target-0"
                legacy_prefix = request_id
                legacy_gamma = prefix_token_ids
                if not isinstance(legacy_prefix, list):
                    raise TypeError("legacy Draft prefix must be a list")
                prefix_token_ids = legacy_prefix
                gamma = int(legacy_gamma)
            else:
                external_request_id = str(request_id)
                if not isinstance(prefix_token_ids, list):
                    raise TypeError("Draft prefix must be a list")

            if gamma <= 0:
                return []

            with self._lock:
                self._activate_request(external_request_id)
                self.sync_prefix([int(token_id) for token_id in prefix_token_ids])
                draft_token_ids: list[int] = []
                for _ in range(int(gamma)):
                    if (
                        len(self.committed_token_ids) + len(draft_token_ids)
                        >= self.max_model_len
                    ):
                        break
                    draft_token_ids.append(self._step_one())
                return draft_token_ids
    '''
)


SHUTDOWN_METHOD = dedent(
    '''
        def shutdown(self) -> None:
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
    '''
)


def patch_draft(source: str) -> str:
    source = add_marker(source)
    source = replace_once(
        source,
        "import os\nimport threading\n",
        "import os\nimport threading\nfrom collections import deque\n",
        "draft deque import anchor",
    )
    source = replace_once(
        source,
        "    def __init__(self, model: str, max_model_len: int) -> None:\n",
        "    def __init__(\n"
        "        self, model: str, max_model_len: int, max_num_seqs: int = 1\n"
        "    ) -> None:\n",
        "draft constructor anchor",
    )
    source = replace_once(
        source,
        "        if self.max_model_len < 2:\n"
        "            raise ValueError(\"max_model_len must be at least 2\")\n\n"
        "        engine_args = EngineArgs(\n",
        "        if self.max_model_len < 2:\n"
        "            raise ValueError(\"max_model_len must be at least 2\")\n"
        "        self.max_num_seqs = int(max_num_seqs)\n"
        "        if self.max_num_seqs < 1:\n"
        "            raise ValueError(\"max_num_seqs must be positive\")\n\n"
        "        engine_args = EngineArgs(\n",
        "draft max_num_seqs validation anchor",
    )
    source = replace_once(
        source,
        "            max_num_seqs=1,\n",
        "            max_num_seqs=self.max_num_seqs,\n",
        "draft EngineArgs max_num_seqs anchor",
    )
    source = replace_once(
        source,
        "            max_num_batched_tokens=self.max_model_len,\n",
        "            max_num_batched_tokens=(\n"
        "                self.max_model_len * self.max_num_seqs\n"
        "            ),\n",
        "draft EngineArgs batched-token anchor",
    )
    old_state = (
        "        self.request_id: str | None = None\n"
        "        self.prompt_token_ids: list[int] | None = None\n"
        "        self.committed_token_ids: list[int] = []\n"
        "        self._request_counter = 0\n"
    )
    new_state = (
        "        self._states: dict[str, dict[str, Any]] = {}\n"
        "        self._active_key: str | None = None\n"
        "        self._pending_tokens: dict[str, deque[int]] = {}\n"
        "        self._request_counter = 0\n\n"
        + class_block(DRAFT_PROPERTIES).rstrip()
        + "\n"
    )
    source = replace_once(source, old_state, new_state, "draft state storage anchor")

    source = replace_method(
        source, "_step_one(", "propose(", class_block(STEP_METHOD)
    )
    source = replace_method(
        source, "propose(", "shutdown(", class_block(PROPOSE_METHOD)
    )
    shutdown_start = source.find("    def shutdown(")
    if shutdown_start < 0:
        raise RuntimeError("draft shutdown method not found; no files were changed")
    suffix = ""
    if "\nif __name__" in source[shutdown_start:]:
        suffix = source[source.find("\nif __name__", shutdown_start):]
    source = (
        source[:shutdown_start]
        + class_block(SHUTDOWN_METHOD).rstrip()
        + "\n"
        + suffix
    )
    return source


COORDINATOR_BATCH_LOOP = dedent(
    '''
            prompts = args.prompts or ["The capital of France is"]
            if args.max_num_seqs < 1:
                raise ValueError("--max-num-seqs must be positive")
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            if args.batch_size > args.max_num_seqs:
                raise ValueError(
                    "--batch-size cannot exceed --max-num-seqs"
                )

            for batch_start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[batch_start : batch_start + args.batch_size]
                batch_prompt_ids = [
                    [
                        int(x)
                        for x in tokenizer.encode(
                            prompt, add_special_tokens=True
                        )
                    ]
                    for prompt in batch_prompts
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids_batch": batch_prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                results = response.get("results")
                if not isinstance(results, list):
                    results = [response]
                if len(results) != len(batch_prompts):
                    raise RuntimeError(
                        "Target returned an unexpected result count: "
                        f"expected={len(batch_prompts)} got={len(results)}"
                    )
                for offset, (prompt, result) in enumerate(
                    zip(batch_prompts, results)
                ):
                    output_ids = [
                        int(x) for x in result.get("token_ids", [])
                    ]
                    print(
                        f"[stage5] prompt {batch_start + offset + 1}: "
                        f"{prompt!r}; output_ids={output_ids}; "
                        f"text={result.get('text')!r}; "
                        f"elapsed_ms={response.get('elapsed_ms')}",
                        flush=True,
                    )
    '''
)


def patch_coordinator(source: str) -> str:
    source = add_marker(source)
    source = replace_once(
        source,
        '    parser.add_argument("--gamma", type=int, default=4)\n',
        '    parser.add_argument("--gamma", type=int, default=4)\n'
        '    parser.add_argument("--max-num-seqs", type=int, default=1)\n'
        '    parser.add_argument("--batch-size", type=int, default=1)\n',
        "coordinator batch parser anchor",
    )
    source = replace_once(
        source,
        "    max_model_len: int,\n    gamma: int,\n",
        "    max_model_len: int,\n    gamma: int,\n    max_num_seqs: int,\n",
        "coordinator launch signature anchor",
    )
    source = replace_once(
        source,
        '        "--gamma",\n        str(gamma),\n',
        '        "--gamma",\n        str(gamma),\n        "--max-num-seqs",\n        str(max_num_seqs),\n',
        "coordinator launch command anchor",
    )
    source = replace_once(
        source,
        "                args.max_model_len,\n                args.gamma,\n                proposal_socket=draft_proposal,\n",
        "                args.max_model_len,\n                args.gamma,\n                args.max_num_seqs,\n                proposal_socket=draft_proposal,\n",
        "coordinator draft launch call anchor",
    )
    source = replace_once(
        source,
        "                args.max_model_len,\n                args.gamma,\n                draft_socket=draft_proposal,\n",
        "                args.max_model_len,\n                args.gamma,\n                args.max_num_seqs,\n                draft_socket=draft_proposal,\n",
        "coordinator target launch call anchor",
    )
    source = replace_once(
        source,
        "    if args.gamma <= 0 or args.max_tokens <= 0:\n"
        "        raise ValueError(\"--gamma and --max-tokens must be positive\")\n",
        "    if args.gamma <= 0 or args.max_tokens <= 0:\n"
        "        raise ValueError(\"--gamma and --max-tokens must be positive\")\n"
        "    if args.max_num_seqs < 1 or args.batch_size < 1:\n"
        "        raise ValueError(\"--max-num-seqs and --batch-size must be positive\")\n"
        "    if args.batch_size > args.max_num_seqs:\n"
        "        raise ValueError(\"--batch-size cannot exceed --max-num-seqs\")\n",
        "coordinator argument validation anchor",
    )
    start = source.find("            prompts = args.prompts or")
    end = source.find(
        '            print(\n                "[stage5] persistent Draft/Target path completed;',
        start,
    )
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "coordinator prompt loop boundaries not found; no files were changed"
        )
    source = (
        source[:start]
        + indent_block(COORDINATOR_BATCH_LOOP, 12)
        + "\n\n"
        + source[end:]
    )
    return source


def patch_benchmark(source: str) -> str:
    source = add_marker(source)
    source = replace_once(
        source,
        '    parser.add_argument("--gamma", type=int, default=2)\n',
        '    parser.add_argument("--gamma", type=int, default=2)\n'
        '    parser.add_argument("--max-num-seqs", type=int, default=1)\n'
        '    parser.add_argument("--batch-size", type=int, default=1)\n',
        "benchmark batch parser anchor",
    )
    source = replace_once(
        source,
        "    if args.num_samples <= 0 or args.max_tokens <= 0 or args.gamma <= 0:\n"
        '        raise ValueError("num-samples, max-tokens, and gamma must be positive")\n',
        "    if args.num_samples <= 0 or args.max_tokens <= 0 or args.gamma <= 0:\n"
        '        raise ValueError("num-samples, max-tokens, and gamma must be positive")\n'
        "    if args.max_num_seqs < 1 or args.batch_size < 1:\n"
        '        raise ValueError("max-num-seqs and batch-size must be positive")\n'
        "    if args.batch_size > args.max_num_seqs:\n"
        '        raise ValueError("batch-size cannot exceed max-num-seqs")\n',
        "benchmark argument validation anchor",
    )
    source = replace_once(
        source,
        '        "--gamma",\n        str(args.gamma),\n',
        '        "--gamma",\n        str(args.gamma),\n        "--max-num-seqs",\n        str(args.max_num_seqs),\n        "--batch-size",\n        str(args.batch_size),\n',
        "benchmark coordinator command anchor",
    )
    source = replace_once(
        source,
        '    print(f"gamma: {args.gamma}", flush=True)\n',
        '    print(f"gamma: {args.gamma}", flush=True)\n'
        '    print(f"max_num_seqs: {args.max_num_seqs}", flush=True)\n'
        '    print(f"batch_size: {args.batch_size}", flush=True)\n',
        "benchmark status output anchor",
    )
    return source


def patch_model_runner(source: str) -> str:
    source = add_marker(source)
    old = (
        "            draft_token_ids = self.drafter.propose(\n"
        "                valid_sampled_token_ids,\n"
        "                self.input_batch.num_tokens_no_spec,\n"
        "                self.input_batch.token_ids_cpu,\n"
        "                slot_mappings=None,\n"
        "            )\n"
    )
    new = (
        "            draft_token_ids = self.drafter.propose(\n"
        "                valid_sampled_token_ids,\n"
        "                self.input_batch.num_tokens_no_spec,\n"
        "                self.input_batch.token_ids_cpu,\n"
        "                slot_mappings=None,\n"
        "                request_ids=list(\n"
        "                    self.input_batch.req_ids[: len(valid_sampled_token_ids)]\n"
        "                ),\n"
        "            )\n"
    )
    return replace_once(source, old, new, "Ascend model-runner request-id anchor")


PATCHERS = {
    Path("pearl_stage5_proposer.py"): patch_proposer,
    Path("pearl_stage5_worker.py"): patch_worker,
    Path("pearl_stage5_draft.py"): patch_draft,
    Path("pearl_stage5_coordinator.py"): patch_coordinator,
    Path("pearl_stage5_gsm8k_ac_benchmark_v1.py"): patch_benchmark,
    Path("vllm_ascend/worker/model_runner_v1.py"): patch_model_runner,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_batch_gt1_v1.{stamp}"
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def write_atomic(target: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        dir=str(target.parent),
        text=True,
    )
    try:
        os.chmod(temp_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    originals: dict[Path, bytes] = {}
    patched: dict[Path, str] = {}
    states: dict[Path, str] = {}
    for relative in TARGETS:
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        data = target.read_bytes()
        source = data.decode("utf-8")
        originals[relative] = data
        states[relative] = "post" if MARKER in source else "pre"

    if any(state == "post" for state in states.values()):
        if not all(state == "post" for state in states.values()):
            raise RuntimeError(
                f"partial batch patch state: {states}; no files were changed"
            )
        for relative in TARGETS:
            print(f"target: {repo / relative}")
            print("state: post")
        print("already patched: no files were changed and no backup was created")
        return

    for relative in TARGETS:
        target = repo / relative
        source = originals[relative].decode("utf-8")
        transformed = PATCHERS[relative](source)
        compile(transformed, str(target), "exec")
        patched[relative] = transformed
        print(f"target: {target}")
        print("state: pre")
    print(
        "change: batch>1 Target verification with request-isolated persistent "
        "Draft states (Draft RPC serialized)"
    )

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "marker": MARKER,
        "mode": "batch_gt1_target_verification_request_isolated_draft",
        "draft_parallelism": "serialized_per_request_for_correctness",
        "files": {},
    }
    for relative in TARGETS:
        target = repo / relative
        backup_file = backup_dir / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_file)
        manifest["files"][str(relative)] = {
            "target": str(target),
            "backup_file": str(backup_file),
            "original_sha256": sha256_bytes(originals[relative]),
            "original_size": len(originals[relative]),
        }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    written: list[Path] = []
    try:
        for relative in TARGETS:
            target = repo / relative
            write_atomic(target, patched[relative], target.stat().st_mode)
            written.append(relative)
    except Exception:
        for relative in written:
            shutil.copy2(backup_dir / relative, repo / relative)
        raise
    print(f"backup: {backup_dir}")
    for relative in TARGETS:
        print(f"patched: {repo / relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
