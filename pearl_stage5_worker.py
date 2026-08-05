# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V4
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V2
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V1
# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_WORKER_V1
# PEARL_STAGE5_NANOPEARL_PROPOSER_V4
# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_WORKER_V1
# PEARL_STAGE5_BATCH_GT1_V3
# PEARL_STAGE5_BATCH_GT1_V1
#!/usr/bin/env python3
"""Two-card Stage-5 worker with an opt-in synchronous AC control.

PEARL_STAGE5_SERIAL_CONTROL_V1
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def send_message(conn: socket.socket, message: dict[str, Any]) -> None:
    conn.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))


def receive_message(reader) -> dict[str, Any] | None:
    line = reader.readline()
    return json.loads(line) if line else None


def bind_server(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(2)
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("draft", "target"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--proposal-socket")
    parser.add_argument("--draft-socket")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--default-max-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    return parser


def _load_target(args: argparse.Namespace):
    from vllm import LLM

    return LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        # PEARL_STAGE5_TARGET_SYNC_CONTROL_V1
        # PEARL_STAGE5_TARGET_ASYNC_OPTIN_V1
        async_scheduling=(
            os.environ.get("PEARL_STAGE5_TARGET_ASYNC_SCHEDULING", "0") == "1"
        ),
        trust_remote_code=True,
        speculative_config={
            "method": "custom_class",
            "model": (
                "pearl_stage5_nanoparl_proposer_v3.PearlNanoPearlProposer"
                if os.environ.get("PEARL_STAGE5_SERIAL_CONTROL", "0") == "1"
                else "pearl_stage5_nanoparl_proposer_v3.PearlNanoPearlProposer"
            ),  # PEARL_STAGE5_NANOPEARL_PROPOSER_V3  # PEARL_STAGE5_NANOPEARL_PROPOSER_V2  # PEARL_STAGE5_NANOPEARL_PROPOSER_V1
            "num_speculative_tokens": args.gamma,
        },
    )


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
                        if command == "rebase_batch":
                            raw_requests = message.get("requests")
                            if not isinstance(raw_requests, list) or not raw_requests:
                                raise ValueError(
                                    "rebase_batch requires a non-empty requests list"
                                )
                            requests = []
                            for index, item in enumerate(raw_requests):
                                if not isinstance(item, dict):
                                    raise ValueError(
                                        f"rebase request {index} must be an object"
                                    )
                                prefix = item.get("prefix_token_ids")
                                if not isinstance(prefix, list) or not prefix:
                                    raise ValueError(
                                        "rebase prefix_token_ids must be non-empty"
                                    )
                                requests.append(
                                    {
                                        "request_id": str(
                                            item.get("request_id", f"row-{index}")
                                        ),
                                        "prefix_token_ids": [int(x) for x in prefix],
                                    }
                                )
                            rebase_batch = getattr(engine, "rebase_batch", None)
                            if not callable(rebase_batch):
                                raise RuntimeError(
                                    "Draft engine has no rebase_batch(); "
                                    "stale nano-PEARL state cannot be safely discarded"
                                )
                            result = rebase_batch(requests)
                            print(
                                "[draft] rebase_batch "
                                f"batch_size={len(requests)} "
                                f"request_ids={[item['request_id'] for item in requests]}",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {"status": "result", "results": result or []},
                            )
                            continue

                        if command == "commit_batch":
                            raw_updates = message.get("updates")
                            if not isinstance(raw_updates, list) or not raw_updates:
                                raise ValueError(
                                    "commit_batch requires a non-empty updates list"
                                )
                            commit_batch = getattr(engine, "commit_batch", None)
                            if not callable(commit_batch):
                                raise RuntimeError(
                                    "Draft engine has no commit_batch(); "
                                    "apply the nano-PEARL commit-state patch first"
                                )
                            updates = []
                            for index, item in enumerate(raw_updates):
                                if not isinstance(item, dict):
                                    raise ValueError(
                                        f"commit update {index} must be an object"
                                    )
                                length_only = bool(item.get("length_only", False))
                                prefix = item.get("prefix_token_ids")
                                valid_len = int(item.get("valid_len", -1))
                                accepted_len = int(item.get("accepted_len", 0))
                                if length_only:
                                    # Strict wire form: no prefix or other
                                    # target-side token metadata is accepted.
                                    if prefix is not None:
                                        raise ValueError(
                                            "strict length-only commit must not "
                                            "carry prefix_token_ids"
                                        )
                                    if valid_len < 0:
                                        raise ValueError(
                                            "strict length-only valid_len must be non-negative"
                                        )
                                    target_prefix_len = valid_len + 1
                                    # Draft's verifier contract requires an
                                    # all-accepted length-only row.  The
                                    # runtime eligibility check established
                                    # that property before sending this row.
                                    draft_len = accepted_len
                                    replacement = None
                                    finished = False
                                    gamma = 0
                                else:
                                    if not isinstance(prefix, list) or not prefix:
                                        raise ValueError(
                                            "commit prefix_token_ids must be non-empty"
                                        )
                                    target_prefix_len = int(
                                        item.get(
                                            "target_prefix_len", len(prefix)
                                        )
                                    )
                                    draft_len = int(item.get("draft_len", 0))
                                    replacement = item.get("replacement_token_id")
                                    finished = bool(item.get("finished", False))
                                    gamma = int(item.get("gamma", 0))
                                    if target_prefix_len != len(prefix):
                                        raise ValueError(
                                            "target_prefix_len does not match prefix length"
                                        )
                                    if valid_len < 0 or valid_len > len(prefix) - 1:
                                        raise ValueError(
                                            "commit valid_len must be in [0, prefix_len-1]"
                                        )
                                normalized = {
                                    "request_id": str(
                                        item.get("request_id", f"row-{index}")
                                    ),
                                    "gamma": gamma,
                                    "accepted_len": accepted_len,
                                    "draft_len": draft_len,
                                    "valid_len": valid_len,
                                    "target_prefix_len": target_prefix_len,
                                    "replacement_token_id": (
                                        None if replacement is None else int(replacement)
                                    ),
                                    "finished": finished,
                                    "length_only": length_only,
                                }
                                if not length_only:
                                    normalized["prefix_token_ids"] = [
                                        int(x) for x in prefix
                                    ]
                                updates.append(normalized)
                            result = commit_batch(updates)
                            print(
                                "[draft] commit_batch "
                                f"batch_size={len(updates)} "
                                f"accepted={sum(x['accepted_len'] for x in updates)} "
                                f"valid={sum(x['valid_len'] for x in updates)} "
                                f"length_only={sum(bool(x['length_only']) for x in updates)} "
                                "wire_length_fields=accepted_len,valid_len",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {"status": "result", "results": result or []},
                            )
                            continue

                        if command == "draft_batch":
                            raw_requests = message.get("requests")
                            if not isinstance(raw_requests, list) or not raw_requests:
                                raise ValueError(
                                    "draft_batch requires a non-empty requests list"
                                )
                            requests = []
                            for index, request in enumerate(raw_requests):
                                if not isinstance(request, dict):
                                    raise ValueError(
                                        f"draft request {index} must be an object"
                                    )
                                request_id = str(
                                    request.get("request_id", f"row-{index}")
                                )
                                prefix = request.get("prefix_token_ids")
                                gamma = int(request.get("gamma", 0))
                                if not isinstance(prefix, list) or not prefix:
                                    raise ValueError(
                                        "prefix_token_ids must be non-empty for "
                                        f"{request_id!r}"
                                    )
                                requests.append(
                                    {
                                        "request_id": request_id,
                                        "prefix_token_ids": [int(x) for x in prefix],
                                        "gamma": gamma,
                                    }
                                )

                            print(
                                "[draft] batch proposal "
                                f"batch_size={len(requests)} "
                                f"request_ids={[item['request_id'] for item in requests]} "
                                f"gamma={[item['gamma'] for item in requests]}",
                                flush=True,
                            )
                            results = engine.propose_batch(requests)
                            by_id = {
                                str(item["request_id"]): item
                                for item in results
                            }
                            for request in requests:
                                item = by_id[request["request_id"]]
                                print(
                                    "[draft] proposal "
                                    f"request_id={request['request_id']!r} "
                                    f"prefix_len={len(request['prefix_token_ids'])} "
                                    f"gamma={request['gamma']} "
                                    f"draft_ids={item['draft_token_ids']}",
                                    flush=True,
                                )
                            send_message(conn, {"status": "result", "results": results})
                            continue

                        if command != "draft":
                            raise ValueError(f"unknown proposal command: {command!r}")
                        prefix = message.get("prefix_token_ids")
                        gamma = int(message.get("gamma", 0))
                        request_id = str(message.get("request_id", "target-0"))
                        if not isinstance(prefix, list) or not prefix:
                            raise ValueError("prefix_token_ids must be non-empty")
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
                        send_message(
                            conn,
                            {
                                "status": "result",
                                "draft_token_ids": draft_ids,
                                "prefix_len": len(prefix),
                                "request_id": request_id,
                            },
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        send_message(conn, {"status": "error", "error": repr(exc)})

def _run_draft(args: argparse.Namespace, control_server: socket.socket) -> None:
    if not args.proposal_socket:
        raise ValueError("Draft role requires --proposal-socket")
    proposal_server = bind_server(Path(args.proposal_socket))
    control_conn, _ = control_server.accept()
    stop_event = threading.Event()
    proposal_thread = None
    engine = None
    try:
        send_message(
            control_conn,
            {
                "status": "loading",
                "role": "draft",
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            },
        )
        from pearl_stage5_draft import PersistentDraftEngine

        started = time.perf_counter()
        engine = PersistentDraftEngine(
            args.model, args.max_model_len, args.max_num_seqs
        )
        load_elapsed_ms = (time.perf_counter() - started) * 1000.0
        proposal_thread = threading.Thread(
            target=_draft_proposal_loop,
            args=(proposal_server, engine, stop_event),
            daemon=True,
        )
        proposal_thread.start()
        send_message(
            control_conn,
            {
                "status": "ready",
                "role": "draft",
                "model": args.model,
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "load_elapsed_ms": round(load_elapsed_ms, 3),
                "persistent_kv": True,
                "enforce_eager": True,
            },
        )

        with control_conn.makefile("r", encoding="utf-8") as reader:
            while True:
                message = receive_message(reader)
                if message is None:
                    break
                command = message.get("cmd")
                if command == "ping":
                    send_message(control_conn, {"status": "pong", "role": "draft"})
                elif command == "stop":
                    send_message(control_conn, {"status": "stopped", "role": "draft"})
                    break
                else:
                    send_message(control_conn, {"status": "error", "error": f"unknown command: {command!r}"})
    finally:
        stop_event.set()
        proposal_server.close()
        if proposal_thread is not None:
            proposal_thread.join(timeout=5)
        if engine is not None:
            engine.shutdown()
        control_conn.close()
        try:
            Path(args.proposal_socket).unlink()
        except FileNotFoundError:
            pass


# PEARL_STAGE5_TARGET_SHUTDOWN_COMPAT_V1
def _shutdown_target_engine(llm: Any) -> None:
    """Run Target cleanup without exposing the known V1 LLMEngine mismatch."""
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return

    try:
        shutdown()
        return
    except AttributeError as exc:
        # vLLM-HUST's V1 LLMEngine currently reaches a legacy cleanup helper
        # that references self.model_executor, which this engine does not own.
        # This is a shutdown-only compatibility issue; inference is complete.
        if "model_executor" not in str(exc):
            raise

        engine_core = getattr(engine, "engine_core", None)
        core_shutdown = getattr(engine_core, "shutdown", None)
        if callable(core_shutdown):
            try:
                core_shutdown()
            except Exception as core_exc:
                print(
                    "[target] EngineCore shutdown fallback warning: "
                    f"{core_exc!r}",
                    flush=True,
                )
        print(
            "[target] shutdown compatibility fallback: ignored missing "
            "LLMEngine.model_executor after EngineCore cleanup",
            flush=True,
        )
    except Exception:
        # Preserve the old diagnostic for unrelated cleanup failures.
        traceback.print_exc()


def _run_target(args: argparse.Namespace, control_server: socket.socket) -> None:
    if not args.draft_socket:
        raise ValueError("Target role requires --draft-socket")
    control_conn, _ = control_server.accept()
    llm = None
    try:
        os.environ["PEARL_DRAFT_SOCKET"] = args.draft_socket
        send_message(
            control_conn,
            {
                "status": "loading",
                "role": "target",
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            },
        )
        started = time.perf_counter()
        llm = _load_target(args)
        load_elapsed_ms = (time.perf_counter() - started) * 1000.0
        send_message(
            control_conn,
            {
                "status": "ready",
                "role": "target",
                "model": args.model,
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "load_elapsed_ms": round(load_elapsed_ms, 3),
                "persistent_target_kv": True,
                "enforce_eager": True,
            },
        )
        with control_conn.makefile("r", encoding="utf-8") as reader:
            while True:
                message = receive_message(reader)
                if message is None:
                    break
                command = message.get("cmd")
                if command == "ping":
                    send_message(control_conn, {"status": "pong", "role": "target"})
                elif command == "generate":
                    try:
                        response = _handle_target_generate(
                            llm, message, args.default_max_tokens, args.max_num_seqs
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        response = {"status": "error", "error": repr(exc)}
                    send_message(control_conn, response)
                elif command == "stop":
                    send_message(control_conn, {"status": "stopped", "role": "target"})
                    break
                else:
                    send_message(control_conn, {"status": "error", "error": f"unknown command: {command!r}"})
    finally:
        if llm is not None:
            _shutdown_target_engine(llm)
        control_conn.close()


def main() -> None:
    args = build_parser().parse_args()
    control_server = bind_server(Path(args.socket))
    try:
        if args.role == "draft":
            _run_draft(args, control_server)
        else:
            _run_target(args, control_server)
    finally:
        control_server.close()
        try:
            Path(args.socket).unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
