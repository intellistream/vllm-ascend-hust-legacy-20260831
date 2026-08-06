#!/usr/bin/env python3
"""Resident Draft/Target worker for the token-level nano-PEARL prototype.

The worker owns one vLLM LLM instance and communicates with the coordinator
over a Unix domain socket.  The socket is deliberately independent from
stdout/stderr because vLLM emits logs while it starts its EngineCore.

The worker accepts both text prompts and token-ID prompts.  The ``draft``
and ``target_probe`` commands are token-level plumbing only; they do not
implement target logits verification or KV-cache rollback yet.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any


def send_message(conn: socket.socket, message: dict[str, Any]) -> None:
    payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(payload)


def receive_message(reader) -> dict[str, Any] | None:
    line = reader.readline()
    if not line:
        return None
    return json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("draft", "target"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--default-max-tokens", type=int, default=8)
    return parser


def load_vllm_model(args: argparse.Namespace):
    # Import vLLM only inside main().  vLLM V1 starts an EngineCore with
    # multiprocessing=spawn, so importing/constructing it at module scope
    # causes the safe-importing-main RuntimeError.
    from vllm import LLM

    return LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        trust_remote_code=True,
    )


def handle_generate(
    llm,
    message: dict[str, Any],
    default_max_tokens: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompt = message.get("prompt")
    prompt_token_ids = message.get("prompt_token_ids")

    if prompt_token_ids is not None:
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            return {
                "status": "error",
                "error": "prompt_token_ids must be a non-empty list",
            }
        try:
            input_token_ids = [int(token_id) for token_id in prompt_token_ids]
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "error": f"invalid prompt_token_ids: {exc!r}",
            }
        prompts = [{"prompt_token_ids": input_token_ids}]
    elif isinstance(prompt, str) and prompt:
        input_token_ids = None
        prompts = [prompt]
    else:
        return {
            "status": "error",
            "error": "request requires prompt or prompt_token_ids",
        }

    max_tokens = int(message.get("max_tokens", default_max_tokens))
    if max_tokens <= 0:
        return {"status": "error", "error": "max_tokens must be positive"}

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    request_output = outputs[0].outputs[0]
    token_ids = getattr(request_output, "token_ids", ())

    return {
        "status": "result",
        "text": request_output.text,
        "token_ids": [int(token_id) for token_id in token_ids],
        "num_tokens": len(token_ids),
        "elapsed_ms": round(elapsed_ms, 3),
        "input_token_ids": input_token_ids,
        "input_len": len(input_token_ids) if input_token_ids is not None else None,
    }


def _prompt_argmax(prompt_logprobs, position: int) -> tuple[int, float]:
    """Return the best token ID visible in one prompt-logprob entry."""

    if prompt_logprobs is None or position >= len(prompt_logprobs):
        raise RuntimeError(
            f"prompt_logprobs does not contain position {position}"
        )

    position_logprobs = prompt_logprobs[position]
    if not position_logprobs:
        raise RuntimeError(
            f"prompt_logprobs[{position}] is empty; cannot verify token"
        )

    best_token_id, best_logprob = max(
        position_logprobs.items(),
        key=lambda item: float(getattr(item[1], "logprob", item[1])),
    )
    return int(best_token_id), float(
        getattr(best_logprob, "logprob", best_logprob)
    )


def handle_verify_greedy(llm, message: dict[str, Any]) -> dict[str, Any]:
    """Verify draft IDs with prompt logprobs for a greedy prototype.

    For a candidate sequence ``prefix + draft``:

    - prompt_logprobs at position ``prefix_len + i`` describes Target's
      distribution for draft position ``i``;
    - the first mismatch supplies the replacement Target token;
    - if all draft tokens match, the one generated output token is the bonus
      token.

    This performs a fresh vLLM request and does not reuse/rollback KV cache.
    It is therefore a correctness prototype, not yet a performance path.
    """

    from vllm import SamplingParams

    candidate_ids = message.get("prompt_token_ids")
    draft_ids = message.get("draft_token_ids")
    prefix_len = message.get("prefix_len")

    if not isinstance(candidate_ids, list) or not candidate_ids:
        return {"status": "error", "error": "missing prompt_token_ids"}
    if not isinstance(draft_ids, list) or not draft_ids:
        return {"status": "error", "error": "missing draft_token_ids"}
    if not isinstance(prefix_len, int) or prefix_len < 1:
        return {"status": "error", "error": "invalid prefix_len"}

    candidate_ids = [int(token_id) for token_id in candidate_ids]
    draft_ids = [int(token_id) for token_id in draft_ids]
    if len(candidate_ids) != prefix_len + len(draft_ids):
        return {
            "status": "error",
            "error": (
                "candidate length mismatch: "
                f"candidate={len(candidate_ids)}, "
                f"prefix={prefix_len}, draft={len(draft_ids)}"
            ),
        }

    # With prompt_logprobs=1, vLLM returns the selected prompt token and the
    # most likely alternatives.  The greedy argmax is therefore available
    # without transferring the whole vocabulary for every prompt position.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": candidate_ids}],
        sampling_params,
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    request_output = outputs[0]
    completion = request_output.outputs[0]
    prompt_logprobs = getattr(request_output, "prompt_logprobs", None)

    target_token_ids = []
    target_token_logprobs = []
    accepted_len = 0
    mismatch_target_token_id = None

    for draft_index, draft_token_id in enumerate(draft_ids):
        position = prefix_len + draft_index
        target_token_id, target_logprob = _prompt_argmax(
            prompt_logprobs,
            position,
        )
        target_token_ids.append(target_token_id)
        target_token_logprobs.append(target_logprob)

        if target_token_id != draft_token_id:
            mismatch_target_token_id = target_token_id
            break
        accepted_len += 1

    if accepted_len < len(draft_ids):
        bonus_token_id = mismatch_target_token_id
    else:
        generated_token_ids = [int(token_id) for token_id in completion.token_ids]
        bonus_token_id = generated_token_ids[0] if generated_token_ids else None

    committed_token_ids = draft_ids[:accepted_len]
    if bonus_token_id is not None:
        committed_token_ids.append(int(bonus_token_id))

    return {
        "status": "verify_result",
        "accepted_len": accepted_len,
        "draft_len": len(draft_ids),
        "target_token_ids": target_token_ids,
        "target_token_logprobs": target_token_logprobs,
        "bonus_token_id": bonus_token_id,
        "committed_token_ids": committed_token_ids,
        "completion_token_ids": [int(token_id) for token_id in completion.token_ids],
        "elapsed_ms": round(elapsed_ms, 3),
        "input_token_ids": candidate_ids,
        "input_len": len(candidate_ids),
    }


def main() -> None:
    args = build_parser().parse_args()
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    if socket_path.exists():
        socket_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    print(
        f"[{args.role}] socket ready: {socket_path}; "
        f"visible_device={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}",
        flush=True,
    )

    conn = None
    llm = None
    try:
        conn, _ = server.accept()
        send_message(
            conn,
            {
                "status": "loading",
                "role": args.role,
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            },
        )

        load_start = time.perf_counter()
        llm = load_vllm_model(args)
        load_elapsed_ms = (time.perf_counter() - load_start) * 1000.0

        send_message(
            conn,
            {
                "status": "ready",
                "role": args.role,
                "model": args.model,
                "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "load_elapsed_ms": round(load_elapsed_ms, 3),
                "enforce_eager": True,
            },
        )

        with conn.makefile("r", encoding="utf-8") as reader:
            while True:
                message = receive_message(reader)
                if message is None:
                    break

                command = message.get("cmd")
                if command == "ping":
                    send_message(conn, {"status": "pong", "role": args.role})
                elif command in ("generate", "draft", "target_probe"):
                    try:
                        response = handle_generate(
                            llm,
                            message,
                            args.default_max_tokens,
                        )
                    except Exception as exc:  # pragma: no cover - runtime path
                        traceback.print_exc()
                        response = {
                            "status": "error",
                            "error": repr(exc),
                        }
                    response["role"] = args.role
                    response["phase"] = command
                    if response.get("status") == "result":
                        if command == "draft":
                            response["draft_token_ids"] = response["token_ids"]
                        elif command == "target_probe":
                            response["target_probe_token_ids"] = response[
                                "token_ids"
                            ]
                    send_message(conn, response)
                elif command == "verify_greedy":
                    try:
                        response = handle_verify_greedy(llm, message)
                    except Exception as exc:  # pragma: no cover - runtime path
                        traceback.print_exc()
                        response = {
                            "status": "error",
                            "error": repr(exc),
                        }
                    response["role"] = args.role
                    response["phase"] = command
                    send_message(conn, response)
                elif command == "stop":
                    send_message(conn, {"status": "stopped", "role": args.role})
                    break
                else:
                    send_message(
                        conn,
                        {
                            "status": "error",
                            "role": args.role,
                            "error": f"unknown command: {command!r}",
                        },
                    )
    except Exception as exc:  # pragma: no cover - runtime path
        traceback.print_exc()
        if conn is not None:
            try:
                send_message(
                    conn,
                    {
                        "status": "error",
                        "role": args.role,
                        "error": repr(exc),
                    },
                )
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            conn.close()
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
