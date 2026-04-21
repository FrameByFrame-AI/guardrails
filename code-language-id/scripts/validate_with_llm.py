#!/usr/bin/env python3
"""Validate code-language labels against a vLLM server's zero-shot classification.

For each row in the provided parquet splits, send the code to the LLM, read a
single programming language name back, and record whether the prediction
matches the canonical_language label.

Output is JSONL so the run is resumable — every completed row is flushed and we
skip rows whose ``id`` already appears in the output file. Uses asyncio with a
configurable concurrency level.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger("validate_with_llm")


SYSTEM_PROMPT = (
    "You identify the programming language of short code snippets. "
    "Reply with ONLY the canonical language name on a single line, with no "
    "explanation, punctuation, or extra words. Use the most widely accepted "
    "name (for example 'Python', 'C++', 'C#', 'JavaScript', 'Mathematica/Wolfram Language')."
)


def normalize_name(value: str) -> str:
    value = (value or "").strip()
    value = value.splitlines()[0] if value else ""
    value = value.strip(" \t`\"'.,;:")
    return value


ALIASES: dict[str, str] = {
    # LHS is a raw LLM reply (lowercased alnum), RHS is the canonical label it maps to.
    "arm": "ARM Assembly",
    "armassembly": "ARM Assembly",
    "armasm": "ARM Assembly",
    "aarch64": "ARM Assembly",
    "aarch64assembly": "ARM Assembly",
    "wolfram": "Mathematica/Wolfram Language",
    "wolframlanguage": "Mathematica/Wolfram Language",
    "mathematica": "Mathematica/Wolfram Language",
    "mathematicawolframlanguage": "Mathematica/Wolfram Language",
    "js": "JavaScript",
    "ts": "TypeScript",
    "objc": "Objective-C",
    "objectivec": "Objective-C",
    "cplusplus": "C++",
    "cpp": "C++",
    "csharp": "C#",
    "cs": "C#",
    "fsharp": "F#",
    "fs": "F#",
    "bash": "Shell",
    "sh": "Shell",
    "shellscript": "Shell",
    "batchscript": "Batchfile",
    "batch": "Batchfile",
    "emacslisp": "Emacs Lisp",
    "elisp": "Emacs Lisp",
    "commonlisp": "Common Lisp",
    "cl": "Common Lisp",
    "standardml": "Standard ML",
    "sml": "Standard ML",
    "vbnet": "Visual Basic .NET",
    "visualbasicnet": "Visual Basic .NET",
    "visualbasic": "Visual Basic .NET",
    "matlaboctave": "MATLAB",
    "octave": "MATLAB",
    "raku": "Raku",
    "perl6": "Raku",
    "nimrod": "Nim",
    "componentpascal": "Component Pascal",
    "modula2": "Modula-2",
    "modula3": "Modula-3",
    "powershellscript": "PowerShell",
    "ps1": "PowerShell",
    "rscript": "R",
}


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def resolve_alias(predicted: str, canonical: str) -> bool:
    pred_key = canonical_key(predicted)
    if pred_key == canonical_key(canonical):
        return True
    mapped = ALIASES.get(pred_key)
    if mapped and canonical_key(mapped) == canonical_key(canonical):
        return True
    return False


def load_rows(splits_dir: Path, splits: list[str]) -> pd.DataFrame:
    frames = []
    for split in splits:
        path = splits_dir / f"{split}.parquet"
        frame = pd.read_parquet(path, columns=["id", "canonical_language", "code", "source"])
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def already_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = record.get("id")
            if isinstance(row_id, str):
                done.add(row_id)
    return done


async def validate_row(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    model: str,
    snippet: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, object]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Code:\n\n{snippet}"},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = await client.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {}) or {}
    predicted = normalize_name(raw)
    return predicted, {
        "raw_response": raw,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def worker(
    row_queue: asyncio.Queue,
    done_queue: asyncio.Queue,
    client: httpx.AsyncClient,
    *,
    server_url: str,
    model: str,
    snippet_max_chars: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_delay: float,
) -> None:
    while True:
        item = await row_queue.get()
        if item is None:
            row_queue.task_done()
            return
        row_id, canonical, code, source, split = item
        snippet = (code or "")[:snippet_max_chars]
        result = {
            "id": row_id,
            "split": split,
            "source": source,
            "canonical_language": canonical,
            "snippet_len": len(snippet),
        }
        for attempt in range(retries + 1):
            try:
                predicted, meta = await validate_row(
                    client,
                    server_url=server_url,
                    model=model,
                    snippet=snippet,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result["predicted_language"] = predicted
                result["match"] = resolve_alias(predicted, canonical)
                result["prompt_tokens"] = meta.get("prompt_tokens")
                result["completion_tokens"] = meta.get("completion_tokens")
                result["raw_response"] = meta.get("raw_response")
                break
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                if attempt >= retries:
                    result["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    break
                await asyncio.sleep(retry_delay * (2**attempt))
        await done_queue.put(result)
        row_queue.task_done()


async def writer(
    done_queue: asyncio.Queue,
    output_path: Path,
    *,
    total: int,
    flush_every: int,
) -> None:
    written = 0
    matches = 0
    mismatches = 0
    errors = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("a", encoding="utf-8")
    last_log = time.time()
    try:
        while True:
            item = await done_queue.get()
            if item is None:
                done_queue.task_done()
                break
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            if item.get("error"):
                errors += 1
            elif item.get("match"):
                matches += 1
            else:
                mismatches += 1
            if written % flush_every == 0:
                handle.flush()
            now = time.time()
            if now - last_log >= 10 or written == total:
                log.info(
                    "progress: %d/%d matches=%d mismatches=%d errors=%d",
                    written,
                    total,
                    matches,
                    mismatches,
                    errors,
                )
                last_log = now
            done_queue.task_done()
    finally:
        handle.flush()
        handle.close()


async def run(args: argparse.Namespace) -> None:
    frame = load_rows(args.splits_dir, args.splits)
    log.info("loaded %d rows from splits %s", len(frame), args.splits)

    done = already_done_ids(args.output)
    if done:
        log.info("resuming: %d rows already recorded", len(done))

    pending = frame[~frame["id"].isin(done)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    log.info("will process %d rows", len(pending))

    if len(pending) == 0:
        log.info("nothing to do")
        return

    row_queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    done_queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 8)

    timeout = httpx.Timeout(
        connect=30.0,
        read=args.request_timeout,
        write=30.0,
        pool=30.0,
    )
    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency * 2,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
        workers = [
            asyncio.create_task(
                worker(
                    row_queue,
                    done_queue,
                    client,
                    server_url=args.server_url.rstrip("/"),
                    model=args.model,
                    snippet_max_chars=args.snippet_max_chars,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                )
            )
            for _ in range(args.concurrency)
        ]
        writer_task = asyncio.create_task(
            writer(
                done_queue,
                args.output,
                total=len(pending),
                flush_every=args.flush_every,
            )
        )

        started = time.time()
        for row in pending.itertuples(index=False):
            await row_queue.put((row.id, row.canonical_language, row.code, row.source, row.split))
        for _ in workers:
            await row_queue.put(None)

        await row_queue.join()
        for w in workers:
            await w

        await done_queue.put(None)
        await writer_task

        log.info("done in %.1fs", time.time() - started)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(
            "~/llm_models/guardrail_code_data/processed/v1_splits"
        ).expanduser(),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8766",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            Path(__file__).resolve().parents[1]
            / "data/validation/label_validation.jsonl"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--snippet-max-chars", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("interrupted; output is partial at %s", args.output)
        sys.exit(130)


if __name__ == "__main__":
    main()
