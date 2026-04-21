#!/usr/bin/env python3
"""Stream code samples from bigcode/the-stack v1, one parquet per canonical language.

Reads the label map at data/the_stack_label_map.csv, iterates each mapped
directory under data/<stack_dir> on the HF hub, filters to content within
``--min-bytes``/``--max-bytes``, and writes up to ``--target-rows`` rows per
language to ``--output-dir``.

Resumable: an existing output file is skipped.

Requires an HF token with gated-repo access to ``bigcode/the-stack``. Pass via
the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger("fetch_the_stack")


SCHEMA_COLUMNS = [
    "id",
    "source",
    "raw_language",
    "canonical_language",
    "task_name",
    "task_url",
    "language_url",
    "code",
    "code_len",
]


def slugify(name: str) -> str:
    replaced = name.replace("#", "_sharp").replace("+", "_plus")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", replaced).strip("_").lower()
    return slug or "unknown"


def read_label_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("stack_dir")]
    return rows


def stream_language(
    *,
    stack_dir: str,
    canonical: str,
    target_rows: int,
    min_bytes: int,
    max_bytes: int,
    max_candidates: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    from datasets import load_dataset

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{stack_dir}",
        split="train",
        streaming=True,
    )
    rows: list[dict[str, object]] = []
    stats = {
        "seen": 0,
        "kept": 0,
        "too_short": 0,
        "too_long": 0,
        "empty_content": 0,
    }
    for record in ds:
        stats["seen"] += 1
        content = record.get("content") or ""
        if not content:
            stats["empty_content"] += 1
        else:
            nbytes = record.get("size") or len(content.encode("utf-8"))
            if nbytes < min_bytes:
                stats["too_short"] += 1
            elif nbytes > max_bytes:
                stats["too_long"] += 1
            else:
                hexsha = record.get("hexsha") or f"anon-{stats['seen']}"
                repo = record.get("max_stars_repo_name") or ""
                path = record.get("max_stars_repo_path") or ""
                rows.append(
                    {
                        "id": f"thestack-{slugify(canonical)}-{hexsha}",
                        "source": "the-stack-v1",
                        "raw_language": record.get("lang") or stack_dir,
                        "canonical_language": canonical,
                        "task_name": (
                            f"thestack/{slugify(canonical)}/{hexsha[:12]}"
                        ),
                        "task_url": (
                            f"https://github.com/{repo}/blob/-/{path}"
                            if repo
                            else ""
                        ),
                        "language_url": "",
                        "code": content,
                        "code_len": len(content),
                    }
                )
                stats["kept"] += 1
                if stats["kept"] >= target_rows:
                    break
        if stats["seen"] >= max_candidates and stats["kept"] < target_rows:
            break
    return rows, stats


def write_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    import pandas as pd

    if not rows:
        return
    frame = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path(
            "/workspace/data/the_stack_label_map.csv"
        ),
        help="CSV mapping canonical_language -> stack_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/models/guardrail_code_data/raw/the-stack-v1"
        ),
        help="Where to write per-language parquet files.",
    )
    parser.add_argument("--target-rows", type=int, default=500)
    parser.add_argument("--min-bytes", type=int, default=100)
    parser.add_argument("--max-bytes", type=int, default=20_000)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30_000,
        help="Give up after scanning this many rows without hitting target-rows.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional allowlist of canonical languages to process.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Optional denylist of canonical languages to skip.",
    )
    return parser.parse_args()


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        log.error("HF_TOKEN not set; aborting.")
        sys.exit(2)

    args = parse_args()
    label_map_rows = read_label_map(args.label_map)
    if args.only:
        keep = {c.strip() for c in args.only}
        label_map_rows = [r for r in label_map_rows if r["canonical_language"] in keep]
    if args.skip:
        drop = {c.strip() for c in args.skip}
        label_map_rows = [r for r in label_map_rows if r["canonical_language"] not in drop]

    log.info("processing %d mapped languages", len(label_map_rows))

    summary = {
        "start_ts": int(time.time()),
        "target_rows": args.target_rows,
        "min_bytes": args.min_bytes,
        "max_bytes": args.max_bytes,
        "max_candidates": args.max_candidates,
        "languages": {},
    }

    for row in label_map_rows:
        canonical = row["canonical_language"]
        stack_dir = row["stack_dir"]
        slug = slugify(canonical)
        output_path = args.output_dir / f"{slug}.parquet"

        if output_path.exists():
            log.info("[skip] %-32s already at %s", canonical, output_path)
            summary["languages"][canonical] = {
                "stack_dir": stack_dir,
                "status": "skipped_existing",
            }
            continue

        started = time.time()
        try:
            rows, stats = stream_language(
                stack_dir=stack_dir,
                canonical=canonical,
                target_rows=args.target_rows,
                min_bytes=args.min_bytes,
                max_bytes=args.max_bytes,
                max_candidates=args.max_candidates,
            )
        except Exception as exc:
            log.warning("[fail] %-32s stack_dir=%s %s", canonical, stack_dir, exc)
            summary["languages"][canonical] = {
                "stack_dir": stack_dir,
                "status": "failed",
                "error": repr(exc)[:400],
            }
            continue

        elapsed = time.time() - started
        write_parquet(rows, output_path)
        summary["languages"][canonical] = {
            "stack_dir": stack_dir,
            "status": "ok" if rows else "empty",
            "kept": stats["kept"],
            "seen": stats["seen"],
            "too_short": stats["too_short"],
            "too_long": stats["too_long"],
            "elapsed_s": round(elapsed, 1),
        }
        log.info(
            "[ok]   %-32s kept=%4d seen=%6d elapsed=%5.1fs path=%s",
            canonical,
            stats["kept"],
            stats["seen"],
            elapsed,
            output_path,
        )

    summary["end_ts"] = int(time.time())
    summary_path = args.output_dir / "_ingest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    log.info("wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
