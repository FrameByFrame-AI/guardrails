#!/usr/bin/env python3
"""Preview dynamic train snippets and deterministic eval snippets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.dataset import CodeLanguageDataset
from code_language_id.snippets import SnippetConfig


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=512)
    args = parser.parse_args()

    parquet_path = args.data_root.expanduser() / "processed/v1_splits" / f"{args.split}.parquet"
    dataset = CodeLanguageDataset(
        parquet_path=parquet_path,
        mode=args.mode,
        snippet_config=SnippetConfig(max_chars=args.max_chars),
    )
    dataset.set_epoch(args.epoch)

    previews = []
    for index in range(min(args.rows, len(dataset))):
        item = dataset[index]
        previews.append(
            {
                "index": index,
                "id": item["id"],
                "label": item["canonical_language"],
                "label_id": item["language_label_id"],
                "snippet_strategy": item["snippet_strategy"],
                "text_len": len(item["text"]),
                "preview": item["text"][:120],
            }
        )

    print(
        json.dumps(
            {
                "parquet_path": str(parquet_path),
                "dataset_rows": len(dataset),
                "mode": args.mode,
                "split": args.split,
                "epoch": args.epoch,
                "previews": previews,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
