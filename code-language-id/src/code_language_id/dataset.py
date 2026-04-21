"""Dataset wrapper for ModernBERT code-language training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from code_language_id.snippets import (
    SnippetConfig,
    make_eval_snippet,
    make_train_snippet,
)


class CodeLanguageDataset:
    """Read split parquet files and create snippet views on demand."""

    def __init__(
        self,
        parquet_path: str | Path,
        mode: str,
        snippet_config: SnippetConfig | None = None,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be either 'train' or 'eval'")

        self.parquet_path = Path(parquet_path)
        self.mode = mode
        self.snippet_config = snippet_config or SnippetConfig()
        self.epoch = 0
        self.frame = pd.read_parquet(self.parquet_path)

        required = {
            "id",
            "canonical_language",
            "language_label_id",
            "code",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(
                f"{self.parquet_path} is missing required columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        code = row["code"]

        if self.mode == "train":
            text, strategy = make_train_snippet(
                code=code,
                index=index,
                epoch=self.epoch,
                config=self.snippet_config,
            )
        else:
            text, strategy = make_eval_snippet(code=code, config=self.snippet_config)

        label_id = int(row["language_label_id"])
        return {
            "id": row["id"],
            "text": text,
            "labels": label_id,
            "language_label_id": label_id,
            "canonical_language": row["canonical_language"],
            "snippet_strategy": strategy,
            "is_code_label": 1,
        }
