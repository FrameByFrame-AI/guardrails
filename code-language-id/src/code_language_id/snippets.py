"""Training-time code snippet views.

These helpers intentionally return in-memory views only. They do not write
augmented examples back to disk.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SnippetConfig:
    max_chars: int = 512
    min_chars: int = 64
    short_chars: int = 128
    train_strategies: tuple[str, ...] = ("variable_window",)
    eval_strategy: str = "head"
    seed: int = 20260420


def normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n")


def stable_rng(seed: int, epoch: int, index: int) -> random.Random:
    payload = f"{seed}:{epoch}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def choose_train_strategy(config: SnippetConfig, rng: random.Random) -> str:
    return rng.choice(config.train_strategies)


def make_snippet(
    code: str,
    strategy: str,
    config: SnippetConfig,
    rng: random.Random | None = None,
) -> str:
    code = normalize_code(code)
    max_chars = config.max_chars
    min_chars = max(1, min(config.min_chars, max_chars))
    short_chars = min(config.short_chars, max_chars)

    if strategy == "variable_window":
        if len(code) <= min_chars:
            return code
        rng = rng or random.Random()
        upper = min(max_chars, len(code))
        lower = min(min_chars, upper)
        target = rng.randint(lower, upper)
        if target >= len(code):
            return code
        start = rng.randint(0, len(code) - target)
        return code[start : start + target]

    if strategy in {"full_truncate", "head"}:
        if len(code) <= max_chars:
            return code
        return code[:max_chars]

    if strategy == "tail":
        if len(code) <= max_chars:
            return code
        return code[-max_chars:]

    if strategy == "random_window":
        if len(code) <= max_chars:
            return code
        rng = rng or random.Random()
        start = rng.randint(0, len(code) - max_chars)
        return code[start : start + max_chars]

    if strategy == "short_window":
        if len(code) <= short_chars:
            return code
        rng = rng or random.Random()
        window = min(short_chars, len(code))
        start = rng.randint(0, len(code) - window)
        return code[start : start + window]

    if strategy == "head_tail":
        if len(code) <= max_chars:
            return code
        head_len = max_chars // 2
        tail_len = max_chars - head_len
        return code[:head_len] + code[-tail_len:]

    raise ValueError(f"Unknown snippet strategy: {strategy}")


def make_train_snippet(
    code: str,
    index: int,
    epoch: int,
    config: SnippetConfig,
) -> tuple[str, str]:
    rng = stable_rng(config.seed, epoch, index)
    strategy = choose_train_strategy(config, rng)
    return make_snippet(code, strategy, config, rng), strategy


def make_eval_snippet(code: str, config: SnippetConfig) -> tuple[str, str]:
    strategy = config.eval_strategy
    return make_snippet(code, strategy, config), strategy
