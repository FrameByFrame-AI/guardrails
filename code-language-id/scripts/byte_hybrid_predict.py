#!/usr/bin/env python3
"""Predict programming language with the lite ByteHybrid checkpoint.

Mirrors PleIAs/CommonLingua's ``predict.py`` but loads ``ByteHybrid`` from this
repo's package. The checkpoint format is identical, so checkpoints produced by
``train_byte_hybrid_language.py`` also load with the upstream script.

    python scripts/byte_hybrid_predict.py "def hello(): pass"
    python scripts/byte_hybrid_predict.py --file snippet.go
    cat snippets.txt | python scripts/byte_hybrid_predict.py --stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.byte_hybrid import CONFIGS, ByteHybrid


def load(checkpoint: str | Path, dtype: str = "fp32", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=ckpt["max_len"],
        **CONFIGS[ckpt["config"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval().to(device_t)
    idx2lang = {v: k for k, v in ckpt["lang2idx"].items()}
    return model, idx2lang, ckpt["max_len"], device_t


def encode(texts: list[str], max_len: int) -> torch.Tensor:
    out = np.full((len(texts), max_len), 256, dtype=np.int64)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        raw = t.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


@torch.no_grad()
def predict(
    model: ByteHybrid,
    texts: list[str],
    idx2lang: dict[int, str],
    max_len: int,
    device: torch.device,
    top_k: int = 3,
    batch_size: int = 256,
) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    for i in range(0, len(texts), batch_size):
        b = encode(texts[i : i + batch_size], max_len).to(device)
        probs = torch.softmax(model(b).float(), dim=-1)
        top_p, top_idx = probs.topk(top_k, dim=-1)
        for p_row, idx_row in zip(top_p.cpu().tolist(), top_idx.cpu().tolist()):
            out.append([(idx2lang[j], float(p)) for p, j in zip(p_row, idx_row)])
    return out


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="*", help="Text(s) to classify (one per arg).")
    p.add_argument("--file", help="Read a single text from FILE.")
    p.add_argument("--stdin", action="store_true", help="One text per line from stdin.")
    p.add_argument(
        "--checkpoint",
        default=str(
            Path(
                "~/llm_models/guardrail_code_models/"
                "programming-language-identification-100plus-lite/model.pt"
            ).expanduser()
        ),
    )
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    if args.stdin:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        texts = [Path(args.file).read_text(encoding="utf-8")]
    elif args.text:
        texts = args.text
    else:
        p.print_help()
        return

    model, idx2lang, max_len, device = load(args.checkpoint, args.dtype, args.device)
    print(
        f"# {len(idx2lang)} languages, max_len={max_len}, dtype={args.dtype}, device={device}",
        file=sys.stderr,
    )
    for text, top in zip(
        texts, predict(model, texts, idx2lang, max_len, device, args.top_k, args.batch_size)
    ):
        preview = text[:80].replace("\n", " ")
        others = " ".join(f"{lg}={p:.3f}" for lg, p in top[1:])
        print(f"{top[0][0]}\t{top[0][1]:.4f}\t{others}\t{preview}")


if __name__ == "__main__":
    _main()
