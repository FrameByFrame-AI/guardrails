"""Run a GLiNER guardrail model over a prompt file, one prompt at a time, as a deployed guardrail would.

    python gliner_caller.py --model runs/gold-gliner --input data/benchmark.jsonl --output runs/gold-gliner/benchmark.jsonl

--model is a run directory (uses checkpoints/best) or any gliner2 model path (e.g. the untrained base).
Inputs are truncated to the run's training max_len (train-summary.json), or --max-len for other models.
Each output row has the prompt, block (block_prob >= --threshold), block_prob, risk and measured latency.
Cost per prompt is GPU time x --gpu-usd-per-hour: an assumed rental price, not a measured bill.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from resume import open_resumable, write_row
from to_gliner import TASKS

log = logging.getLogger("gliner")


def resolve(path: str) -> str:
    for sub in ("checkpoints/best", "checkpoints/final"):
        if (Path(path) / sub).exists():
            return str(Path(path) / sub)
    return path


def run_max_len(model_arg: str):
    """max_len from the run's train-summary.json; model_arg may be the run dir or runs/<run>/checkpoints/<name>."""
    p = Path(model_arg).resolve()
    for d in (p, *list(p.parents)[:2]):
        if (d / "train-summary.json").exists():
            return json.loads((d / "train-summary.json").read_text()).get("max_len")
    return None


def block_prob(res: dict) -> float:
    """P(block) from gliner2's {label, confidence} for the two-label block task."""
    b = res["block"]
    return b["confidence"] if b["label"] == "block" else 1.0 - b["confidence"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-len", type=int, default=None, help="default: the run's training max_len, else 1024")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=1.0, help="assumed GPU price for the cost column")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    from gliner2 import AutoExtractor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve(args.model)
    max_len = args.max_len or run_max_len(args.model) or 1024
    texts = list(dict.fromkeys(json.loads(l)["text"] for l in open(args.input)))
    model = AutoExtractor.from_pretrained(path).to(device).eval()
    done, fh = open_resumable(args.output, "text")  # rerun after a crash continues where it stopped
    todo = [t for t in texts if t not in done]
    log.info(f"{path} on {device}, max_len {max_len}: {len(texts)} prompts, {len(done)} already done")

    with torch.inference_mode(), fh:
        for t in todo[:5]:  # warm-up
            model.classify_text(t, TASKS, include_confidence=True, max_len=max_len)
        for t in todo:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model.classify_text(t, TASKS, include_confidence=True, max_len=max_len)
            if device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            p = block_prob(res)
            write_row(fh, {"text": t, "model": path, "block": p >= args.threshold, "block_prob": round(p, 4),
                           "risk": res["risk"]["label"], "latency_ms": round(ms, 1),
                           "cost": ms / 3.6e6 * args.gpu_usd_per_hour})
    log.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
