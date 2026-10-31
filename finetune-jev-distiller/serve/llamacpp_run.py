"""Score an evaluation file with a d3 GGUF served by llama-server (/v1/systemone); host Python, no torch.

    runs/llamacpp/build/bin/llama-server -m runs/gguf/<model>.gguf --port 9932 -t 8 -c 8192 -b 8192 -ub 8192 -np 1
    python3 llamacpp_run.py --url http://127.0.0.1:9932 --input data/aa_guardrail.jsonl --output runs/<dir>/aa_guardrail.jsonl

The same questions as `guardrate.py run --model d3` (block question for requests, RESPONSE_QUESTIONS for
responses), so predictions compare item by item with the PyTorch ones. -ub must hold the whole prompt: d3's
full-attention layers are non-causal, so llama.cpp evaluates a prompt in one batch. Writes one line per item as it
goes and skips items already in --output (resumable). latency_ms is the HTTP round trip of one request.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "callers"))  # host Python: no PYTHONPATH
from jev_caller import QUESTIONS, RESPONSE_QUESTIONS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:9932")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/v1/systemone"
    done = {json.loads(l)["id"] for l in open(args.output)} if os.path.exists(args.output) else set()
    items = [it for it in map(json.loads, open(args.input)) if it["id"] not in done]
    print(f"{len(done)} done, {len(items)} to go", flush=True)
    with open(args.output, "a") as fh:
        for n, it in enumerate(items, 1):
            if it["response"] is None:
                state, qs = it["prompt"], {"block": QUESTIONS["block"]}
            else:
                state, qs = {"user": it["prompt"], "assistant": it["response"]}, RESPONSE_QUESTIONS
            body = json.dumps({"state": state, "questions": qs}).encode()
            t = time.perf_counter()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            (a,) = json.load(urllib.request.urlopen(req, timeout=600))["answers"].values()
            ms = (time.perf_counter() - t) * 1000
            fh.write(json.dumps({"id": it["id"], "p_unsafe": a["noul"], "latency_ms": round(ms, 1)}) + "\n")
            fh.flush()
            if n % 500 == 0:
                print(f"{n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
