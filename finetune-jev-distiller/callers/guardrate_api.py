"""Run a hosted guardrail over the GuardRate English rebuild, in guardrate.py's output format, so `guardrate.py
score` ranks it with the local models.

    python guardrate_api.py --system jev --output runs/jev/guardrate_en.jsonl --limit 300   # check first
    python guardrate_api.py --system bedrock --output runs/bedrock/guardrate_en.jsonl

- jev: TypeSafe Jev via OpenRouter's decisions API, on the same questions as the d3 runs (jev_caller.QUESTIONS
  block for prompts, RESPONSE_QUESTIONS on {"user", "assistant"} for responses). p_unsafe = its noul probability.
- bedrock: Amazon Bedrock ApplyGuardrail (AWS SSO profile, no keys; guardrail from BEDROCK_GUARDRAIL_ID /
  _VERSION / _REGION, as bedrock_caller.py). Prompts go in as source=INPUT, responses as source=OUTPUT (the
  response text only: ApplyGuardrail has no conversation input). Bedrock gives a decision, not a probability:
  p_unsafe = 1 if it intervened else 0, and p_graded = the highest filter confidence (none 0, low 1/3, medium 2/3,
  high 1) for ranking (AUC). Cost = text units x --price-per-1k-units.
Rows: id, p_unsafe, latency_ms (one call, network included), cost, and the raw answer. Resumable: rerun to continue;
failed calls are not written and are retried on the next run. Exits non-zero if any call failed.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from resume import open_resumable, write_row

log = logging.getLogger("guardrate-api")
CONFIDENCE = {"NONE": 0.0, "LOW": 1 / 3, "MEDIUM": 2 / 3, "HIGH": 1.0}


def jev_runner(args):
    import httpx

    from jev_caller import MODEL, QUESTIONS, RESPONSE_QUESTIONS, URL

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    client = httpx.Client(headers={"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "jev-distiller"}, timeout=60)

    def run(it):
        state, qs = ((it["prompt"], {"block": QUESTIONS["block"]}) if it["response"] is None
                     else ({"user": it["prompt"], "assistant": it["response"]}, RESPONSE_QUESTIONS))
        t0 = time.perf_counter()
        r = client.post(URL, json={"model": MODEL, "state": state, "questions": qs})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        body = r.json()
        p = body["answers"]["block"]["noul"]
        return {"p_unsafe": round(p, 4), "latency_ms": round(ms, 1), "cost": body.get("usage", {}).get("cost"),
                "model": body.get("model")}
    return run


def bedrock_runner(args):
    import boto3
    from botocore.config import Config

    gid = os.environ.get("BEDROCK_GUARDRAIL_ID")
    version = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "1")
    region = os.environ.get("BEDROCK_REGION", "ap-south-1")
    if not gid:
        raise SystemExit("set BEDROCK_GUARDRAIL_ID")
    client = boto3.Session().client("bedrock-runtime", region_name=region,
                                    config=Config(retries={"mode": "adaptive", "max_attempts": 8},
                                                  max_pool_connections=16))

    def run(it):
        source, text = ("INPUT", it["prompt"]) if it["response"] is None else ("OUTPUT", it["response"])
        t0 = time.perf_counter()
        r = client.apply_guardrail(guardrailIdentifier=gid, guardrailVersion=version, source=source,
                                   content=[{"text": {"text": text}}])
        ms = (time.perf_counter() - t0) * 1000
        filters = [{"type": f["type"], "confidence": f["confidence"], "action": f["action"]}
                   for a in r.get("assessments", []) for f in a.get("contentPolicy", {}).get("filters", [])]
        units = r.get("usage", {}).get("contentPolicyUnits", 0)
        return {"p_unsafe": 1.0 if r["action"] == "GUARDRAIL_INTERVENED" else 0.0,
                "p_graded": round(max([CONFIDENCE.get(f["confidence"], 0.0) for f in filters], default=0.0), 4),
                "latency_ms": round(ms, 1), "cost": units * args.price_per_1k_units / 1000, "units": units,
                "filters": filters, "guardrail": f"{gid}:{version}:{region}"}
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True, choices=["jev", "bedrock"])
    ap.add_argument("--input", default="data/guardrate_en.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="run at most N items in total (a check before the full run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--price-per-1k-units", type=float, default=0.15, help="bedrock: USD per 1,000 text units")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    items = [json.loads(l) for l in open(args.input)]
    if args.limit:  # a fixed, label- and dataset-mixed sample: every k-th item
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done, fh = open_resumable(args.output, "id")
    todo = [it for it in items if it["id"] not in done]
    run = {"jev": jev_runner, "bedrock": bedrock_runner}[args.system](args)
    log.info(f"{args.system}: {len(items)} items, {len(items) - len(todo)} done, {len(todo)} to run")

    def safe(it):
        try:
            return run(it)
        except Exception as e:  # any failure: not written, retried on the next run
            return {"error": f"{type(e).__name__}: {e}"}

    errors, cost = 0, 0.0
    with fh, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(safe, it): it for it in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.system, mininterval=30):
            res = fut.result()
            if "error" in res:
                errors += 1
                if errors <= 20:
                    log.warning(res["error"])
                continue
            cost += res.get("cost") or 0
            write_row(fh, {"id": futures[fut]["id"], **res})
    log.info(f"done: {len(todo) - errors} run, {errors} errors, cost ${cost:.4f}")
    if errors:
        raise SystemExit(f"{errors} items failed; the rest are saved. Rerun to retry them.")


if __name__ == "__main__":
    main()
