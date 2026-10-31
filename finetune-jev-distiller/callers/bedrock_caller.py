"""Screen prompts with an Amazon Bedrock guardrail (ApplyGuardrail, source=INPUT) and cache the decisions.

    python bedrock_caller.py --input tests/fixtures/failure_cases.jsonl --cache data/bedrock_labels.jsonl

Uses the AWS profile in AWS_PROFILE (SSO, no keys) and the guardrail in BEDROCK_GUARDRAIL_ID / _VERSION / _REGION.
block = the guardrail intervened. The cache is append-only and keyed by (text, guardrail id, version).
Cost = content-filter text units x --price-per-1k-units (USD; set it to your region's price, units are exact).
Exits non-zero if any call failed, unless --allow-partial.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

log = logging.getLogger("bedrock")


def call_bedrock(client, gid: str, version: str, text: str, price_per_1k: float) -> dict:
    t0 = time.perf_counter()
    try:
        r = client.apply_guardrail(guardrailIdentifier=gid, guardrailVersion=version, source="INPUT",
                                   content=[{"text": {"text": text}}])
    except (ClientError, BotoCoreError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    latency = (time.perf_counter() - t0) * 1000
    filters = [{"type": f["type"], "confidence": f["confidence"], "action": f["action"]}
               for a in r.get("assessments", []) for f in a.get("contentPolicy", {}).get("filters", [])]
    units = r.get("usage", {}).get("contentPolicyUnits", 0)
    return {"block": r["action"] == "GUARDRAIL_INTERVENED", "filters": filters, "units": units,
            "cost": units * price_per_1k / 1000, "latency_ms": round(latency, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--guardrail-id", default=os.environ.get("BEDROCK_GUARDRAIL_ID"))
    ap.add_argument("--guardrail-version", default=os.environ.get("BEDROCK_GUARDRAIL_VERSION", "1"))
    ap.add_argument("--region", default=os.environ.get("BEDROCK_REGION", "ap-south-1"))
    ap.add_argument("--price-per-1k-units", type=float, default=0.15,
                    help="USD per 1,000 content-filter text units (1 unit = up to 1,000 characters)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.guardrail_id:
        raise SystemExit("set BEDROCK_GUARDRAIL_ID or --guardrail-id")

    client = boto3.Session().client("bedrock-runtime", region_name=args.region,
                                    config=Config(retries={"mode": "adaptive", "max_attempts": 8}, max_pool_connections=16))
    key = lambda text: (text, args.guardrail_id, args.guardrail_version)
    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache):
            c = json.loads(line)
            cache[(c["text"], c["guardrail_id"], c["guardrail_version"])] = c
    texts = list(dict.fromkeys(json.loads(l)["text"] for l in open(args.input)))
    todo = [t for t in texts if key(t) not in cache]
    log.info(f"{len(texts)} prompts, {len(texts) - len(todo)} cached, {len(todo)} to screen "
             f"(guardrail {args.guardrail_id} v{args.guardrail_version}, {args.region})")

    errors = cost = 0
    with open(args.cache, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(call_bedrock, client, args.guardrail_id, args.guardrail_version, t,
                               args.price_per_1k_units): t for t in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="bedrock"):
            res = fut.result()
            if "error" in res:
                errors += 1
                log.warning(res["error"])
                continue
            cost += res["cost"]
            fh.write(json.dumps({"text": futures[fut], "guardrail_id": args.guardrail_id,
                                 "guardrail_version": args.guardrail_version, "region": args.region, **res},
                                ensure_ascii=False) + "\n")
            fh.flush()
    log.info(f"done: {len(todo) - errors} screened, {errors} errors, cost ${cost:.4f} at ${args.price_per_1k_units}/1k units")
    if errors and not args.allow_partial:
        raise SystemExit(f"{errors} prompts failed; results so far are cached. Rerun to retry, or pass --allow-partial.")


if __name__ == "__main__":
    main()
