"""Score an evaluation file through a System One server (vllm-srun or llama-server) with C concurrent clients;
report latency, throughput and agreement with reference predictions. Host Python, no torch.
    python3 srun_bench.py URL INPUT REF_PREDS CONCURRENCY OUT     (server: scripts/serve_srun.sh)
The first 10 items are sent once as warm-up. With CONCURRENCY above 1 vllm-srun batches requests, which pads rows
and can move scores slightly ("approximate numerics"); one at a time it matches the in-process runtime."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "callers"))  # host Python: no PYTHONPATH
from jev_caller import QUESTIONS, RESPONSE_QUESTIONS

url, inp, ref, conc, out = sys.argv[1].rstrip("/") + "/v1/systemone", sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
items = [json.loads(l) for l in open(inp)]
refp = {r["id"]: r["p_unsafe"] for r in map(json.loads, open(ref))}

def ask(it):
    if it["response"] is None:
        state, qs = it["prompt"], {"block": QUESTIONS["block"]}
    else:
        state, qs = {"user": it["prompt"], "assistant": it["response"]}, RESPONSE_QUESTIONS
    body = json.dumps({"state": state, "questions": qs}).encode()
    t = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    (a,) = json.load(urllib.request.urlopen(req, timeout=600))["answers"].values()
    return {"id": it["id"], "p_unsafe": a["noul"], "latency_ms": (time.perf_counter() - t) * 1000}

for it in items[:10]:
    ask(it)  # warm-up
t0 = time.perf_counter()
with ThreadPoolExecutor(conc) as ex:
    res = list(ex.map(ask, items))
wall = time.perf_counter() - t0
with open(out, "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in res)
lat = sorted(r["latency_ms"] for r in res)
d = [abs(r["p_unsafe"] - refp[r["id"]]) for r in res]
flips = sum((r["p_unsafe"] >= 0.5) != (refp[r["id"]] >= 0.5) for r in res)
print(f"c={conc} n={len(res)} p50 {lat[len(lat)//2]:.1f} ms p95 {lat[int(.95*len(lat))]:.1f} ms "
      f"throughput {len(res)/wall:.1f} req/s | vs ref: flips {flips}, max |dp| {max(d):.4f}, mean |dp| {sum(d)/len(d):.5f}")
