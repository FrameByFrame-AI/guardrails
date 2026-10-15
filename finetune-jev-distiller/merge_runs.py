"""Average the weights of d3 packages fine-tuned from the same base (a model soup), as one package.

    python merge_runs.py --run runs/d3-ft/package --run runs/d3-v9/package --out runs/d3-soup59/package

Runs 5 and 9 make different mistakes: averaging their probabilities per item passes run 9's decision rule
(experiments.md, "Run 9 results"). Both are LoRA fine-tunes of d3-mini merged into the base, so averaging the
weights averages the two LoRA updates and gives one model at one model's latency. Every tensor in the model shards
and readout.safetensors is averaged in float32 (--weight per run, default equal) and cast back to its dtype;
tensors the runs left at the base are identical in each, so they are unchanged. Other files come from the first run.
"""

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="package directory; give two or more")
    ap.add_argument("--weight", action="append", type=float, help="per --run, default equal; normalised to sum 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [Path(r) for r in args.run]
    w = args.weight or [1.0] * len(runs)
    assert len(runs) >= 2 and len(w) == len(runs)
    w = [x / sum(w) for x in w]
    out = Path(args.out)
    if (out / "merge.json").exists():
        print(f"{out} already merged")
        return
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(p.name for p in runs[0].glob("*.safetensors"))
    for r in runs[1:]:
        assert sorted(p.name for p in r.glob("*.safetensors")) == shards, f"{r}: different shard files"
    for f in runs[0].iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)

    changed = 0
    for name in shards:
        with ExitStack() as stack:
            handles = [stack.enter_context(safe_open(r / name, "pt")) for r in runs]
            keys = sorted(handles[0].keys())
            for h, r in zip(handles[1:], runs[1:]):
                assert sorted(h.keys()) == keys, f"{r / name}: different tensors"
            merged = {}
            for k in keys:
                ts = [h.get_tensor(k) for h in handles]
                if all(torch.equal(ts[0], t) for t in ts[1:]):
                    merged[k] = ts[0]
                    continue
                merged[k] = sum(wi * t.float() for wi, t in zip(w, ts)).to(ts[0].dtype)
                changed += 1
            metadata = handles[0].metadata()
        save_file(merged, str(out / name), metadata=metadata)
        (out / name).chmod(0o644)  # safetensors saves through a temporary file, which leaves it 0600
        print(f"{name}: {len(keys)} tensors")
    json.dump({"runs": [str(r) for r in runs], "weights": w, "tensors_averaged": changed},
              open(out / "merge.json", "w"), indent=1)
    print(f"{changed} tensors differ between runs and were averaged -> {out}")


if __name__ == "__main__":
    main()
