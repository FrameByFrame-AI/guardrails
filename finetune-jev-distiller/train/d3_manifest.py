"""Write a d3 package's MODEL_MANIFEST.json (d3-package-manifest/1), which vLLM Semantic Router needs to serve it.

    python d3_manifest.py <package dir> [--name accuknox-guard-1.2] [--repo-id Accuknoxtechnologies/accuknox-guard-1.2]

vllm-srun (`vllm-sr serve -e <package>`, or `vllm-srun serve <package>`) refuses a d3 package without one: the
manifest lists the SHA-256 and size of every file, the parameter counts and the identity of the scored checkpoint,
and the runtime checks the files it loads against it. Training (train_d3.py) and souping (merge_runs.py) write new
weights, so the base model's manifest no longer matches; both call write_manifest at the end. Rerun this after
editing any file in a package (a README edit included), since every file is listed. Standard library only; the
identity and parameter counts follow vllm_srun.families.decision3.package (vllm-srun 0.2.0), whose verify() passes
on the packages this writes.
"""

import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "MODEL_MANIFEST.json"
SCHEMA = "d3-package-manifest/1"
WEIGHT_INDEX = "model.safetensors.index.json"
# decision_config.json fields that change answers; the identity covers them
INFERENCE_FIELDS = ("format_version", "format_id", "prompt", "base_model", "revision", "codes", "token_ids",
                    "temperature", "attention_mode", "pooling", "max_length", "readout_dtype")
IDENTITY_FILES = ("readout.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
RUNTIME_FILES = ("d3_runtime.py", "d3_fast.py", "d3_kernels.py", "modeling_d3.py", "d3_format.py", "d3_engine.py",
                 "d3_server.py", "pipeline_d3.py")
VISION_PREFIXES = ("visual.", "model.visual.")
MAX_OPTIONS = 255
# kept from the base model's manifest: they describe the architecture and licence, not the weights
CARRIED = ("max_input_tokens", "base_model", "licence", "backbone")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(size))


def weight_files(root):
    """The index first, then its shards; or the single model.safetensors."""
    if (root / WEIGHT_INDEX).is_file():
        shards = json.loads((root / WEIGHT_INDEX).read_text())["weight_map"].values()
        return (WEIGHT_INDEX, *sorted(set(shards)))
    return ("model.safetensors",)


def parameter_counts(root, weights):
    counts = {"text": 0, "vision": 0}
    for name in weights:
        if not name.endswith(".safetensors"):
            continue
        for tensor, meta in safetensors_header(root / name).items():
            if tensor == "__metadata__":
                continue
            assert not tensor.endswith("weight_packed"), f"{tensor}: quantized weights are not supported here"
            counts["vision" if tensor.startswith(VISION_PREFIXES) else "text"] += math.prod(meta["shape"])
    return counts


def write_manifest(root, name=None, repo_id=None, base_manifest=None):
    """Write <root>/MODEL_MANIFEST.json for the files now in root; returns the model identity."""
    root = Path(root)
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
    sha = {f: sha256_file(root / f) for f in files}
    decision = json.loads((root / "decision_config.json").read_text())
    weights = weight_files(root)
    identity_files = {f: sha[f] for f in (*[w for w in weights if w.endswith(".safetensors")],
                                         *[f for f in IDENTITY_FILES if f in sha])}
    fields = {k: decision[k] for k in INFERENCE_FIELDS if k in decision}
    model_sha256 = hashlib.sha256(json.dumps({"files": identity_files, "decision": fields}, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    counts = parameter_counts(root, weights)
    hidden = json.loads((root / "config.json").read_text())["text_config"]["hidden_size"]
    readout = MAX_OPTIONS * hidden
    base = json.loads(Path(base_manifest).read_text()) if base_manifest else {}
    manifest = {"schema": SCHEMA, "model_name": name or root.resolve().name, "repo_id": repo_id,
                "format_id": decision["format_id"], "prompt": decision.get("prompt", "d3"),
                "attention_mode": decision.get("attention_mode"),
                **{k: base[k] for k in CARRIED if k in base},
                "identity": {"model_sha256": model_sha256, "files_sha256": identity_files},
                "parameters": {**counts, "readout": readout, "loaded": counts["text"] + counts["vision"] + readout},
                "runtime": {"files_sha256": {f: sha[f] for f in RUNTIME_FILES if f in sha}},
                "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files_sha256": sha, "files_bytes": {f: (root / f).stat().st_size for f in files}}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return model_sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("--name", help="model_name (default: the directory name)")
    ap.add_argument("--repo-id")
    ap.add_argument("--base-manifest", help="the base model's MODEL_MANIFEST.json, for its licence and base fields")
    args = ap.parse_args()
    root = Path(args.package)
    base = args.base_manifest
    if base is None and (root / MANIFEST_NAME).is_file():  # rewriting: keep the fields carried before
        base = root / MANIFEST_NAME
        carried = json.loads(base.read_text())
        args.name = args.name or carried.get("model_name")
        args.repo_id = args.repo_id or carried.get("repo_id")
    print(f"{root}: model identity {write_manifest(root, args.name, args.repo_id, base)}")


if __name__ == "__main__":
    main()
