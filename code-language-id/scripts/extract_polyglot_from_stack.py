#!/usr/bin/env python3
"""Extract single- and multi-language training rows from the-stack-v1 dumps.

Reads the per-canonical-language parquets produced by ``fetch_the_stack_samples.py``
(``shell.parquet``, ``dockerfile.parquet``, ``makefile.parquet``, ``yaml.parquet``,
``sql.parquet``, ``hcl.parquet``) and emits rows in the AccuKnox CodeLanguage
schema:

    prompt   str           (file contents)
    target   str (JSON)    {"is_valid": true, "category": {"<Lang>": true, ...}}
    kind     str           "single" | "multi"

The "multi" label fires when a host file *naturally* embeds shell commands —
Dockerfiles with RUN lines, docker-compose / GitHub-Actions YAML with `run:`
blocks, Makefile recipes, Terraform `local-exec` provisioners. These are the
real polyglot patterns the model misses; we don't need synthetic concatenation.

Run inside the container that mounts ``~/llm_models`` (any service in
``docker-compose.yml`` works — there's no GPU/HF dependency).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SHELL_TOKENS = (
    "apt-get ", "apt ", "yum ", "dnf ", "apk ", "pip ", "pip3 ", "npm ",
    "echo ", "curl ", "wget ", "tar ", "git ", "chmod ", "chown ", "rm ",
    "mkdir ", "cp ", "mv ", "ln ", "find ", "sed ", "awk ", "grep ",
    "export ", "source ", "bash ", "sh -", "&&", "|| ", "$(", "${",
)


def looks_like_shell(s: str) -> bool:
    """Cheap heuristic: does this text smell like a shell snippet?"""
    if len(s) < 6:
        return False
    s_low = s.lower()
    return any(tok in s_low for tok in SHELL_TOKENS) or s.startswith("#!/")


# ---------- per-source extractors ----------


def from_shell(content: str) -> tuple[set[str], str] | None:
    """Plain shell file -> single Bash."""
    if len(content) < 40:
        return None
    return ({"Bash"}, "single")


def from_dockerfile(content: str) -> tuple[set[str], str] | None:
    """Dockerfile with executable RUN/CMD/ENTRYPOINT lines -> Dockerfile + Bash."""
    if len(content) < 40:
        return None
    lines = content.splitlines()
    run_lines = []
    in_continuation = False
    current_run: list[str] = []
    for line in lines:
        if in_continuation:
            current_run.append(line)
            if not line.rstrip().endswith("\\"):
                run_lines.append(" ".join(current_run))
                current_run = []
                in_continuation = False
            continue
        upper = line.lstrip().upper()
        if upper.startswith(("RUN ", "CMD ", "ENTRYPOINT ", "SHELL ")):
            if line.rstrip().endswith("\\"):
                current_run.append(line)
                in_continuation = True
            else:
                run_lines.append(line)
    has_shell = any(looks_like_shell(rl) for rl in run_lines)
    if len(run_lines) >= 2 and has_shell:
        return ({"Dockerfile", "Bash"}, "multi")
    return ({"Dockerfile"}, "single")


def from_makefile(content: str) -> tuple[set[str], str] | None:
    """Makefile with multiple TAB-indented recipe lines -> Makefile + Bash."""
    if len(content) < 40:
        return None
    lines = content.splitlines()
    recipe_lines = [ln for ln in lines if ln.startswith("\t") and ln.strip() and not ln.lstrip().startswith("#")]
    has_shell = any(looks_like_shell(rl) for rl in recipe_lines)
    if len(recipe_lines) >= 3 and has_shell:
        return ({"Makefile", "Bash"}, "multi")
    return ({"Makefile"}, "single")


YAML_RUN_KEY = re.compile(r"^\s*(run|command|entrypoint|script|shell)\s*:", re.MULTILINE)
YAML_RUN_BLOCK = re.compile(
    r"^\s*(run|command|entrypoint|script|shell)\s*:\s*[|>]?[+\-]?\s*\n((?:^\s+.+\n?)+)",
    re.MULTILINE,
)


def from_yaml(content: str) -> tuple[set[str], str] | None:
    """YAML with explicit shell blocks -> YAML + Bash."""
    if len(content) < 40:
        return None
    blocks: list[str] = []
    for m in YAML_RUN_BLOCK.finditer(content):
        blocks.append(m.group(2))
    # Also catch inline single-line runs (e.g. `command: echo hi && ls`)
    inline_runs: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^\s*(run|command|entrypoint|script|shell)\s*:\s*(.+)$", line)
        if m and m.group(2).strip() not in ("|", ">", "|-", ">-", ""):
            inline_runs.append(m.group(2))
    blob = "\n".join(blocks + inline_runs)
    if blob and looks_like_shell(blob):
        return ({"YAML", "Bash"}, "multi")
    return ({"YAML"}, "single")


HCL_LOCAL_EXEC = re.compile(
    r'provisioner\s+"(local|remote)-exec"\s*\{[^}]*command(?:s)?\s*=\s*([^\n]+)',
    re.DOTALL,
)


def from_hcl(content: str) -> tuple[set[str], str] | None:
    """Terraform/HCL with local-exec/remote-exec provisioners -> Terraform + Bash."""
    if len(content) < 40:
        return None
    if HCL_LOCAL_EXEC.search(content):
        # Verify the command text smells like shell
        if any(looks_like_shell(m.group(2)) for m in HCL_LOCAL_EXEC.finditer(content)):
            return ({"Terraform", "Bash"}, "multi")
    return ({"Terraform"}, "single")


def from_sql(content: str) -> tuple[set[str], str] | None:
    """SQL files are basically always pure SQL."""
    if len(content) < 40:
        return None
    return ({"SQL"}, "single")


SOURCE_EXTRACTORS = {
    "Bash": ("bash.parquet", from_shell),
    "Dockerfile": ("dockerfile.parquet", from_dockerfile),
    "Makefile": ("makefile.parquet", from_makefile),
    "YAML": ("yaml.parquet", from_yaml),
    "SQL": ("sql.parquet", from_sql),
    "Terraform": ("terraform.parquet", from_hcl),
}


def make_row(content: str, langs: set[str], kind: str) -> dict[str, str]:
    target = {"is_valid": True, "category": {lang: True for lang in sorted(langs)}}
    return {
        "prompt": content,
        "target": json.dumps(target, separators=(",", ":")),
        "kind": kind,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stack-dir", type=Path, default=Path(
        "~/llm_models/guardrail_code_data/raw/the-stack-v1"
    ).expanduser())
    p.add_argument("--out-parquet", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/stack-polyglot-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--per-source-cap", type=int, default=2500,
                   help="Maximum rows kept per source parquet (random sample if exceeded).")
    p.add_argument("--max-bytes", type=int, default=12000,
                   help="Drop oversized files so they fit comfortably under max_len=4096 bytes "
                        "(after the model truncates).")
    p.add_argument("--seed", type=int, default=20260601)
    args = p.parse_args()

    out_rows: list[dict[str, str]] = []
    kind_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    per_source: list[tuple[str, int, int, int]] = []  # (source, total, multi, single)

    for lang, (filename, extractor) in SOURCE_EXTRACTORS.items():
        path = args.stack_dir / filename
        if not path.exists():
            print(f"[skip] {lang}: {path} missing", flush=True)
            per_source.append((lang, 0, 0, 0))
            continue
        df = pd.read_parquet(path)
        print(f"loaded {lang}: {len(df)} rows from {path.name}", flush=True)

        # cap then random sample
        if len(df) > args.per_source_cap:
            df = df.sample(n=args.per_source_cap, random_state=args.seed).reset_index(drop=True)

        n_multi = n_single = 0
        for _, row in df.iterrows():
            content = row.get("code") or ""
            if not isinstance(content, str):
                continue
            if not content or len(content.encode("utf-8", errors="replace")) > args.max_bytes:
                continue
            result = extractor(content)
            if result is None:
                continue
            langs, kind = result
            out_rows.append(make_row(content, langs, kind))
            kind_counts[kind] += 1
            for L in langs:
                lang_counts[L] += 1
            if kind == "multi":
                n_multi += 1
            else:
                n_single += 1
        per_source.append((lang, n_multi + n_single, n_multi, n_single))

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)

    print()
    print("per-source yield:")
    print(f"  {'source':12s} {'kept':>6} {'multi':>6} {'single':>6}")
    for src, total, m, s in per_source:
        print(f"  {src:12s} {total:>6} {m:>6} {s:>6}")
    print()
    print(f"kinds: {dict(kind_counts)}")
    print(f"top labels: {lang_counts.most_common()}")
    print(f"wrote {len(out_df)} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
