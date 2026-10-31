# finetune-jev-distiller

An open red-teaming guardrail: **GLiNER2.5-Decide** (486M parameters, DeBERTa-v3-large backbone) fine-tuned to decide, for any input
to an AI agent, whether to **block** it (harmful request, jailbreak or prompt injection) and which of 24 **risk**
categories it carries. It is benchmarked against TypeSafe's **Jev**, **AWS Bedrock Guardrails**, GLM 5.3 and two open decision models
run zero-shot: **LiquidAI d1-3B** and **Mistral Shieldstral-1.0-3B**.

The model learns from the **datasets' own labels**, not from Jev's: each public dataset's label is mapped to
block/allow, then checked by an independent LLM judge (GLM 5.3) before it is used for training or benchmarking.

Latest results (4,222-prompt verified benchmark v2, details in [docs/experiments.md](docs/experiments.md)). This benchmark
was filtered by GLM, so these numbers are optimistic and favour GLM-like models; the gold-label eval that replaces
it is planned in [docs/EVAL.md](docs/EVAL.md).

| System | Accuracy | Recall | False-block rate | p50 latency | $ / 1k prompts |
|---|---|---|---|---|---|
| Jev | 0.938 | 0.881 | **0.014** | 443 ms | 0.046 |
| Bedrock Guardrails (Classic) | 0.801 | 0.797 | 0.196 | 141 ms | 0.199 (assumed price) |
| LiquidAI d1-3B (zero-shot) | 0.856 | 0.724 | 0.035 | 22 ms | 0.007 (GPU at $1/h) |
| Mistral Shieldstral-3B (zero-shot) | 0.851 | 0.795 | 0.103 | 18 ms | 0.006 (GPU at $1/h) |
| vllm-sr d3-mini (zero-shot) | 0.921 | 0.948 | 0.101 | 54 ms | 0.020 (GPU at $1/h) |
| This model, GLiNER2.5-Decide (run 3) | 0.930 | 0.905 | 0.049 | 31 ms | 0.011 (GPU at $1/h) |
| **This model, d3-mini LoRA (run 5)** | **0.973** | **0.973** | 0.027 | 56 ms | 0.018 (GPU at $1/h) |

On indirect prompt injection (attack emails) run 5 catches 99% with no false blocks; Jev catches 67%, Bedrock 28%.

## Run

```bash
cp .env.example .env   # OPENROUTER_API_KEY; AWS SSO profile and Bedrock guardrail; GPU UUID
make smoke             # 10 failure cases through Jev, GLM, Bedrock, compare (about $0.001)
make smoke-gpu         # the same cases through convert, split, train (2 steps), score, compare
make bench-data        # verified benchmark + Jev and Bedrock answers   (label caches: pay once per prompt)
make train-data        # verified training set, no benchmark prompt
make baselines         # d1-3B and Shieldstral-3B zero-shot on the benchmark (GPU)
make train score compare RUN_NAME=gold-gliner
make train-d3 score-d3 RUN_NAME=d3-ft   # the d3-mini fine-tune (best so far)
```

Everything runs on the shared base image `ml-base:unsloth-trl0.22-transformers-main`
(`docker_containers/ml_base_unsloth_trl_transformers`: torch 2.11, transformers main, gliner2 2.0.0, boto3); no
project image is built. Mounts: this directory at `/workspace` (code edits apply immediately), the HF cache at
`/cache/huggingface`, `~/llm_models` read-only at `/models`, and `~/.aws` for the SSO login
(`aws sso login --profile guardrail-bench`). Compose has two services, `cpu` and `gpu`; the Makefile holds the
commands. Training settings are Makefile variables: `make train EPOCHS=5 RUN_NAME=exp3`.

## Layout

Code is grouped by step. Compose puts every code folder on `PYTHONPATH`, so modules import each other by name
(`from guardrate import ...`) from any folder; run a script by its path, e.g. `docker compose run --rm cpu
eval/scorecard.py`. A command that sets its own `-e PYTHONPATH` must repeat the folders (`SRC` in the Makefile).

| Folder | Contents |
|---|---|
| `build/` | Datasets: dataset spec, benchmark and training candidates, data for runs 11-13 (`build_v11..v13.py`), AA, GuardBench and injection benchmarks, splits, hard benign emails, gliner records |
| `callers/` | Systems scored as they are deployed: Jev (`jev_caller.py` also defines the trained questions), Bedrock, open decision models, GLiNER, other open guards, GLM label checks and adjudication |
| `train/` | `train_d3.py` (LoRA fine-tune of a d3 model, exported as a package with its manifest), `merge_runs.py` (soups), `d3_manifest.py`, `train_gliner.py` |
| `eval/` | `guardrate.py` (GuardRate rebuild: build, audit, run, score), `scorecard.py` (the internal suite), injection and model-card tables, AUC / false-block tables, system comparison |
| `serve/` | Latency and serving clients: vllm-srun (`srun_bench.py`), llama-server (`llamacpp_run.py`), CPU latency, remote latency test (`latency_bench.py`) |
| `lib/` | `resume.py`: crash-safe, resumable jsonl output used by every long job |
| `archive/` | Data builders and experiments of runs 6-9 and dropped ideas (int8); kept for reproducibility, not used by current runs |
| `scripts/` | Shell entry points for each run and evaluation, `gpu_watch.sh` (GPU telemetry before every GPU job) |
| `docs/` | `experiments.md` (the log of every run and decision), `EVAL.md`, `SUMMARY.md` |
| `hf/` | Release folders (cards, compose, `check.py`, manifests in git; weights are hard links, ignored) |
| `tests/` | `smoke.sh`, `smoke_gpu.sh` and the 10 failure cases they use |

`data/` holds the append-only label caches (`jev_labels`, `glm_labels`, `bedrock_labels`), the benchmark, the
training set and run 1's data (`data/run1_jev/`). Nothing deletes `data/`; `make clean` deletes `runs/`.

## Latency test (GPU and CPU servers)

Serve a model (`scripts/serve_gpu.sh` for 1.2 or Lite on a GPU, `scripts/serve_cpu.sh` for the GGUF on a CPU), then measure latency from any machine (the client is one file, Python 3.8+,
standard library only).

```bash
# on the server (Docker; llama.cpp official image + nginx)
scripts/serve_cpu.sh start                    # 127.0.0.1:47613, 1 worker x 8 threads, API key required
scripts/serve_cpu.sh start --bind 0.0.0.0     # accept remote clients (or --bind <LAN / VPN address>)
scripts/serve_cpu.sh status | stop
cat runs/serve_cpu/api_key                    # generated on first start

# on the client: copy serve/latency_bench.py
python3 latency_bench.py --api-key KEY                                     # server on the same machine
python3 latency_bench.py --url http://<server-ip>:47613 --api-key KEY      # remote server
python3 latency_bench.py --url ... --api-key KEY --sweep 1,2,4,8           # queueing under concurrency
python3 latency_bench.py --url ... --api-key KEY --input prompts.xlsx --csv out.csv
```

| Server option | Default | Meaning |
|---|---|---|
| `--workers` | 1 | llama-server processes behind nginx (least-busy routing) |
| `--threads` | 8 | threads per worker; workers x threads <= free physical cores |
| `--slots` | 1 | parallel slots per worker (`-np`); more adds nothing on CPU |
| `--memory-gb` | 4 | memory cap per worker (peak 2.4 GiB at a 5,400-token input) |
| `--ctx` | 8192 | longest input in tokens |
| `--port` / `--bind` | 47613 / 127.0.0.1 | listening port and address |
| `--api-key` / `--no-key` | generated | key kept in `runs/serve_cpu/api_key` |
| `--first-core` | auto | first physical core; workers are pinned to disjoint cores, highest first |

| Client option | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:47613` (env `GUARD_URL`) | server |
| `--api-key` | env `GUARD_API_KEY` | |
| `--concurrency` / `--sweep` | 1 | requests in flight / list of levels |
| `--requests` / `--duration` | 200 | per level, or seconds per level |
| `--rate` | closed loop | open loop: Poisson arrivals, requests/s |
| `--warmup` | 10 | untimed requests first |
| `--input` / `--sheet` | 29 built-in toxicity prompts | xlsx (Prompts / Expected Output), csv, jsonl or txt |
| `--threshold`, `--timeout`, `--csv`, `--json` | 0.5, 120 s | decision threshold, per-request timeout, outputs |

It prints p50 / p90 / p95 / p99 / max, requests/s, errors, network RTT, input tokens and decisions against the
expected labels; every response carries the server's workers / threads / slots / memory.

**Read concurrency right.** A worker answers one request at a time, so more requests in flight than workers only
queue. Ryzen 9 5900X, 1 worker x 8 threads, ~160 tokens: concurrency 1 / 2 / 4 / 8 gave p50 367 / 734 / 1,468 /
2,913 ms at a flat 2.7 requests/s. The team's 1.2 s (GuardLite1.2GGUFCPU.xlsx) was four sheets run at once against
one server; the last rows, run alone, took ~385 ms. For more requests/s add workers on separate cores
(`--workers 2 --threads 4`: ~17% more requests/s, each request slower); for the lowest latency keep 1 x 8.

## How the labels are made

1. **Dataset label → block/allow** (`build/datasets_spec.py`). For example WildGuard `prompt_harm_label`, AEGIS
   `prompt_label`, label 1 in the injection datasets, any OpenAI moderation flag, XSTest `unsafe`. Datasets that are
   harmful by construction (HarmBench, LLMail, Gandalf, JailbreakV) are always block; Enron emails (aeslc) are
   always allow. BeaverTails rates responses, not prompts, so it is not used.
2. **Benchmark vs train.** Each dataset's test split goes to the benchmark and its train split to training. Some
   datasets are benchmark-only (HarmBench, JBB, XSTest, OpenAI moderation, in-the-wild jailbreaks, ChatGPT
   jailbreaks) so the benchmark also measures generalisation. Training excludes every benchmark candidate.
3. **Verification** (`callers/verify_glm.py`). GLM 5.3 Flash, pinned to Fireworks with a strict JSON schema, labels each
   prompt without seeing the dataset label. If it disagrees, GLM 5.3 Prime (Alibaba) labels it; the prompt is kept
   when Prime agrees with the dataset. Unresolved prompts go to `*_disputed.jsonl`. The benchmark also drops a whole
   dataset under 80% verified; training keeps every verified prompt. Providers are pinned with no
   fallbacks: providers differ in quantisation, and some ignore the schema.

## Notes

- **Every long job resumes after a crash** (the GPU has fallen off the bus twice, Xid 79). Scorers
  (`callers/gliner_caller.py`, `callers/decision_caller.py`, `eval/guardrate.py run`) write one flushed row per item and skip finished
  items on rerun (`lib/resume.py`); API labelling uses append-only caches; `train/train_d3.py` checkpoints every 200 steps to
  `<out>/resume.pt` and continues from it; finished training runs skip themselves. `train/train_gliner.py` cannot resume
  mid-run (gliner2's trainer has no resume). GPU make targets start `scripts/gpu_watch.sh`, which logs power, temperature
  and clocks to `runs/logs/gpu_dmon.log`. After a crash, rerun the same command.

- Long inputs are truncated to `max_len=1024` gliner2 word units; 2048 runs out of memory on 96 GB because a
  19-word punctuation string tokenizes to 10k tokens. The preflight runs the longest batch by tokens first.
- gliner2's defaults are overridden so training prompts match inference (no dropped or renamed labels, descriptions
  always shown), `drop_last` is off, and a failed micro-batch aborts training.
- Bedrock's cost assumes $0.15 per 1,000 text units (not confirmed for ap-south-1). GLiNER's, d1's and Shieldstral's cost
  assumes a GPU at $1/hour, one prompt at a time.
- d1-3B is under the LFM Open License (commercial use above $10M annual revenue needs a separate licence);
  Shieldstral is Apache 2.0.
