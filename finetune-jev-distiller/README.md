# finetune-jev-distiller

An open red-teaming guardrail: **GLiNER2.5-Decide** (486M parameters, DeBERTa-v3-large backbone) fine-tuned to decide, for any input
to an AI agent, whether to **block** it (harmful request, jailbreak or prompt injection) and which of 24 **risk**
categories it carries. It is benchmarked against TypeSafe's **Jev**, **AWS Bedrock Guardrails**, GLM 5.3 and two open decision models
run zero-shot: **LiquidAI d1-3B** and **Mistral Shieldstral-1.0-3B**.

The model learns from the **datasets' own labels**, not from Jev's: each public dataset's label is mapped to
block/allow, then checked by an independent LLM judge (GLM 5.3) before it is used for training or benchmarking.

Latest results (4,222-prompt verified benchmark v2, details in [experiments.md](experiments.md)). This benchmark
was filtered by GLM, so these numbers are optimistic and favour GLM-like models; the gold-label eval that replaces
it is planned in [EVAL.md](EVAL.md).

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

## Files

| File | Role |
|---|---|
| `datasets_spec.py` | Every dataset: HF id, text column, group, train splits, benchmark splits, and the mapping from its label to block/allow |
| `build_benchmark.py` | Benchmark candidates from benchmark splits (and benchmark-only datasets) |
| `build_train.py` | Training candidates from train splits, excluding every benchmark candidate |
| `gen_benign_emails.py` | Synthetic benign emails with LLMail attacks' surface features (resumable) |
| `verify_glm.py` | GLM 5.3 Flash checks each dataset label; GLM 5.3 Prime settles disagreements; drops datasets under 80% verified |
| `to_gliner.py` | Verified rows → gliner2 records (`block` from the dataset label, `risk` from the judge) |
| `split_data.py` | Train / val by prompt hash |
| `train_gliner.py` | Fine-tune, with a memory preflight on the longest batch and train/inference prompt parity |
| `train_d3.py` | LoRA fine-tune of d3-mini through its own inference path (parity-checked), exported as a d3 package |
| `build_v6.py` | Run 6 data: run 5 prompts minus two sources, plus harmful requests and response pairs, decontaminated against every eval item |
| `split_v6.py` | Holds out whole sources (one per kind) as the validation set for checkpoint and threshold choice |
| `build_v7.py` | Run 7 data: run 6 plus 4,000 benign WildGuardMix train prompts |
| `build_v8.py` | Run 8 data: run 7 rows, each asked only the question its source labels (attack or unsafe content) |
| `gliner_caller.py` | Run a GLiNER model over prompts one at a time; block probability, risk, latency, GPU cost |
| `jev_caller.py` | Jev via OpenRouter's decisions API; also defines the 24 risk categories and the two questions |
| `bedrock_caller.py` | AWS Bedrock `ApplyGuardrail` (input source) |
| `decision_caller.py` | Zero-shot open decision models: d1-3B (asked Jev's exact questions) and Shieldstral-3B (one yes/no "Is this input unsafe?" under a policy that names harmful requests, jailbreaks and prompt injection) |
| `compare_systems.py` | Accuracy, precision, recall, F1, false-block rate overall and per group; latency; cost |
| `tests/` | `smoke.sh`, `smoke_gpu.sh` and the 10 failure cases they use |

`data/` holds the append-only label caches (`jev_labels`, `glm_labels`, `bedrock_labels`), the benchmark, the
training set and run 1's data (`data/run1_jev/`). Nothing deletes `data/`; `make clean` deletes `runs/`.

## How the labels are made

1. **Dataset label → block/allow** (`datasets_spec.py`). For example WildGuard `prompt_harm_label`, AEGIS
   `prompt_label`, label 1 in the injection datasets, any OpenAI moderation flag, XSTest `unsafe`. Datasets that are
   harmful by construction (HarmBench, LLMail, Gandalf, JailbreakV) are always block; Enron emails (aeslc) are
   always allow. BeaverTails rates responses, not prompts, so it is not used.
2. **Benchmark vs train.** Each dataset's test split goes to the benchmark and its train split to training. Some
   datasets are benchmark-only (HarmBench, JBB, XSTest, OpenAI moderation, in-the-wild jailbreaks, ChatGPT
   jailbreaks) so the benchmark also measures generalisation. Training excludes every benchmark candidate.
3. **Verification** (`verify_glm.py`). GLM 5.3 Flash, pinned to Fireworks with a strict JSON schema, labels each
   prompt without seeing the dataset label. If it disagrees, GLM 5.3 Prime (Alibaba) labels it; the prompt is kept
   when Prime agrees with the dataset. Unresolved prompts go to `*_disputed.jsonl`. The benchmark also drops a whole
   dataset under 80% verified; training keeps every verified prompt. Providers are pinned with no
   fallbacks: providers differ in quantisation, and some ignore the schema.

## Notes

- **Every long job resumes after a crash** (the GPU has fallen off the bus twice, Xid 79). Scorers
  (`gliner_caller.py`, `decision_caller.py`, `guardrate.py run`) write one flushed row per item and skip finished
  items on rerun (`resume.py`); API labelling uses append-only caches; `train_d3.py` checkpoints every 200 steps to
  `<out>/resume.pt` and continues from it; finished training runs skip themselves. `train_gliner.py` cannot resume
  mid-run (gliner2's trainer has no resume). GPU make targets start `gpu_watch.sh`, which logs power, temperature
  and clocks to `runs/logs/gpu_dmon.log`. After a crash, rerun the same command.

- Long inputs are truncated to `max_len=1024` gliner2 word units; 2048 runs out of memory on 96 GB because a
  19-word punctuation string tokenizes to 10k tokens. The preflight runs the longest batch by tokens first.
- gliner2's defaults are overridden so training prompts match inference (no dropped or renamed labels, descriptions
  always shown), `drop_last` is off, and a failed micro-batch aborts training.
- Bedrock's cost assumes $0.15 per 1,000 text units (not confirmed for ap-south-1). GLiNER's, d1's and Shieldstral's cost
  assumes a GPU at $1/hour, one prompt at a time.
- d1-3B is under the LFM Open License (commercial use above $10M annual revenue needs a separate licence);
  Shieldstral is Apache 2.0.
