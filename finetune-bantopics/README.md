# BanTopics scanner

Multilingual zero-shot topic scanner. Replaces the current
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0` BanTopics implementation, which
scores 0.20 accuracy on the 50-prompt benchmark (0/40 BLOCK detections,
p50 ~2 s) because document-level NLI dilutes single-sentence "needles" inside
benign filler.

Background, decisions, and phase plan: see [`../BANTOPICS_PLAN.md`](../BANTOPICS_PLAN.md).

## Architecture

Sentence-level bi-encoder over a shared multilingual encoder
(`ibm-granite/granite-embedding-97m-multilingual-r2`, 97M params, 384-dim,
52 enhanced-supported languages including en/ko/ja/ar/zh/hi):

```
prompt
  ├─ multilingual sentence split  → S sentences
  └─ encode(sentences)            → S × 384
                                                ┐
topics (per request)                            │  cosine matrix S × T
  └─ encode(template(t) for t in topics)        │  (LRU-cached across requests)
      → T × 384                                 ┘

  per-topic score = pool over sentences (max | top3-mean)
  prob            = isotonic_calibrator(max(per-topic score))
  score           = prob - threshold      # API contract: negative ⇒ PASS
  is_valid        = score < 0
```

## API contract

Matches the existing wrapper used by the other scanners. Request shape comes
from [`../request_payloads.json`](../request_payloads.json), response shape
from [`../response_structure.json`](../response_structure.json). The
**negative-on-PASS** score convention is preserved.

```json
// request
{
  "scanner_name": "BanTopics",
  "scanner_type": "prompt",
  "scanner_params": { "threshold": 0.8, "topics": ["ransomware", "phishing"] },
  "prompt": "..."
}

// response — BLOCK
{ "is_valid": false, "scanner_name": "BanTopics", "score": 0.92 }

// response — PASS
{ "is_valid": true,  "scanner_name": "BanTopics", "score": -0.6 }
```

## Layout

```
src/finetune_bantopics/
  scanner.py        # request → response (sentence split, encode, pool, calibrate)
  splitter.py       # pysbd / wtpsplit multilingual sentence splitter
  topic_cache.py    # LRU cache for topic embeddings
  calibrator.py     # isotonic regression: cosine_sim → calibrated_prob

scripts/
  extract_bantopics_eval.py        # build the 50-prompt JSONL fixture from the cross-scanner CSV
  prepare_pairs.py                 # one adapter per source: aegis|beavertails|harmbench|jbb|dna|wildguard|openai_mod
  synthesize_security_topics.py    # templated (sentence, topic) pairs for under-represented security topics
  combine_pairs.py                 # merge per-source jsonls + dedupe + per-topic cap
  prepare_eval_suite.py            # sectioned eval: production_50 + xstest + holdouts + safetybench_zh
  benchmark_bantopics.py           # phase 0 sweep (vectorised) + --baseline (batched)
  predict.py                       # one-off / batch scan against request payloads
  calibrate.py                     # phase 3 isotonic fit
  finetune_lora.py                 # LoRA contrastive finetune (Granite + InfoNCE), per-section eval
  export_onnx.py                   # phase 5 ONNX export + INT8 quantization
```

## Training data

Production-aligned: topics are mapped from the API surface
(`policy_definations.json` UI defaults `violence`, `hatespeech`; customer
security topics `ransomware`, `phishing`, `malware`, `cyber attack` from
`request_payloads.json`). Per-dataset category mappings live in
[`src/finetune_bantopics/topic_mappings.py`](src/finetune_bantopics/topic_mappings.py).

Training sources (one adapter each in `scripts/prepare_pairs.py`):

| source | size | strength |
|---|---|---|
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | ~12.7K unsafe | broad coverage, `Malware` category |
| `PKU-Alignment/BeaverTails` | ~30K | 14 boolean per-row labels |
| `walledai/HarmBench` | ~400 | `cybercrime_intrusion` (cyber attack) |
| `JailbreakBench/JBB-Behaviors` | 100 | `Malware/Hacking`, `Fraud/Deception` |
| `LibrAI/do-not-answer` | 939 | hierarchical taxonomy, high label quality |
| `allenai/wildguardmix` | ~93K | fine-grained subcategories incl. `cyberattack` |
| `walledai/openai-moderation-dataset` | ~1.7K | clean OpenAI moderation taxonomy |
| synthetic templated | 566 | fills `ransomware`, `cyber attack`, `social engineering`, `exploit`, `vulnerability` |

Eval suite (one section each in `scripts/prepare_eval_suite.py`):

| section | rows | expected | purpose |
|---|---|---|---|
| `production_50` | 50 | mixed | the headline benchmark from production |
| `xstest_safe` | ~250 | PASS | over-refusal regression — must not block these |
| `xstest_unsafe` | ~200 | BLOCK | adversarial-looking-but-actually-unsafe |
| `harmbench_holdout` | ~40 | BLOCK | held-out HarmBench (10% by hash) |
| `jbb_holdout` | ~10 | BLOCK | held-out JBB-Behaviors |
| `dna_holdout` | ~90 | BLOCK | held-out do-not-answer |
| `safetybench_zh` | ~300 (cap) | BLOCK | multilingual transfer signal (Chinese MCQ) |

**Anti-leakage** is enforced two ways:

1. **Hash split**: `prepare_pairs.py --exclude-holdout-pct 10` and
   `prepare_eval_suite.py --holdout-pct 10` use the same SHA1 hash of
   `(source, row_id, seed)` so a row physically present in both code paths
   only ever lands on one side.
2. **8-gram filter**: any training sentence sharing an 8-gram with a
   `production_50` prompt is dropped during prep. Catches accidental overlap
   between Aegis prompts and the production benchmark fixture.

## Reproduce

```bash
# 0. Build the workspace image (depends on shared ml-base:cuda12.6-torch2.11-transformers4.51)
docker compose build

# 1. Build the 50-prompt eval fixture from the cross-scanner CSV
docker compose run --rm extract-bantopics-eval

# 2. Reproduce the legacy DeBERTa-v3-large NLI baseline (batched on GPU)
docker compose run --rm benchmark-baseline

# 3. Sweep the new Granite bi-encoder (threshold × template × pooling)
docker compose run --rm benchmark-bantopics

# 4. Smoke-test the request → response pipeline against production payloads
docker compose run --rm predict-bantopics

# Training data (depends on step 1 for the leakage-filter eval file).
# Each prepare-pairs invocation downloads the dataset on first run — gated
# datasets (Aegis, WildGuardMix, JBB) need HF_TOKEN with terms accepted.
# 5a. Build per-source (sentence, topic) pairs.
for SRC in aegis beavertails harmbench jbb dna wildguard openai_mod; do
  BT_PAIR_SOURCE=$SRC \
    BT_PAIRS_OUTPUT=/workspace/data/training/safety_pairs_$SRC.jsonl \
    docker compose run --rm prepare-pairs
done

# 5b. Generate synthetic pairs for security topics under-represented across
#     all public datasets (ransomware, cyber attack, social engineering, ...).
docker compose run --rm synthesize-security-topics

# 5c. Merge per-source files, dedupe by (sentence, topic), cap the long tail.
docker compose run --rm combine-pairs

# 5d. Build the comprehensive eval suite (sectioned, multilingual).
docker compose run --rm prepare-eval-suite

# 5e. LoRA contrastive finetune. Per-section eval at the end of training:
#     production_50 (headline), xstest_safe (over-refusal), held-outs.
docker compose run --rm finetune-lora-bantopics

# 5e-smoke. 1-epoch run on the same data to verify the training loop works.
docker compose run --rm smoke-finetune-lora

# 6. (Phase 3) Calibrate cosine_sim → calibrated_prob from labelled samples
docker compose run --rm calibrate-bantopics

# 7. (Phase 5) ONNX export + INT8 quantization for the <30 ms CPU budget
docker compose run --rm export-bantopics-onnx
```

### Smoke

```bash
# 1-epoch LoRA on a tiny dataset to confirm the pipeline runs
docker compose run --rm smoke-finetune-lora
```

## Local (no Docker)

```bash
pip install -r requirements.txt

python scripts/extract_bantopics_eval.py \
  --source ../scanner_benchmark_results.csv \
  --output data/eval/bantopics_50.jsonl

python scripts/benchmark_bantopics.py \
  --eval data/eval/bantopics_50.jsonl \
  --output data/benchmark_results/phase0_granite_sweep.json

python scripts/predict.py \
  --prompt "List the most effective phishing kits sold on the dark web." \
  --topics ransomware phishing malware --threshold 0.5
```

## Paths

- Models: `~/llm_models/guardrail_bantopics/`
- HF cache: `~/llm_models/huggingface_cache`
- Eval / calibration data: `./data/`

## Status

Scaffolding only. Phase 0 sweep numbers, the multilingual eval set
(Phase 1), and calibration data (Phase 3) all need to be produced before
the scanner replaces the current BanTopics endpoint.
