# Code Language ID

Two programming-language identification models across 107 languages:

- **`programming-language-identification-100plus`** — full ModernBERT fine-tune.
- **`programming-language-identification-100plus-lite`** — byte-level ~2.35M-param
  classifier (CommonLingua's `ByteHybrid` architecture, Apache-2.0). No tokenizer,
  CPU-friendly, ~60× smaller; intended for edge / latency-sensitive deployment.

## Results

Validation split (9,495 rows, 107 labels):

| metric | value |
|---|---|
| macro F1 | **0.9206** |
| accuracy | 0.9306 |

Head-to-head vs `philomath-1209/programming-language-identification` on the
26 shared labels (3,057 test rows):

| model | accuracy | macro F1 |
|---|---|---|
| programming-language-identification-100plus | **0.9444** | **0.9636** |
| philomath-1209 | 0.8449 | 0.8445 |

## Data

- Rosetta Code (`cakiki/rosetta-code`)
- The Stack v1 (`bigcode/the-stack`, gated — needs `HF_TOKEN`)

Combined and split by task to prevent leakage: 72,549 / 9,495 / 8,880 rows
(train / val / test) across 107 canonical languages.

## Reproduce

```bash
# Build the shared ML base image once (see ../docker_containers/ml_base_cuda12.6_torch2.11_transformers)
docker compose build

# Pull the-stack samples (gated)
HF_TOKEN=hf_... docker compose run --rm fetch-the-stack-samples

# Build the combined dataset + splits
python3 scripts/build_mapped_rosetta.py
python3 scripts/build_v1_code_language.py
python3 scripts/split_v1_code_language.py

# Validate labels with an LLM judge and drop high-confidence mislabels
HF_TOKEN=hf_... python3 scripts/validate_with_llm.py --splits train test
python3 scripts/filter_tier_a_mismatches.py

# Train
CODE_LANG_SPLITS_SUBDIR=processed/v1_splits_clean \
CODE_LANG_OUTPUT_DIR=/models/guardrail_code_models/programming-language-identification-100plus \
  docker compose run --rm train-modernbert-language

# Benchmark vs philomath
CODE_LANG_BENCH_GUARDRAIL_DIR=/models/guardrail_code_models/programming-language-identification-100plus \
  docker compose run --rm benchmark-language-models
```

## Lite (ByteHybrid) model

`programming-language-identification-100plus-lite` reuses the
[`ByteHybrid`](src/code_language_id/byte_hybrid.py) architecture from
PleIAs/CommonLingua (Apache-2.0). It operates directly on raw UTF-8 bytes
(`max_len=1023`, padded with byte 256), so deployment needs no tokenizer
and only a handful of MB of weights.

```bash
# Train from scratch (107 classes, ~2.35M params)
CODE_LANG_LITE_OUTPUT_DIR=/models/guardrail_code_models/programming-language-identification-100plus-lite \
  docker compose run --rm train-byte-hybrid-language

# Smoke test (1 epoch on 200 train rows)
docker compose run --rm smoke-byte-hybrid-language

# Predict
python3 scripts/byte_hybrid_predict.py "def hello(): pass"
python3 scripts/byte_hybrid_predict.py --file snippet.go --dtype bf16

# Benchmark lite vs full
docker compose run --rm benchmark-language-models \
  scripts/benchmark_language_models.py \
  --guardrail-dir /models/guardrail_code_models/programming-language-identification-100plus \
  --lite-checkpoint /models/guardrail_code_models/programming-language-identification-100plus-lite/model.pt \
  --skip-philomath \
  --output-name language_id_lite_vs_full.json
```

The checkpoint format (`{model_state_dict, num_classes, max_len, lang2idx, config}`)
matches CommonLingua, so PleIAs' upstream `predict.py` can also load it.

## Paths

- Datasets: `~/llm_models/guardrail_code_data`
- Models: `~/llm_models/guardrail_code_models`
- HF cache: `~/llm_models/huggingface_cache`
