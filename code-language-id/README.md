# Code Language ID

Fine-tune ModernBERT for programming-language identification across 107
languages. Released model: `programming-language-identification-100plus`.

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
# Build the shared ML base image once (see ../docker_containers/ml_base_cuda12.4_torch2.4_transformers)
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

## Paths

- Datasets: `~/llm_models/guardrail_code_data`
- Models: `~/llm_models/guardrail_code_models`
- HF cache: `~/llm_models/huggingface_cache`
