# Data Pipeline

Training data is **not committed** to git (too large, license-sensitive). This doc
describes the data layout and how to rebuild from source.

## Dataset layout convention

All fine-tuning pipelines expect a dataset directory with this structure:

```
<your-dataset-root>/
├── processed/                     # per-source JSONL (optional — raw data before split)
│   ├── KDPII.jsonl
│   ├── KMHaS.jsonl
│   └── ...
└── processed_split/               # required — 90/10 train/test splits
    ├── KDPII.train.jsonl
    ├── KDPII.test.jsonl
    ├── KMHaS.train.jsonl
    ├── KMHaS.test.jsonl
    └── ...
```

Every record has the same schema regardless of language:
```json
{
  "query": "string — the input text to classify",
  "blocked": true,
  "type": "pii-filter | moderation | safety-classifier | rules-based-protections | output-validation",
  "answer": [{"form": "entity text", "label": "entity type"}],
  "topic": ["topic tags"]
}
```

## Pointing a pipeline at your data

Each `finetune-*/docker-compose.yml` reads the dataset directory from the
`DATASET_DIR` environment variable, defaulting to `../datasets/korean` (a
symlink in this repo pointing at the Korean dataset).

**Train on a different language** — just override the env var:

```bash
cd finetune-gemma
DATASET_DIR=../datasets/hindi docker compose run --rm format-data
DATASET_DIR=../datasets/hindi docker compose run --rm prepare-train-data
DATASET_DIR=../datasets/hindi docker compose up train
```

Or set it once in a `.env` file at the project root.

## Rebuilding from source (Korean reference)

### 1. Clone the external dataset repo

```bash
cd <repo-root>
git clone https://github.com/<org>/korean-guardrail-dataset.git
```

This repo contains preprocessing scripts that download and normalize raw
sources (KDPII, KMHaS, PIGuard, etc.) into `processed/*.jsonl`.

### 2. Create the splits

```bash
cd korean-guardrail-dataset/data
python src/split_datasets.py \
    --input-dir processed \
    --output-dir processed_split \
    --test-fraction 0.1 \
    --seed 42
```

Output: `processed_split/*.{train,test}.jsonl` (deterministic, reproducible).

### 3. Symlink into this repo

```bash
cd <repo-root>
mkdir -p datasets
ln -s ../korean-guardrail-dataset/data datasets/korean
```

### 4. Format training data for each model

```bash
# Gemma
cd finetune-gemma
docker compose run --rm format-data
docker compose run --rm prepare-train-data

# Qwen
cd ../finetune-qwen
docker compose run --rm format-data
docker compose run --rm prepare-train-data

# GLiNER2 (PII-only)
cd ../gliner2_korean_pii
docker compose run --rm gliner2-format
```

Outputs inside each project's `data/`:
- `guardrail_train.jsonl` (Gemma/Qwen — formatted as chat-template conversations, ~643MB)
- `guardrail_train_prepared/` (tokenized HF Dataset, filtered by seq length)
- `pii_gliner2_train.jsonl` (GLiNER2 NER format, ~39MB)

### 5. Train

```bash
# Gemma
cd finetune-gemma
docker compose up train

# Qwen (single GPU, batch 16 avoids OOM spikes)
cd ../finetune-qwen
BATCH_SIZE=16 GRAD_ACCUM=4 MAX_SEQ_LENGTH=1024 docker compose up train

# GLiNER2 (LoRA, gradient checkpointing for large batches)
cd ../gliner2_korean_pii
GLINER_BATCH_SIZE=16 docker compose up gliner2-train
```

### 6. Benchmark

```bash
# Start vLLM serving
docker compose up -d vllm

# Concurrent benchmark on held-out test split
docker compose up benchmark
```

Results saved to `{project}/data/benchmark_results/*.json` (these **are** committed
so others can see how the model performs without needing to retrain).

## Adding a new language

Same pattern. Example for Hindi:

1. Prepare a preprocessing repo:
   ```
   hindi-guardrail-dataset/
   └── data/
       ├── processed/              # your preprocessed JSONL per source
       └── src/
           └── split_datasets.py   # copy from korean-guardrail-dataset
   ```

2. Run the split:
   ```bash
   cd hindi-guardrail-dataset/data
   python src/split_datasets.py --input-dir processed --output-dir processed_split
   ```

3. Symlink and train:
   ```bash
   cd <repo-root>
   ln -s ../hindi-guardrail-dataset/data datasets/hindi
   cd finetune-gemma
   DATASET_DIR=../datasets/hindi docker compose run --rm format-data
   DATASET_DIR=../datasets/hindi docker compose up train
   ```

No code changes needed — the same `format_training_data.py` script auto-detects
`*.train.jsonl` files and uses them if present (falls back to `*.jsonl` with a warning).

## What's committed vs regenerated

| Path | Status |
|---|---|
| Source code (`src/*.py`), `Dockerfile`, `docker-compose.yml` | **Committed** ✅ |
| `README.md`, `DATA.md`, `.gitignore` | **Committed** ✅ |
| `benchmark-results/*.json`, `*/data/benchmark_results/*.json` | **Committed** ✅ |
| `gliner2_korean_pii/data/korean_rrn_synthetic.jsonl` | **Committed** (our synthetic data) ✅ |
| `datasets/<lang>/` symlinks | Not committed (local only) ❌ |
| `korean-guardrail-dataset/` submodule | External repo, gitignored ❌ |
| `*/data/*.jsonl` (training files) | Regenerate with `format-data` ❌ |
| `*/data/processed*/` | External — comes from dataset repo ❌ |
| `*/data/guardrail_train_prepared/` | Regenerate with `prepare-train-data` ❌ |
| Model weights, checkpoints | Download from HF or retrain ❌ |

## Dataset sources (Korean baseline)

| Dataset | License | Attribution |
|---|---|---|
| KDPII | CC BY 4.0 | [Zenodo 10968609](https://zenodo.org/records/10968609) |
| synthetic_pii_finance_multilingual | MIT | [gretelai](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) |
| KMHaS, KOLD, APEACH, korean_unsmile | Varies | Korean NLP research papers |
| selectstar | See publisher | Commercial Korean moderation dataset |
| PIGuard, RaccoonBench | Varies | `PIGuard/` submodule |
| `korean_rrn_synthetic.jsonl` | MIT (ours) | Generated + Gemma-verified via `generate_korean_rrn.py` |

## Reproducibility

Train/test split uses `seed=42 + hash(dataset_name)` — same input produces identical
output, so benchmark numbers are comparable across runs and contributors.

```bash
# Any user running:
python src/split_datasets.py --seed 42 --test-fraction 0.1
# gets the exact same 90/10 split as ours.
```
