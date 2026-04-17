# Multilingual AI Guardrail Fine-Tuning

**Fine-tuning pipelines for multilingual content-safety guardrails** — PII detection,
content moderation, prompt-injection detection — across multiple model architectures
and languages.

Currently ships Korean + English. Hindi, Tamil, English-Hindi, and more on the
roadmap.

## Models

| Pipeline | Base model | Size | Task | Status |
|---|---|---|---|---|
| `finetune-gemma/` | Gemma 4 E2B | 5B (2B active) | Full guardrail (PII + moderation + injection) | Trained (Korean) |
| `finetune-qwen/` | Qwen 3.5 0.8B | 0.85B | Full guardrail | Trained (Korean) |
| `gliner2_korean_pii/` | GLiNER2-multi | 180M | PII NER only | Training |

All three pipelines share the same dataset format and can be retargeted to any
language by pointing `DATASET_DIR` at a different corpus.

## Benchmarks (held-out test split — no train/test leak)

| Dataset | Gemma 4 E2B | Qwen 3.5 0.8B |
|---|---|---|
| Korean PII (KDPII) | 0.971 | 0.958 |
| Prompt Injection (PIGuard) | 0.988 | 0.964 |
| Prompt Injection (RaccoonBench) | 1.000 | 1.000 |
| Korean Moderation (selectstar) | 0.995 | 1.000 |
| Korean Slang Blocklist | 1.000 | 0.925 |
| Korean Hate Speech (KMHaS) | 0.869 | 0.643 |
| **Overall (13 datasets)** | **0.922** | **0.861** |

Full per-dataset breakdown in `benchmark-results/*.json`.

## Quick start

```bash
# 1. Clone external dataset repos (not committed; see DATA.md)
git clone https://github.com/<org>/korean-guardrail-dataset.git
# etc.

# 2. Point datasets/ at the data you want to train on
mkdir -p datasets
ln -s ../korean-guardrail-dataset/data datasets/korean

# 3. Build training data
cd finetune-gemma
docker compose run --rm format-data
docker compose run --rm prepare-train-data

# 4. Train
docker compose up train

# 5. Serve with vLLM
docker compose up -d vllm

# 6. Benchmark
docker compose up benchmark
```

## Adding a new language

The pipeline is language-agnostic. To add a new language (e.g. Hindi):

1. Prepare a dataset repo with the same schema as `korean-guardrail-dataset`:
   ```
   <your-hindi-repo>/data/
   ├── processed/                # raw per-source JSONL, one per dataset
   └── processed_split/          # train/test splits (run split_datasets.py)
   ```

   Each record:
   ```json
   {
     "query": "...",
     "blocked": true,
     "type": "pii-filter | moderation | safety-classifier | rules-based-protections | output-validation",
     "answer": [{"form": "...", "label": "..."}],
     "topic": ["..."]
   }
   ```

2. Symlink it into `datasets/`:
   ```bash
   ln -s ../hindi-guardrail-dataset/data datasets/hindi
   ```

3. Train by pointing `DATASET_DIR` at it:
   ```bash
   cd finetune-gemma
   DATASET_DIR=../datasets/hindi docker compose run --rm format-data
   DATASET_DIR=../datasets/hindi docker compose up train
   ```

The same three pipelines (`finetune-gemma`, `finetune-qwen`, `gliner2_korean_pii`)
train on any language that follows the schema — no code changes required.

## Folder layout

```
guardrail/
├── README.md                   # this file
├── DATA.md                     # data pipeline / rebuild instructions
├── .gitignore
├── datasets/                   # per-language dataset mountpoints (symlinks, gitignored)
│   └── korean → ../korean-guardrail-dataset/data
├── finetune-gemma/             # Gemma 4 E2B fine-tuning pipeline
├── finetune-qwen/              # Qwen 3.5 0.8B fine-tuning pipeline
├── gliner2_korean_pii/         # GLiNER2 NER pipeline (PII-only)
└── benchmark-results/          # honest held-out F1 scores (committed)
```

Each `finetune-*/` project is self-contained:
```
finetune-<model>/
├── Dockerfile                  # training environment
├── Dockerfile.vllm             # inference serving environment
├── docker-compose.yml          # format-data / prepare / train / vllm / benchmark services
└── src/
    ├── format_training_data.py # dataset → chat-template JSONL
    ├── prepare_dataset.py      # tokenize, filter, cache
    ├── training_data.py        # shared helpers
    ├── train.py                # TRL + Unsloth fine-tuning
    ├── merge_and_push.py       # merge LoRA, upload to HF
    ├── benchmark.py            # concurrent vLLM API benchmark
    └── inference.py            # ad-hoc testing
```

## Roadmap

- [x] Korean (Gemma 4 E2B, Qwen 3.5 0.8B, GLiNER2)
- [x] English (implicit in both models via multilingual pretraining + English PII data)
- [ ] Hindi
- [ ] Tamil
- [ ] English + Hindi code-switching
- [ ] Adversarial augmentation (char obfuscation, homoglyphs, indirect injection — see [AccuKnox blog](https://accuknox.com/blog/illusion-of-security-prompt-guardrails-zero-trust-ai-spm))
- [ ] Multimodal (image/PDF prompt injection) for Gemma's vision tower

## Key design principles

- **Language-agnostic pipelines** — same Docker setup, swap `DATASET_DIR`
- **Honest evaluation** — deterministic 90/10 train/test split, no leak
- **Reproducibility** — seeded splits, pinned Docker images, all configs versioned
- **Model diversity** — three model families covering different compute/accuracy trade-offs

## See also

- [`DATA.md`](./DATA.md) — how to rebuild training data from scratch
- [`benchmark-results/`](./benchmark-results/) — per-dataset F1 scores
- [`finetune-gemma/docs/`](./finetune-gemma/docs/) — Gemma-specific plan and notes
