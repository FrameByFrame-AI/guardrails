# GLiNER2 Korean PII NER

Fine-tune [GLiNER2](https://github.com/fastino-ai/GLiNER2) for Korean PII (Personally Identifiable Information) entity extraction.

Replaces the default English DeBERTa-v3-base encoder with [team-lucid/deberta-v3-base-korean](https://huggingface.co/team-lucid/deberta-v3-base-korean) (89% Korean vocab, pretrained on Korean text) and fine-tunes on 105k Korean PII records from KDPII + synthetic_pii_finance_multilingual datasets.

## Entity Types

| Entity | Description |
|---|---|
| person | Full name |
| phone | Phone / mobile number |
| email | Email address |
| address | Street / mailing address |
| credit_card | Credit / debit card number |
| bank_account | Bank account (IBAN/BBAN) |
| id_number | Identity / customer / employee ID |
| ssn | Social security / resident registration number |
| passport | Passport number |
| driver_license | Driver's license number |
| ip_address | IPv4 / IPv6 address |
| url | Website URL |
| username | User ID / nickname |
| password | Password / API key / PIN |
| date_of_birth | Date of birth |
| organization | Company / organization name |

## Prerequisites

- Docker with NVIDIA runtime (`nvidia-container-toolkit`)
- GPU with CUDA support

## Setup

```bash
# Build the Docker image (only needed once)
docker compose build
```

## 1. Generate Training Data

Convert raw KDPII and synthetic_pii_finance datasets into GLiNER2 JSONL format.

```bash
docker compose run --rm gliner2-benchmark python format_pii_gliner2.py
```

Output: `data/pii_gliner2_train.jsonl` (105,852 records)

To limit records per dataset:

```bash
docker compose run --rm gliner2-benchmark python format_pii_gliner2.py --max-per-dataset 10000
```

## 2. Benchmark Base Model (Before Training)

Evaluate the unmodified GLiNER2 base model on Korean PII data to establish a baseline.

```bash
docker compose up gliner2-benchmark
```

Or with custom options:

```bash
docker compose run --rm gliner2-benchmark python benchmark_pii.py \
    --model fastino/gliner2-base-v1 \
    --samples 500 \
    --data-dir /data/processed \
    --output /data/benchmark_results/gliner2_pii_benchmark.json
```

Results are saved to `data/benchmark_results/`.

## 3. Train

Fine-tune GLiNER2 with Korean DeBERTa encoder on the PII dataset.

```bash
docker compose up gliner2-train
```

Or with custom options:

```bash
docker compose run --rm gliner2-train python train_pii.py \
    --model fastino/gliner2-multi-v1 \
    --train-data /data/pii_gliner2_train.jsonl \
    --output-dir /models/gliner2-multi-korean-pii-lora \
    --epochs 10 \
    --batch-size 4 \
    --encoder-lr 1e-5 \
    --task-lr 5e-4 \
    --korean-tokens 0 \
    --gradient-checkpointing \
    --lora \
    --bf16
```

Default `docker compose up gliner2-train` now trains `fastino/gliner2-multi-v1` with LoRA adapters, encoder gradient checkpointing, and `bf16`.

### Training Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `fastino/gliner2-base-v1` | Base GLiNER2 model |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `16` | Batch size per device |
| `--encoder-lr` | `1e-5` | Learning rate for encoder |
| `--task-lr` | `5e-4` | Learning rate for task heads |
| `--korean-tokens` | `5000` | Number of Korean tokens to add to the tokenizer |
| `--token-sample-limit` | `10000` | Number of records scanned to mine Korean tokens |
| `--gradient-checkpointing` | off | Enable encoder gradient checkpointing |
| `--lora` | off | Enable LoRA adapter training instead of full fine-tuning |
| `--lora-rank` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA scaling alpha |
| `--lora-dropout` | `0.0` | LoRA dropout |
| `--lora-targets` | `encoder,span_rep,classifier,count_embed,count_pred` | GLiNER2 module groups that receive LoRA adapters |
| `--fp16` | off | Enable fp16 mixed precision |
| `--bf16` | off | Enable bf16 mixed precision |
| `--no-korean-embedding-init` | off | Leave new Korean token embeddings randomly initialized |
| `--val-ratio` | `0.1` | Validation split ratio |
| `--max-train-samples` | `-1` | Limit train samples for smoke tests |
| `--max-eval-samples` | `-1` | Limit eval samples for smoke tests |

Trained model is saved to `models/gliner2-korean-pii/`.

## 4. Inference

After training:

```bash
docker compose run --rm gliner2-inference \
    python inference.py --text "김민수의 전화번호는 010-1234-5678이고 이메일은 minsu@example.com입니다."
```

## Project Structure

```
gliner2_korean_pii/
├── Dockerfile              # Dependencies only (no code)
├── docker-compose.yml      # Services: benchmark, train, inference
├── src/                    # All code (volume-mounted into containers)
│   ├── format_pii_gliner2.py
│   ├── benchmark_pii.py
│   ├── train_pii.py
│   ├── inference.py
│   └── debug_pii.py
├── data/                   # Training data + benchmark results
│   └── pii_gliner2_train.jsonl
└── models/                 # Trained model output
```

## Architecture

```
GLiNER2 (fastino/gliner2-base-v1)
│
├── Encoder: team-lucid/deberta-v3-base-korean  ← swapped from English DeBERTa
│            64k vocab, 89% Korean tokens
│            Pretrained on Korean text (TPU)
│
├── Task Heads (from GLiNER2)                   ← fine-tuned on Korean PII
│   ├── Span detection
│   ├── Entity classification
│   └── Count prediction
│
└── Tokenizer: Korean DeBERTa WordPiece
              김민수 → 1 token (vs 3 chars with English tokenizer)
```
