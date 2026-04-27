# Privacy Filter — Korean PII fine-tune

LoRA fine-tune of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)
on Korean PII data (KDPII + `korean_rrn_synthetic`), targeting span-level
detection of personal identifiers in Korean chat-style and form-style text.

## Final results (held-out test, span F1)

| label | base model | fine-tuned (v6 LoRA) |
|---|---|---|
| `private_phone` | 0.65 | **1.00** |
| `private_url` | 0.21 | **1.00** |
| `private_email` | 0.86 | **1.00** |
| `account_number` | 0.31 | **0.98** |
| `private_date` | 0.00 | **0.90** |
| `private_address` | 0.00 | **0.78** |
| `private_person` | 0.06 | **0.69** |

Aggregate: **+0.58 F1** vs base, all on a 1.5B-parameter sparse-MoE base
adapted with **only ~614k trainable parameters** (LoRA rank 16 on attention
projections + classifier head; MoE experts and FFN frozen).

## Method

The base model is a sparse MoE token classifier (gpt-oss-style architecture,
1.5B total / ~50M active per forward pass, 128 experts, top-4 routing).
Naively full-fine-tuning it on KDPII *hurt* private_person and private_address
because Korean tokens routed to a small subset of experts received sparse,
noisy gradient updates that corrupted pretrained Korean knowledge faster than
they taught the new task.

LoRA gates gradients to attention projections and the new classifier head only —
every other parameter (MoE experts, FFN, embeddings, router, layernorms) stays
frozen. The base's pretrained capability is preserved while attention adapts to
the new label vocabulary.

Recipe:

| | value |
|---|---|
| Base model | `openai/privacy-filter` (`OpenAIPrivacyFilterForTokenClassification`) |
| Adaptation | LoRA r=16, alpha=32, dropout=0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Modules saved fully | `score` (classifier head, label vocab differs from base) |
| Trainable params | ~614k (~0.04% of 1.5B) |
| Optimizer | AdamW, lr 5e-4, cosine schedule, warmup 0.1 |
| Effective batch | 128 (per_device 64 × 2 GPUs) |
| Epochs | 10, early stopping on `eval_span_f1` (patience 3) |
| Precision | bf16 mixed |
| Sequence length | 512 |
| Save format | bf16 safetensors (after `merge_and_unload()`) |

## Data

| source | split | usage |
|---|---|---|
| `KDPII.jsonl` | deterministic text-hash (5/5/90% test/val/train) | primary Korean PII corpus |
| `korean_rrn_synthetic.jsonl` | train only | Korean RRN coverage; never in val/test to avoid template leakage |

Label space (9 categories):
`private_person`, `personal_handle`, `private_phone`, `private_email`,
`private_address`, `private_date`, `private_url`, `account_number`, `ip_address`.

KDPII source-label remap applied during conversion:

- `PS_NAME`, `PS_NICKNAME` → `private_person`
- `PS_ID` → `personal_handle`
- All `QT_*_NUMBER` (resident, passport, driver, alien, plate, account, card) → `account_number`
- `LC_ADDRESS` → `private_address` (with single-token toponyms `한국/서울/부산/수원` filtered)
- `QT_IP` → `ip_address`, `TMI_EMAIL` → `private_email`, `TMI_SITE` → `private_url`,
  `QT_PHONE/QT_MOBILE` → `private_phone`, `DT_BIRTH` → `private_date`
- 1-character `PS_NAME` (surname-only) filtered as noise

Post-processing strips leading/trailing whitespace from all spans (defensive).

## Layout

```
src/
  convert_korean_pii_to_opf.py   # KDPII + RRN → OPF JSONL
  train_korean_hf_ddp.py         # DDP trainer with LoRA support
  train_korean_hf_ddp.sh         # launcher (env-var configured)
  benchmark_pii_heldout.py       # base-vs-finetuned span benchmark
  analyze_errors.py              # per-category EXACT/BOUNDARY/MISSED breakdown
data/                            # gitignored: generated/, checkpoints/, benchmark_results/
docker-compose.yml               # convert-pii, train-pii, benchmark-pii services
Dockerfile                       # CUDA 12.4 + torch 2.4 + transformers (HEAD) + peft
```

## Usage

```bash
# Build
docker compose build

# 1) Convert Korean PII sources → OPF JSONL
docker compose run --rm convert-pii

# 2) Train (LoRA, KDPII v4)
docker compose run --rm \
  -e PF_OUTPUT_DIR=/workspace/data/checkpoints/ko_pii_hf_ddp_v6_lora \
  -e PF_USE_LORA=1 \
  -e PF_LEARNING_RATE=5e-4 \
  -e PF_EPOCHS=10 \
  -e PF_EARLY_STOPPING_PATIENCE=3 \
  -e PF_BATCH_SIZE=64 \
  -e PF_GRAD_ACCUM=1 \
  -e PF_EVAL_BATCH_SIZE=128 \
  train-pii bash -lc 'bash /workspace/train_korean_hf_ddp.sh'

# 3) Benchmark base vs fine-tuned
docker compose run --rm benchmark-pii \
  python benchmark_pii_heldout.py \
  --models /models/privacy-filter /data/checkpoints/ko_pii_hf_ddp_v6_lora \
  --model-names base finetuned \
  --dataset /data/generated/ko_pii_opf_v4/test.jsonl \
  --label-space-json /data/generated/ko_pii_opf_v4/label_space.json \
  --max-length 512 \
  --output /data/benchmark_results/privacy_filter_v6_lora_full.json

# 4) Per-category error breakdown (EXACT / BOUNDARY / MISSED / SPURIOUS)
docker compose run --rm benchmark-pii \
  python analyze_errors.py \
  --model /data/checkpoints/ko_pii_hf_ddp_v6_lora \
  --dataset /data/generated/ko_pii_opf_v4/test.jsonl \
  --max-length 512 \
  --examples-per-bucket 50 \
  --output /data/benchmark_results/v6_lora_full_breakdown.json
```

### Configurable launcher env vars

| var | default | meaning |
|---|---|---|
| `PF_OUTPUT_DIR` | `/workspace/data/checkpoints/ko_pii_hf_ddp_v4` | checkpoint dir |
| `PF_DATA_DIR` | `/workspace/data/generated/ko_pii_opf_v4` | OPF data dir |
| `PF_USE_LORA` | (off) | set `1` to enable LoRA |
| `PF_LORA_R`, `PF_LORA_ALPHA`, `PF_LORA_DROPOUT` | `16, 32, 0.05` | LoRA config |
| `PF_LORA_TARGET_MODULES` | `q_proj,k_proj,v_proj,o_proj` | which modules get LoRA |
| `PF_EPOCHS`, `PF_BATCH_SIZE`, `PF_GRAD_ACCUM` | `5, 16, 4` | training schedule |
| `PF_LEARNING_RATE`, `PF_LR_SCHEDULER_TYPE` | `2e-5, cosine` | optimizer |
| `PF_MAX_LENGTH` | `512` | tokenizer max length |
| `PF_EARLY_STOPPING_PATIENCE` | `2` | epochs with no eval_span_f1 improvement before stopping |
| `PF_RESUME_FROM_CHECKPOINT` | (unset) | resume from a specific checkpoint |
| `PF_EXTRA_TRAIN_DATASET` | (unset) | optional second train JSONL (concatenated) |

## Known limitations

- **`private_person` residual error** is dominated by KDPII's `PS_NICKNAME`
  tagging policy. ~40% of remaining person errors are online-handle-style
  strings (e.g., `탕비실맥심킹`, `퍼터요정`) that are tagged as
  `PS_NICKNAME → private_person` in KDPII gold. They aren't real personal
  names, but the current label policy treats them as such. Downstream
  redaction-style systems are unaffected (the entity is still detected);
  classification-style systems may want to post-classify handles separately.
- **Foreign names** (Western, Japanese, Arabic transliterations) are detected
  at lower rates due to limited training exposure in KDPII.
- **`private_address` boundaries** follow KDPII's split convention (each
  toponym component is a separate span). The benchmark applies whitespace
  trimming during scoring; raw model output may have leading-whitespace
  offsets that are cosmetic but noticeable in inference.

## Why MoE + LoRA

We tried full fine-tuning first. F1 on `private_person` and `private_address`
got stuck around 0.13–0.20 — *worse than the base model on some labels*.
Diagnosis: with 128 experts and top-4 routing, Korean-specific tokens are
routed to a small expert subset. Across 5–10 epochs each expert gets sparse
gradient updates relative to its parameter count, and the optimizer pulls
those experts away from their pretrained representations faster than it
teaches the new task. The base's pretrained Korean capability gets corrupted
before the new task is learned.

LoRA on attention only avoids this entirely: experts, FFN, embeddings, and
router stay exactly as the base shipped them; only attention re-routing and
the classifier head adapt. Result: F1 0.69 / 0.78 on the previously-stuck
labels, with every other label at or above ceiling.

Higher-rank LoRA (r=32) with `embed_tokens` added overfits without helping
the targeted labels. Remapping `PS_NICKNAME → personal_handle` shifts gold
counts in a way that doesn't net-help (private_person F1 down marginally,
personal_handle F1 down significantly).
