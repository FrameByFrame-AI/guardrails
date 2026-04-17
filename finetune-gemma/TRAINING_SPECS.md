# Gemma 4 E2B — Working Training Specs

Confirmed stable config for training Gemma 4 E2B on 2× NVIDIA RTX A5000 (24GB each).

## Why specific values matter

Gemma 4 E2B has a **262,144-token vocabulary**. The lm_head projection produces a
huge `batch × seq_len × 262144` bf16 tensor during loss computation. This is the
main memory bottleneck — not the model weights themselves.

Memory math (per GPU):
- Model (5.12B params × bf16) = **10.2 GB**
- LoRA optimizer state (rank 64) = **~1 GB**
- Activations + attention cache = **~4 GB**
- Logits tensor (`batch × seq_len × 262k × 2 bytes`) = **variable, grows fast**
- DDP overhead (grad broadcast buffers) = **~1 GB**

Leaving only **~7 GB** for the logits tensor. At seq_len=2048:
- batch=4  → `4 × 2048 × 262k × 2 = 4.3 GB` ✅ fits
- batch=8  → `8 × 2048 × 262k × 2 = 8.6 GB` ❌ OOM on GPU 1 (DDP)
- batch=16 → `16 × 2048 × 262k × 2 = 17.2 GB` ❌ OOM even single-GPU

## Verified working configurations

### 2-GPU DDP (fastest — ~4 hr/epoch on 67k records)

```yaml
NUM_GPUS: 2
BATCH_SIZE: 4              # per GPU
GRAD_ACCUM: 4
# Effective batch = 4 × 4 × 2 = 32
MAX_SEQ_LENGTH: 2048
LEARNING_RATE: 2e-4        # fresh training
EPOCHS: 2
```

### Single-GPU (fallback — slower but more memory headroom)

```yaml
NUM_GPUS: 1
BATCH_SIZE: 16             # no DDP overhead
GRAD_ACCUM: 4
# Effective batch = 16 × 4 = 64
MAX_SEQ_LENGTH: 2048
LEARNING_RATE: 2e-4
EPOCHS: 2
```

### Warm-start (fine-tune on augmented data)

```yaml
NUM_GPUS: 2
BATCH_SIZE: 4
GRAD_ACCUM: 4
MAX_SEQ_LENGTH: 2048
LEARNING_RATE: 5e-5        # 4× lower than fresh training
WARMUP_RATIO: 0.03         # shorter warmup (model already trained)
EPOCHS: 1                  # single pass
GEMMA_MODEL: /models/gemma4-guardrail-ko  # start from trained model, not raw base
```

## Common OOM failures

| Config | What happens |
|---|---|
| `BATCH_SIZE=8 NUM_GPUS=2` | OOM at lm_head softmax (rank 1 dies) |
| `BATCH_SIZE=16 NUM_GPUS=2` | OOM immediately |
| `BATCH_SIZE=16 NUM_GPUS=1` + seq_len=4096 | OOM |

**Rule**: `BATCH_SIZE × MAX_SEQ_LENGTH × 2e-6 GB` should be ≤ 6 for 24GB cards.

## Resume after OOM

HuggingFace Trainer auto-resumes if `--resume-from-checkpoint` points at a
checkpoint directory. Our `train.py` supports `--resume` flag that loads from
the most recent checkpoint in `OUTPUT_DIR`.

```bash
docker compose run --rm train --resume
```

## Training logs to expect

**Healthy training** (step 1):
```
loss=X.XX, grad_norm=0.0X, learning_rate=X.XXe-05, samples/s=20-30
```

**OOM warning sign**: `torch.OutOfMemoryError: ... 262144` — shrink batch.

## Related specs

- Qwen 3.5 0.8B: see `../finetune-qwen/TRAINING_SPECS.md` (if exists) — smaller vocab,
  higher batch possible.
- GLiNER2: see `../gliner2_korean_pii/TRAINING_SPECS.md` — very different profile
  (180M model, different memory issues from dynamic schema).
