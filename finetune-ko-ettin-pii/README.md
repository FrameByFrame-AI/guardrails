# finetune-ko-ettin-pii

Fine-tune the **ko-ettin** Korean foundation encoder (from `ko-ettin-foundation`,
stage-3 continued-MLM) as a **Korean PII token-classifier**, and benchmark it
**head-to-head against the privacy-filter (opf) fine-tune** on the *identical* data,
label space, and span metric.

This is the apples-to-apples test of the question: *does a Korean-native encoder
(−58% fertility, 5B-token Korean MLM) beat the English-tokenizer opf backbone on
Korean PII?*

## What's reused (so the comparison is fair)

Everything except the backbone is shared with `../finetune-privacy-filter`:

| Shared artifact | Path |
|---|---|
| Train / val / test splits | `../finetune-privacy-filter/data/generated/ko_pii_opf_v4/{train,validation,test}.jsonl` |
| Label space (9 PII classes, BIOES) | `.../ko_pii_opf_v4/label_space.json` |
| Span metric (char-offset exact match) | `src/benchmark_pii.py` (copied verbatim) |
| Trainer + BIOES projection + span-F1 | `src/train_pii.py` (copied; only the opf-specific classifier-head warm-start is made optional) |

Split sizes: **train 41,197 · validation 2,227 · test 2,252** (seed=42, KDPII + Korean-RRN synthetic).

## The bar to beat

opf fine-tuned on this exact data (`ko_pii_hf_ddp_v6_lora`) scored, on the v4 test split:

- **`test_span_f1 = 0.8780`** (trainer metric, exact BIOES span match)
- benchmark harness (char-offset): overall span **F1 ≈ 0.869–0.877**

opf leaks heavily on **boundary errors** for Korean `private_person` (63 BOUNDARY vs 47
EXACT) and `private_address` (51 BOUNDARY vs 6 EXACT) — the classic symptom of an
English tokenizer fragmenting Hangul. ko-ettin's native Korean subwords should convert
many BOUNDARY→EXACT; that's the hypothesized source of any gain.

## The only backbone difference

`train_pii.py` is the privacy-filter trainer with one change: `--skip-classifier-remap`.
opf ships a pretrained PII head (`.score`) whose rows it warm-starts; a fresh encoder
(ko-ettin / ModernBERT) has no such head, so the token-classification head is randomly
initialized (correct for a fresh backbone). Full fine-tune (no LoRA) — the encoder is
only 150M.

## Run

Uses **prebuilt images** (no build): `ml-base` for training / ko-only benchmark
(proven for ko-ettin/ModernBERT), and the existing known-good
`finetune-privacy-filter-benchmark-pii` image for the head-to-head (it registers
opf's custom arch and also loads ModernBERT). Do **not** rebuild a fresh
transformers-main image — it currently breaks on a `GenerationMixin` import.

```bash
cd finetune-ko-ettin-pii

# 1) Fine-tune ko-ettin (defaults to the stage-3 checkpoint, GPU 0).
#    Run AFTER stage-3 finishes, or set KOE_GPU=1 to use the other card.
KOE_GPU=0 docker compose run --rm train-pii

# 2) Head-to-head vs opf on the same test split (one process, same metric).
KOE_GPU=0 docker compose run --rm benchmark-pii
#    -> data/benchmark_results/ko_ettin_vs_opf_pii.json

# (optional) ko-ettin only, no opf env needed:
docker compose run --rm benchmark-ko-only
```

### Knobs

| var | default | meaning |
|---|---|---|
| `KOE_GPU` | `0` | GPU id (use `1` while stage-3 still holds GPU 0) |
| `KOE_CHECKPOINT` | stage-3 model | backbone to fine-tune |
| `KOE_DATA` | `/gen/ko_pii_opf_v4` | shared split dir (mounted from sibling) |
| `KOE_OUTPUT` | `…/ko-ettin-encoder-150m-ko-pii` | fine-tuned model out |
| `OPF_CHECKPOINT` | `/opf_ckpts/ko_pii_hf_ddp_v6_lora` | opf model to compare against |
| `KOE_EPOCHS` / `KOE_BS` / `KOE_LR` / `KOE_MAX_LENGTH` | 5 / 64 / 3e-5 / 256 | training |

## Honest caveat

KDPII is the only *real* Korean PII source (RRN is synthetic); the 2,252-record test
set is modest, so read per-label F1, not just the headline. A win on `private_person` /
`private_address` (boundary-sensitive) is the most meaningful signal here.
