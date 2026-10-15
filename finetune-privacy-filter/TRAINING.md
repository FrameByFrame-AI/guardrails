# Privacy Filter — Korean PII fine-tune

LoRA fine-tune of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)
on Korean PII data (KDPII + `korean_rrn_synthetic`), targeting span-level
detection of personal identifiers in Korean chat-style and form-style text.

## Reproducibility note (2026-07-01)

The numbers below (and the ones published on the HF model card) were measured
against whatever `ko_pii_opf_v4/test.jsonl` contained *at benchmark time* —
and that file was regenerated in place at least once between the v6 and v8
checkpoint runs (`convert_korean_pii_to_opf.py` had no overwrite guard), so
its gold-span count silently changed from 494 to 434. v6/v7/v8 were each
benchmarked against a slightly different test set and were never directly
comparable, despite living side by side in `data/checkpoints/`.

Two fixes landed for this:
- `convert_korean_pii_to_opf.py` now refuses to overwrite a non-empty
  `--output-dir` unless `--force` is passed, and records a sha256 hash of its
  input files in `conversion_summary.json` so future regenerations are
  auditable.
- `analyze_errors.py` (the per-category error breakdown tool) did not apply
  the same whitespace/punctuation boundary trimming as
  `benchmark_pii_heldout.py` before classifying a prediction as EXACT vs
  BOUNDARY. This made its breakdowns badly overstate real errors — e.g.
  `private_email` looked like only 9/38 exact matches (implying a boundary
  problem) when the benchmark script's own F1 said 0.94, because 29 of those
  "boundary" mismatches were just the model including one leading whitespace
  character, which the benchmark's normalization already treats as correct.
  Both scripts now share `span_matching.py`'s trimming logic.

The table below is a **reconciled** re-run of v6/v7/v8 against the current
(434-gold-span) `ko_pii_opf_v4/test.jsonl` in one sitting, `--max-length 512`
throughout, so it's the first apples-to-apples comparison of the three:

| label | base | v6 (published) | v7 (rejected, see below) | v8 |
|---|---|---|---|---|
| `private_phone` | 0.65 | 1.00 | 1.00 | 1.00 |
| `private_url` | 0.21 | 1.00 | 1.00 | 1.00 |
| `private_email` | 0.83 | 1.00 | 0.94 | 0.94 |
| `account_number` | 0.31 | 0.98 | 0.94 | 0.95 |
| `private_date` | 0.00 | 0.90 | 0.94 | 0.92 |
| `private_address` | 0.00 | 0.77 | 0.77 | 0.77 |
| `private_person` | 0.07 | 0.58 | 0.56 | **0.66** |
| **Overall span F1** | 0.31 | 0.854 | 0.831 | **0.874** |

v7 (r=32, α=64, +`embed_tokens`, 15 epochs) is confirmed worse than v6 on
every metric that matters (`private_person`, overall F1) — consistent with
the "Why MoE + LoRA" section below; its checkpoint is kept only as a record
of a rejected experiment, not a candidate.

v8 used **v6's exact recipe**, re-run — and came out ahead of v6 on this
frozen test set (overall F1 0.874 vs 0.854, `private_person` 0.66 vs 0.58).
This is run-to-run variance (early-stopping epoch, MoE routing), not a config
change. v8 was the best checkpoint at this point, superseded below by
`ko_pii_hf_ddp_v9_xlingual`; the HF model card still reflects v6 and needs
updating.

## v9 candidates: quality push beyond v8 (2026-07-04)

`analyze_errors.py`'s boundary-trim fix (above) also corrected the error
diagnosis it's used for: with the fix applied, v8's real residual errors are
**MISSED spans (recall), not boundary/merge convention issues** —
`private_person`/`personal_handle` miss ordinary short (2-char) Korean given
names in casual/vocative chat register and stylized handles;
`private_address` misses a handful of homonym-ambiguous place names (`인도` =
"India" or "sidewalk", `공주` = "Gongju" or "princess"). A previously-planned
fix (merging KDPII's per-toponym-split address spans into one) was dropped
once the corrected breakdown showed only 5/65 genuine boundary mismatches —
not worth building.

`data/generated/` already contained two unused dataset variants prepared for
training but never actually trained on: `ko_pii_xlingual` (v4 + ai4privacy-400k,
186k records, non-Korean names/PII) and `ko_pii_klue` (v4 + KLUE-NER
news-domain + synthetic augmentation, 106k records) — both aimed roughly at
the person/handle recall gap. Both label `private_address` as one merged
span (KDPII splits per toponym component), a real convention conflict with
v4, so records containing a `private_address` label were filtered out of
each before concatenating with v4 for training (dropping the label instead
risks teaching a false negative — the address text still there, just
unlabeled). `ko_pii_klue`'s addresses are also geographically inconsistent
random component combinations (e.g. `광주광역시 서산시` — not a real
hierarchy), confirming synthetic generation rather than real extracted
spans; fine for span-shape learning, just not semantically meaningful.

Both candidates use v6/v8's exact recipe (LoRA r=16, α=32, 10 epochs,
early-stopping patience 3) — only the training data changes, to isolate its
effect. Numbers below are a single reconciled run of all four models
together against the frozen test set:

| label | base | v8 (baseline) | v9 + klue | v9 + xlingual |
|---|---|---|---|---|
| `private_person` | 0.07 | 0.6629 | 0.6524 | **0.7293** |
| `private_address` | 0.00 | **0.7717** | 0.6557 | 0.7097 |
| `account_number` | 0.31 | 0.9547 | **0.9710** | 0.9544 |
| `private_email` | 0.83 | 0.9351 | 0.9737 | 0.9474 |
| `private_date` | 0.00 | 0.9200 | 0.9362 | 0.9200 |
| `private_phone` / `private_url` | 0.65 / 0.21 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| **Overall span F1** | 0.31 | 0.8737 | 0.8618 | **0.8791** |

**`ko_pii_klue` is rejected** — it regressed both the metrics it was meant to
help (`private_person` down slightly, `private_address` down sharply,
-0.11 F1 vs v8) without improving recall on the target failure mode. Most
likely explanation: its synthetic news-domain/templated text is
distributionally different enough from KDPII's real chat register that
training on it pulls the model away from the actual test distribution
rather than reinforcing the weak spot.

**`ko_pii_xlingual` is the new best checkpoint.** `private_person` improved
meaningfully (0.663 → 0.729, +0.066) — the hypothesis that additional name
diversity would help held, though via general recall improvement rather
than specifically fixing the short-Korean-given-name failure mode originally
observed (the confirmed error-analysis miss cases were all native Korean
names, but ai4privacy's broader multi-lingual name distribution apparently
still generalizes useful signal). `private_address` gave back some ground
versus v8 (0.772 → 0.710) — expected, since xlingual's address labels use
the merged-span convention filtered out for training but the underlying
person/handle records still shift the overall example distribution — but
the net effect across all labels is positive.

## v10: adding ko_pii_sota_v2 (2026-07-04)

`data/generated/` also had `ko_pii_sota`/`ko_pii_sota_v2` (120-123k records,
generated but never trained on) sitting unused — native Korean synthetic
data from `finetune-ko-ettin-pii/src/gen_spy_korean.py` (a local-LLM
generator covering 18 domains, including SNS/gaming/community text
specifically for handle/nickname coverage), unlike `ko_pii_klue`'s
news-domain-derived text. Same address-record-filtering applied (same
merged-span-convention conflict, same geographically-inconsistent synthetic
addresses — e.g. `강남구` labeled under `부산광역시`, though 강남구 is a
Seoul district — confirms the same synthetic-generation family as `klue`,
fine for span-shape learning not semantic accuracy).

v10 combines v9's winning `ko_pii_xlingual` supplement with address-filtered
`ko_pii_sota_v2` (v4 + both supplements = 280,837 train records, ~1.6× v9's
size), same recipe, ran the full 10 epochs (no early stop this time):

| label | v9 (xlingual only) | v10 (xlingual + sota_v2) |
|---|---|---|
| `private_person` | 0.7293 | **0.7624** |
| `personal_handle` | 0.6484 | **0.6842** |
| `account_number` | 0.9544 | **0.9793** |
| `ip_address` | 0.9412 | **1.0000** |
| `private_address` | **0.7097** | 0.6885 |
| `private_date` | **0.9200** | 0.9020 |
| `private_email` | 0.9474 | 0.9487 |
| `private_phone` / `private_url` | 1.00 / 1.00 | 1.00 / 1.00 |
| **Overall span F1 (7-label, base-comparable)** | 0.8791 | **0.8897** |
| **Overall span F1 (all 9 labels)** | 0.8404 | **0.8550** |

**`ko_pii_hf_ddp_v10_xlingual_sota` is the new recommended checkpoint**,
superseding v9. `personal_handle` and `private_person` both improved
further (confirming `sota_v2`'s handle-rich synthetic domains helped as
intended), `account_number` and `ip_address` also improved, at a small cost
to `private_address`/`private_date` — net positive across the full label
set.

## Competitive benchmark vs. OpenMed/privacy-filter-multilingual-v2 (2026-07-04)

[OpenMed/privacy-filter-multilingual-v2](https://huggingface.co/OpenMed/privacy-filter-multilingual-v2)
is a full fine-tune of the same `openai/privacy-filter` base across 16
languages and 54 PII categories; its own model card admits "CJK languages
(Japanese, Korean, Chinese) ... remain the main bottleneck." `benchmark_pii_heldout.py`
gained generic `--label-remap-json` support (a per-model category remap,
dropping categories with no correspondence) specifically to compare it
fairly against this project's 9-category Korean schema — see
`data/remaps/openmed_multilingual_v2_to_ko9.json` for the mapping.

Result on the frozen `ko_pii_opf_v4/test.jsonl` (re-confirmed against a local
copy of the OpenMed checkpoint, `~/llm_models/OpenMed-privacy-filter-multilingual-v2`,
avoiding the HF hub download):

| model | overall span F1 | precision | recall |
|---|---|---|---|
| base (`openai/privacy-filter`) | 0.3070 | 0.3148 | 0.2995 |
| **OpenMed/privacy-filter-multilingual-v2** | 0.2802 | 0.2035 | 0.4493 |
| **v10 (this project, current best)** | **0.8897** | **0.8970** | **0.8825** |

OpenMed's model is barely above the untouched base on this benchmark and
massively over-predicts (958 predicted spans vs. 434 gold — precision 0.204),
consistent with its own documented CJK weakness. `private_person` F1 0.0514
(414 predicted spans vs. 92 gold). Checked OpenMed's declared training
sources (`ai4privacy` 200k/400k/500k/1m, `nvidia/Nemotron-PII`,
`gretelai/gretel-pii-masking-en-v1`, `piimb/privy`) — **none contain any
Korean-language records at all**, which is presumably why.

## Independent eval set: transliterated-foreign-name gap found, out of scope (2026-07-04)

A hand-authored external eval log (50 Korean PII redaction test cases,
`Input_Text` → `Expected_Output` with `[LABEL]` placeholders) was converted
to OPF format via `src/convert_pii_eval_log_to_opf.py` (character-offset
spans recovered by aligning literal segments around each placeholder; 42/50
rows converted cleanly, 8 skipped — 1 to Korean particle-allomorphy grammar
changes around the redaction, 7 to a lower-quality tail of the source CSV
using inconsistent ad-hoc `[S-word]` tags on garbled English-transliterated
text, one of which flagged a discount percentage as PII). Saved as
`data/generated/pii_eval_log_v1/`.

v10 scores only **0.24 F1** on this set (vs. 0.89 on the KDPII-derived test
set), though still ahead of base (0.11) and OpenMed (0.14). Root cause,
confirmed via `analyze_errors.py`: this eval set tests **Indian names and
places transliterated phonetically into Hangul** (라굴/Raghul, 아룬/Arun,
프라부/Prabhu, 벵갈루루 카르나타카/Bangalore-Karnataka), with phone/account/IP
numbers **spelled out as Korean number-words** instead of digits — a
distribution KDPII (the sole source of this project's real Korean training
signal) never contains. Faced with these out-of-vocabulary Hangul strings,
the model hallucinates `private_person` on random fragments (샌디스크/SanDisk,
첸나/Chennai-fragment) — pattern-matching "unusual token sequence" to
"probably a name" rather than a real detection failure in-distribution.

**Decision: out of scope for now** — this doesn't reflect expected
production traffic. Not pursued further, but the eval set and conversion
script are kept for whenever this needs revisiting (e.g. if transliterated
foreign names become a real requirement, synthetic training data covering
that + spelled-out numbers would be the next step, followed by a v11
candidate on the same recipe).

## ONNX export: unblocked at fp32 (2026-07-04)

ONNX export was previously fully blocked (see `src/export_and_benchmark.py`'s
module docstring for the complete history). Three sequential, process-local
patches unblock it:

1. A `functools.wraps`-based signature mismatch across several decorated
   `forward` methods that broke `torch.export`'s argument binding (fixed by
   synthesizing a real, matching bytecode-level signature).
2. `optimum-onnx`'s vendored masking shim expecting a `cache_position`
   argument that current `transformers` no longer passes (synthesized from
   the modern `q_length`/`q_offset` args instead).
3. The default eager MoE experts forward's data-dependent
   `for expert_idx in expert_hit:` loop, untraceable by `torch.export` in
   principle — worked around by switching to `transformers.integrations.moe`'s
   `batched_mm` vectorized experts backend for export only (no data-dependent
   control flow, mathematically equivalent).

**fp32 ONNX export works and is numerically verified**: two independent runs
against **v8** scored overall F1 0.8388 and 0.8396 against baseline bf16
PyTorch's 0.8367 — matching within noise (the export's own
`PF_EXPORT_MAX_F1_DROP` gate flagged one run as FAIL over a single small
class, `private_email` @ n=38 gold spans, which independently showed the
same ~0.04 swing between a CPU and GPU benchmark run of the *identical*
PyTorch checkpoint — known measurement noise at this sample size, not an
ONNX-specific regression). The export mechanism (all three patches) is
checkpoint-agnostic, but **not yet re-verified against v9/v10** — worth
re-running `export-pii` against whichever checkpoint is actually deployed.

**fp16 and int8 are still blocked**, each by a separate, genuine bug in its
respective third-party conversion tool when applied to this model's MoE
graph shapes (not fixable with existing flags) — see the `STATUS` notes on
`export_fp16`/`export_int8` in `src/export_and_benchmark.py` for the exact
errors and what was ruled out.

## Final results (held-out test, span F1) — as published (v6, HF model card)

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

**These specific numbers are not reproducible against the current
`ko_pii_opf_v4/test.jsonl`** per the reproducibility note above — see the
reconciled table for current, trustworthy numbers.

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
  benchmark_pii_heldout.py       # base-vs-finetuned span benchmark (PT + ONNX wrapper, CPU/GPU speed)
  analyze_errors.py              # per-category EXACT/BOUNDARY/MISSED breakdown
  span_matching.py               # shared boundary-trim normalization (benchmark + analyze_errors)
  export_and_benchmark.py        # ONNX export pipeline (BLOCKED — see below)
data/                            # gitignored: generated/, checkpoints/, benchmark_results/
docker-compose.yml               # convert-pii, train-pii, benchmark-pii, export-pii services
Dockerfile                       # CUDA 12.4 + torch 2.9 + transformers (HEAD) + peft + onnxruntime-gpu
```

### ONNX export — blocked

`src/export_and_benchmark.py` and the `export-pii` compose service are
end-to-end scaffolding for fp16/int8 ONNX export with F1-deviation gating
and CPU/GPU speed benchmarks. They are kept as a record of what's needed
once the modeling code becomes exportable.

The custom `OpenAIPrivacyFilterForTokenClassification` (sparse MoE,
gpt-oss-style) does not currently round-trip through the public toolchain:

- **Legacy tracer path** (`torch.onnx.export` / optimum non-dynamo) →
  `sdpa_mask_without_vmap() missing 1 required positional argument:
  'cache_position'`. optimum-onnx PR #113's transformers-5.x compat shim
  doesn't reach this path.
- **Dynamo path** (`torch.export.export` / optimum `dynamo=True`,
  needs `torch>=2.9`) → `TypeError: missing a required argument: 'self'`
  on the decorator-wrapped MoE forward. Model-source issue, not tooling.

Patching `modeling_openai_privacy_filter.py` to be `torch.export`-compatible
is the unblocker. The base model's published ONNX exports were almost
certainly produced with internal tooling.

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

Corrected via `analyze_errors.py`'s error-bucket breakdown (v8, reconciled
test set) after fixing its boundary-normalization bug — see the
reproducibility note above. Boundary/merge issues are **not** the dominant
error mode; recall (MISSED spans) is:

- **`private_person`/`personal_handle` recall gap (20 + 23 missed spans).**
  Mostly ordinary short (2-character) Korean given names used in casual,
  vocative-style chat register without a surname or title (e.g. `하준`,
  `민주`, `연지`, `혜원`), plus stylized handles (`flip2`, `탕비실맥심킹`).
  This is **not** a foreign-name problem — every missed example observed is
  native Korean. `private_person` residual error is also partly KDPII's
  `PS_NICKNAME` tagging policy: ~40% of remaining errors are online-handle
  strings (e.g. `탕비실맥심킹`, `퍼터요정`) tagged `PS_NICKNAME →
  private_person` in gold. They aren't real personal names, but current label
  policy treats them as such; downstream redaction is unaffected (the entity
  is still detected), classification-style systems may want to post-classify
  handles separately. A smaller but real slice is genuine `private_person` ↔
  `personal_handle` label confusion at inference time.
- **`private_address` recall gap (9 missed spans)**, dominated by
  administrative-division names that are also common Korean homonyms —
  `인도` (India *or* "sidewalk"), `공주` (Gongju city *or* "princess") — a
  genuine disambiguation problem, not a span-boundary or merge-convention
  issue. (KDPII's per-toponym-component split convention was previously
  suspected to dominate `private_address` errors; the corrected breakdown
  shows only 5/65 genuine boundary mismatches, so that convention is not
  worth changing.)
- **Foreign names** (Western, Japanese, Arabic transliterations) are still
  expected to be detected at lower rates due to limited training exposure in
  KDPII, though no foreign-name misses showed up in the reconciled sample —
  this is inferred from training-data composition, not directly observed.

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
