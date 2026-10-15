---
license: apache-2.0
language:
  - ko
  - en
tags:
  - privacy-filter
  - pii-detection
  - token-classification
  - korean
  - lora
  - openai-privacy-filter
  - bioes
base_model: openai/privacy-filter
pipeline_tag: token-classification
---

# Privacy Filter — Korean

Korean fine-tune of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)
for span-level PII detection. Adapted via **LoRA** on attention projections only —
the base's sparse-MoE backbone (1.5B / 50M active params) is kept frozen, with
just **~614k trainable parameters** (~0.04% of the model).

**[Open Test Notebook](https://huggingface.co/FrameByFrame/privacy-filter-korean/blob/main/test_privacy_filter_ko.ipynb)** — load the model and run all examples interactively.

## Capabilities

| Category | Description | Example |
|---|---|---|
| `private_person` | Personal name (Korean / Western / handles) | 김민수, John Smith |
| `private_address` | Physical / postal address | 서울특별시 강남구 테헤란로 123 |
| `private_phone` | Phone number | 010-1234-5678 |
| `private_email` | Email address | minsu@example.com |
| `private_date` | Birthday / personally-identifying date | 1985년 3월 12일 |
| `private_url` | Personal URL | github.com/minsu |
| `account_number` | Bank, card, RRN, passport, etc. | 110-234-567890 |
| `personal_handle` | Username / handle | @minsu_dev |
| `ip_address` | IP address | 192.168.1.5 |

## Benchmark Results

Held-out KDPII Korean PII test set, span-level F1 — reconciled against a
single frozen test split (see [`TRAINING.md`](./TRAINING.md) for why this
supersedes the previously-published numbers, which were measured against a
test split that had silently changed):

| label | base | fine-tuned (`v10`) | Δ |
|---|---|---|---|
| `private_phone` | 0.65 | **1.00** | +0.35 |
| `private_url` | 0.21 | **1.00** | +0.79 |
| `account_number` | 0.31 | **0.98** | +0.67 |
| `private_email` | 0.83 | **0.95** | +0.12 |
| `private_date` | 0.00 | **0.90** | +0.90 |
| `private_address` | 0.00 | **0.69** | +0.69 |
| `private_person` | 0.07 | **0.76** | +0.69 |
| **Overall** | 0.31 | **0.89** | +0.58 |

This model card and the published weights currently reflect an earlier
checkpoint (`v6`); `v10_xlingual_sota` (trained on KDPII/RRN plus two
supplemental sources — non-Korean PII from ai4privacy-400k, and native
Korean synthetic chat-register data covering handle/nickname-rich domains)
is the new best-performing checkpoint and supersedes it — see
[`TRAINING.md`](./TRAINING.md) for the full per-candidate comparison,
including two rejected experiments (`v7`, a higher-rank LoRA config;
`ko_pii_klue`, a synthetic news-domain augmentation that regressed
accuracy) and why they didn't make the cut.

This fine-tune also outperforms
[OpenMed/privacy-filter-multilingual-v2](https://huggingface.co/OpenMed/privacy-filter-multilingual-v2)
(a 54-category, 16-language full fine-tune of the same base model) on Korean
PII by a wide margin — 0.89 vs. 0.28 overall F1 on this benchmark; OpenMed's
own model card names CJK languages as its weak point, and none of its seven
declared training data sources contain any Korean-language records at all.
See `TRAINING.md` for the label-remapping methodology used for a fair
comparison.

## Quick Start

### Install

> ⚠️ **Requires `transformers` 5.x (currently dev / from source).** The
> `openai_privacy_filter` architecture is *not* in any stable 4.x PyPI release.
> If you `pip install transformers` and load this model, you'll see
> `KeyError: 'openai_privacy_filter'`.

```bash
pip install --upgrade "git+https://github.com/huggingface/transformers.git" peft torch safetensors accelerate
```

The `--upgrade` flag is critical — without it, `pip install` is silently
no-op when an older transformers is already present.

After installing, **restart your Python runtime / kernel** so the new
transformers replaces any version pre-loaded into the process. Sanity-check:

```bash
python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; assert 'openai_privacy_filter' in CONFIG_MAPPING_NAMES, 'openai_privacy_filter missing — re-install transformers from source and restart runtime'"
```

If you're using Colab, the test notebook handles this automatically (auto-restart).

### Load Model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

MODEL_ID = "FrameByFrame/privacy-filter-korean"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
)
model.eval()
if torch.cuda.is_available():
    model.cuda()
```

`trust_remote_code=True` is required because Privacy Filter ships a custom
`OpenAIPrivacyFilterForTokenClassification` class (gpt-oss-style sparse MoE).

### Inference

The model emits per-token BIOES labels. The helper below decodes them into
character-offset spans with simple constrained logic:

```python
def extract_pii(text: str, max_length: int = 512):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()
    id2label = model.config.id2label

    spans = []
    active = None  # (label, start, end)
    for tok_idx, lid in enumerate(pred_ids):
        label = id2label[int(lid)]
        if label == "O":
            if active is not None:
                spans.append(active); active = None
            continue
        prefix, cat = label.split("-", 1)
        c_start, c_end = offsets[tok_idx]
        if prefix == "S":
            if active is not None: spans.append(active); active = None
            spans.append((cat, c_start, c_end))
        elif prefix == "B":
            if active is not None: spans.append(active)
            active = (cat, c_start, c_end)
        elif prefix in ("I", "E"):
            if active and active[0] == cat:
                active = (active[0], active[1], c_end)
            else:
                if active is not None: spans.append(active); active = None
                if prefix == "E":
                    spans.append((cat, c_start, c_end))
    if active is not None:
        spans.append(active)

    return [
        {"label": cat, "start": s, "end": e, "text": text[s:e].strip()}
        for cat, s, e in spans
        if text[s:e].strip()
    ]
```

### Test

#### Korean: name + phone + email
```python
>>> extract_pii("김민수의 전화번호는 010-1234-5678이고 이메일은 minsu@example.com입니다.")
[
  {"label": "private_person", "start": 0, "end": 3, "text": "김민수"},
  {"label": "private_phone",  "start": 12, "end": 25, "text": "010-1234-5678"},
  {"label": "private_email",  "start": 33, "end": 50, "text": "minsu@example.com"},
]
```

#### Korean: address + name
```python
>>> extract_pii("서울특별시 강남구 테헤란로 123에 사는 박지영씨에게 연락주세요.")
[
  {"label": "private_address", "start": 0, "end": 5, "text": "서울특별시"},
  {"label": "private_address", "start": 6, "end": 9, "text": "강남구"},
  {"label": "private_address", "start": 10, "end": 17, "text": "테헤란로 123"},
  {"label": "private_person",  "start": 22, "end": 25, "text": "박지영"},
]
```

> Note: the model follows KDPII's address convention where each toponym
> component is its own span. Most downstream redaction systems concatenate
> adjacent address spans.

#### Korean: form-style document
```python
>>> extract_pii('''고객 정보
... 이름: 이수진
... 생년월일: 1985년 3월 12일
... 주소: 부산광역시 해운대구 우동 1457
... 연락처: 010-9876-5432''')
[
  {"label": "private_person",  ..., "text": "이수진"},
  {"label": "private_date",    ..., "text": "1985년 3월 12일"},
  {"label": "private_address", ..., "text": "부산광역시"},
  {"label": "private_address", ..., "text": "해운대구"},
  {"label": "private_address", ..., "text": "우동 1457"},
  {"label": "private_phone",   ..., "text": "010-9876-5432"},
]
```

#### English: account + email
```python
>>> extract_pii("Wire to acct 110-234-567890, contact minsu@example.com")
[
  {"label": "account_number", "start": 13, "end": 26, "text": "110-234-567890"},
  {"label": "private_email",  "start": 36, "end": 53, "text": "minsu@example.com"},
]
```

### Redaction

Wrap the spans into a redactor:

```python
def redact(text: str, mask: str = "[REDACTED]") -> str:
    spans = extract_pii(text)
    spans.sort(key=lambda s: s["start"], reverse=True)
    out = text
    for s in spans:
        out = out[: s["start"]] + f"[{s['label'].upper()}]" + out[s["end"]:]
    return out

>>> redact("김민수님의 번호는 010-1234-5678입니다.")
"[PRIVATE_PERSON]님의 번호는 [PRIVATE_PHONE]입니다."
```

## Output Schema

Each detected entity is one dict:

| field | description |
|---|---|
| `label` | One of the 9 categories above |
| `start` | Character offset start (inclusive) |
| `end` | Character offset end (exclusive) |
| `text` | The matched substring |

## Training Details

| | |
|---|---|
| **Base model** | `openai/privacy-filter` (sparse MoE, 1.5B total / 50M active params, 128 experts top-4) |
| **Method** | LoRA r=16, alpha=32, dropout=0.05 on attention projections (`q/k/v/o_proj`); classifier head fully trainable; everything else frozen |
| **Trainable params** | ~614k (~0.04% of the model) |
| **Datasets** | KDPII (Korean, ~53k records, deterministic 5/5/90 test/val/train), `korean_rrn_synthetic` (train only), plus two supplements with address-labeled records excluded (span-convention conflict with KDPII): ai4privacy-400k non-Korean PII (`ko_pii_xlingual`) and native Korean synthetic chat-register data covering handle/nickname-rich domains (`ko_pii_sota_v2`) |
| **Optimizer** | AdamW, lr=5e-4, cosine schedule, warmup 0.1 |
| **Batch** | 128 effective (per-device batch and GPU count vary by hardware available at train time; LR/schedule are what matters for reproducing results, not the specific device split) |
| **Epochs** | 10, early stopping on `eval_span_f1` (patience 3) — this run used the full 10 epochs |
| **Sequence length** | 512 |
| **Precision** | bf16 mixed (saved as bf16 safetensors after `merge_and_unload`) |
| **Final eval span F1** | 0.848 (validation), 0.89 (held-out test, reconciled benchmark above) |

For full reproduction details, see [`TRAINING.md`](./TRAINING.md).

## ONNX Export

fp32 ONNX export works and is numerically verified (overall F1 within noise
of the PyTorch checkpoint) — see `finetune-privacy-filter/src/export_and_benchmark.py`
in the training repo for the exporter and the three modeling/tooling patches
that unblock it. fp16 and int8 precision reduction currently fail on
separate bugs in their respective conversion libraries when applied to this
model's MoE graph shapes; not yet resolved.

## Known Limitations

- **`private_person`/`personal_handle` residual error is dominated by
  recall**, not boundary or label-convention issues (an earlier read of the
  error breakdown, before a tooling bug in the analysis script was fixed,
  overstated boundary/merge problems — see `TRAINING.md`). Missed cases are
  mostly short (2-character) Korean given names in casual, vocative-style
  chat register without a surname or title, plus stylized online handles.
  Separately, ~40% of remaining `private_person` errors are KDPII's own
  `PS_NICKNAME` labeling policy tagging handle-like strings (e.g.
  `탕비실맥심킹`, `퍼터요정`) as `private_person`; downstream redaction is
  unaffected, classification systems may want to post-classify handles
  separately.
- **`private_address` recall gap** is mostly administrative-division names
  that are also common Korean homonyms (`인도` = "India" or "sidewalk", `공주`
  = "Gongju" or "princess") — a genuine disambiguation problem, not a
  boundary issue.
- **`private_address` boundaries** follow KDPII's split convention (each
  toponym component is a separate span). Production redactors typically
  concatenate adjacent address spans during post-processing.
- Raw model output may have leading/trailing whitespace in span offsets;
  the `extract_pii` helper above strips them via `text.strip()` on the slice.

## License

Apache 2.0 (inherited from base
[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)).

## Citation

If you use this model:

```bibtex
@misc{framebyframe-privacy-filter-korean-2026,
  title  = {Privacy Filter Korean: LoRA fine-tune of OpenAI Privacy Filter for Korean PII},
  author = {FrameByFrame},
  year   = {2026},
  url    = {https://huggingface.co/FrameByFrame/privacy-filter-korean}
}
```
