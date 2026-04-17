# Gemma 4 E2B Guardrail Plan

## Context

We reviewed Unsloth's `Gemma4_(E4B)-Vision.ipynb` notebook first:

- It uses `FastVisionModel.from_pretrained(...)`, not `FastLanguageModel`.
- The example model is `unsloth/gemma-4-E4B-it`.
- The notebook loads the model in `4bit` and enables `use_gradient_checkpointing="unsloth"`.
- LoRA is added with `FastVisionModel.get_peft_model(...)`.
- The example enables both language and vision fine-tuning and uses `target_modules="all-linear"`.
- The sample task is multimodal OCR, so the dataset format is image + conversation, not text-only JSONL.

This matters because our current guardrail training data is text-only. We should not copy the notebook literally.

## What We Will Do

### Phase 1: Keep Workspaces Separate

- Keep `qwen-guardrail-finetune/` as the Qwen workspace.
- Use `gemma-guardrail-finetune/` as the Gemma workspace.
- Keep separate container names, output directories, and prepared dataset directories.

### Phase 2: Build a Text-First Gemma E2B Baseline

Use Gemma 4 E2B for the first guardrail baseline, but keep it text-only initially.

Decisions:

- Base model: `unsloth/gemma-4-E2B-it` if we want to stay closest to the notebook path.
- Fallback model id: `google/gemma-4-E2B-it` only if Unsloth compatibility is confirmed in this repo.
- Trainer class: `FastLanguageModel` for the initial text-only baseline.
- Thinking: controlled explicitly via the system prompt.
- Max sequence length: `2048`.
- Precision:
  - First try `16bit LoRA` if it fits comfortably on `2xA5000`.
  - Fall back to `4bit QLoRA` only if memory requires it.
- LoRA:
  - start with `r=16`, `alpha=16`, `dropout=0`
  - start with standard text linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

Why:

- The notebook proves the Unsloth Gemma 4 stack is ready.
- Our immediate bottleneck is training speed and precision-first guardrail behavior, not multimodal training yet.
- A text-first baseline is simpler, faster to debug, and directly comparable to the current Qwen path.

### Phase 2A: Thinking-Mode Contract

Gemma 4 uses explicit thinking control:

- thinking enabled:
  - prepend `<|think|>` to the system prompt
- thinking disabled:
  - do not prepend `<|think|>`

Output behavior from the docs:

- thinking enabled:
  - model may emit `<|channel>thought ... <channel|>` before the visible answer
- thinking disabled:
  - larger models may still emit an empty thought block before the visible answer

Implications for our training setup:

- We should not mix incompatible thinking formats inside one training run.
- We should keep the dataset in standard `system/user/assistant` roles.
- We should choose one of these two training modes per run:
  - `thinking-on` dataset:
    - system prompt starts with `<|think|>`
    - assistant target includes the thought channel plus final answer
  - `thinking-off` dataset:
    - system prompt does not include `<|think|>`
    - assistant target contains only the final visible answer

Recommended first path:

- train `thinking-off` first for the production guardrail path
- keep inference cheap by default
- add a second `thinking-on` experiment only if we need a higher-quality fallback path for harder cases

Reason:

- the Unsloth docs explicitly say Gemma 4 thinking is format-sensitive
- they also explicitly say not to mix incompatible thinking formats in one dataset
- for most production assistants, their simplest recommendation is to fine-tune on the final visible answer only

### Phase 3: Reduce Dataset Size Before Training

The full current dataset is too large for fast iteration.

Current distribution:

- moderation: `272692`
- pii-filter: `105902`
- rules-based-protections: `5274`
- safety-classifier: `5219`
- output-validation: `50`

Preparation plan:

- cap `moderation` to `20k`
- cap `pii-filter` to `20k`
- keep the smaller classes intact for now
- prepare separate datasets per thinking mode
  - first run: `think_mode=off`
  - optional later run: `think_mode=required`
- keep prepared data in a Gemma-specific directory

This gives us a much smaller first-pass training set while preserving the rare safety classes.

### Phase 4: Optimize for Precision, Not Generic Generation

The guardrail objective is not open-ended generation. The model should emit a compact JSON verdict.

We should bias the first Gemma version toward:

- minimum false positives
- clean `blocked` decisions
- correct `type`
- useful `topics`

We should not optimize for:

- long chain-of-thought output
- verbose reasons
- large generation budgets

Inference budget policy:

- low-latency path:
  - no `<|think|>` in the system prompt
  - short generation budget
  - preferred default for guardrails
- higher-budget path:
  - prepend `<|think|>` in the system prompt
  - allow the model to produce the thought channel before the final verdict
  - use only if we decide the extra latency is justified

### Phase 5: Add Prompt Injection Data Deliberately

To improve prompt-injection handling without increasing false positives:

- add prompt-injection positives
- add hard benign negatives modeled after `NotInject`
- add Korean prompt-injection variants
- add domain-specific benign prompts containing trigger words like `system`, `instruction`, `ignore`, `override`, `policy`, `token`, `prompt`

This should be treated as a precision-focused extension to the main guardrail dataset, not as a separate model.

### Phase 6: Add Vision Only After Text Baseline Works

If we later need image and PDF guardrails:

- switch from `FastLanguageModel` to `FastVisionModel`
- follow the notebook structure more directly
- support image-containing conversation records
- render PDFs to page images for multimodal inspection
- optionally also extract text from PDFs and compare text-only vs vision-assisted performance

Important:

- the reviewed notebook is a valid template for the multimodal phase
- it is not the right starting point for our current text-only guardrail dataset

## Immediate Next Steps

1. Finish isolating `gemma-guardrail-finetune/` with Gemma-specific defaults and service names.
2. Point the Gemma workspace to `unsloth/gemma-4-E2B-it`.
3. Reuse the current text dataset, but prepare a capped Gemma-specific training split.
4. Run a small smoke train on `2xA5000`.
5. Measure:
   - time per step
   - total steps per epoch
   - precision / false positives on benign prompts
   - prompt-injection recall on held-out tests
6. Only after that, decide whether to add the multimodal path from the Vision notebook.

## Open Questions

- Should the Gemma workspace default to `unsloth/gemma-4-E2B-it` or `google/gemma-4-E2B-it`?
- Should the first run use `16bit LoRA` or `4bit QLoRA`?
- Do we want a second, separate `thinking-on` adapter after the `thinking-off` production baseline?
- Do we want the first Gemma milestone to be text-only, or do we want image/PDF support in v1?

## Reference

- Unsloth notebook reviewed: `Gemma4_(E4B)-Vision.ipynb`
  - GitHub: `unslothai/notebooks`
  - Colab: `https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb`
