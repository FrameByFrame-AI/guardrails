#!/usr/bin/env python3
"""Shared helpers for preparing and loading Qwen guardrail datasets."""

import json
from collections import Counter
from pathlib import Path
import random

from datasets import Dataset, load_from_disk


def extract_target_payload(record):
    assistant_content = record["conversations"][2]["content"].strip()
    if "</think>" in assistant_content:
        assistant_content = assistant_content.split("</think>", 1)[1].strip()
    return json.loads(assistant_content)


def extract_target_type(record):
    try:
        payload = extract_target_payload(record)
        return payload.get("type", "unknown")
    except (IndexError, json.JSONDecodeError, KeyError):
        return "unknown"


def extract_target_blocked(record):
    try:
        payload = extract_target_payload(record)
        return bool(payload.get("blocked"))
    except (IndexError, json.JSONDecodeError, KeyError):
        return False


def summarize_records(records):
    counts = Counter(extract_target_type(record) for record in records)
    print("Target distribution:")
    for guardrail_type, count in counts.most_common():
        print(f"  {guardrail_type:<24} {count:>7}")


def get_text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def tokenize_text(text_tokenizer, text):
    if hasattr(text_tokenizer, "encode"):
        return text_tokenizer.encode(text, add_special_tokens=False)
    tokenized = text_tokenizer(text, add_special_tokens=False)
    return tokenized["input_ids"]


def parse_type_caps(type_cap_args):
    caps = {}
    for raw in type_cap_args or []:
        guardrail_type, sep, count = raw.partition("=")
        if not sep:
            raise ValueError(f"Invalid --type-cap value: {raw}. Expected TYPE=COUNT.")
        guardrail_type = guardrail_type.strip()
        if not guardrail_type:
            raise ValueError(f"Invalid --type-cap value: {raw}. Missing TYPE.")
        caps[guardrail_type] = int(count)
    return caps


THINK_PREFIX = "<|think|>\n"

# Types that benefit from reasoning traces (harder classification tasks)
THINKING_CANDIDATE_TYPES = {"pii-filter", "safety-classifier", "output-validation"}


def is_thinking_candidate(record):
    """Decide if a record is a good candidate for thinking mode.
    Prioritizes: PII, safety, output-validation, and blocked moderation."""
    guardrail_type = extract_target_type(record)
    if guardrail_type in THINKING_CANDIDATE_TYPES:
        return True
    if guardrail_type == "moderation" and extract_target_blocked(record):
        return True
    return False


def build_gemma_thinking_trace(record):
    """Build a Gemma-format thinking trace from the record's existing reasoning."""
    payload = extract_target_payload(record)
    blocked = payload.get("blocked", False)
    guardrail_type = payload.get("type", "unknown")
    topics = payload.get("topics", [])
    entities = payload.get("entities", [])

    lines = []
    lines.append(f"Classifying input for content safety.")

    if guardrail_type == "pii-filter":
        lines.append("Scanning for personally identifiable information.")
        if blocked and entities:
            labels = list(set(e["label"] for e in entities[:5]))
            lines.append(f"Found PII entities: {', '.join(labels)}.")
            for e in entities[:3]:
                lines.append(f'  - "{e["form"]}" is a {e["label"]}.')
        elif not blocked:
            lines.append("No PII patterns found in the input.")
    elif guardrail_type == "safety-classifier":
        lines.append("Checking for prompt injection, jailbreak attempts, or other safety threats.")
        if blocked and topics:
            lines.append(f"Detected threat categories: {', '.join(topics)}.")
            lines.append("This input attempts to manipulate or bypass safety controls.")
        elif not blocked:
            lines.append("No safety threats detected. Input appears benign.")
    elif guardrail_type == "output-validation":
        lines.append("Evaluating whether this request asks for unsafe or misleading output.")
        if blocked and topics:
            lines.append(f"Risk categories: {', '.join(topics)}.")
        elif not blocked:
            lines.append("Requested output appears safe.")
    else:  # moderation
        lines.append("Checking for hate speech, harassment, profanity, or offensive content.")
        if blocked and topics:
            lines.append(f"Detected harmful categories: {', '.join(topics)}.")
        elif not blocked:
            lines.append("Content appears safe and within policy.")

    verdict = "BLOCKED — content violates policy." if blocked else "ALLOWED — no violations detected."
    lines.append(f"Verdict: {verdict}")

    return "\n".join(lines)


def convert_to_gemma_format(conversations, thinking_on):
    """Convert conversations to Gemma format, optionally with thinking."""
    conversations = [dict(msg) for msg in conversations]
    payload_str = conversations[2]["content"]

    # Extract JSON payload (strip Qwen <think> blocks if present)
    if "</think>" in payload_str:
        payload_str = payload_str.split("</think>", 1)[1].strip()

    if thinking_on:
        # Add <|think|> to system prompt
        conversations[0]["content"] = THINK_PREFIX + conversations[0]["content"]
        # Wrap reasoning in Gemma thought channel
        return conversations, payload_str
    else:
        # No thinking, just the JSON
        conversations[2]["content"] = payload_str
        return conversations, None


def maybe_strip_think(record, think_mode):
    """Prepare conversations for the given think_mode.
    For 'mixed', the caller handles per-record decisions separately."""
    if think_mode == "off":
        conversations = [dict(message) for message in record["conversations"]]
        payload = extract_target_payload(record)
        conversations[2]["content"] = json.dumps(payload, ensure_ascii=False)
        return conversations
    elif think_mode == "required":
        # Convert Qwen <think> to Gemma <|channel>thought format
        conversations = [dict(message) for message in record["conversations"]]
        payload = extract_target_payload(record)
        thinking_trace = build_gemma_thinking_trace(record)
        conversations[0]["content"] = THINK_PREFIX + conversations[0]["content"]
        conversations[2]["content"] = (
            f"<|channel>thought\n{thinking_trace}\n<channel|>\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        return conversations
    else:
        # Default: return as-is (for 'mixed', caller handles it)
        return record["conversations"]


def apply_type_caps(records, type_caps, seed):
    if not type_caps:
        return records

    by_type = {}
    for record in records:
        by_type.setdefault(extract_target_type(record), []).append(record)

    rng = random.Random(seed)
    sampled_records = []
    for guardrail_type, group in by_type.items():
        cap = type_caps.get(guardrail_type)
        if cap is None or cap <= 0 or len(group) <= cap:
            sampled_records.extend(group)
            continue

        blocked = [record for record in group if extract_target_blocked(record)]
        safe = [record for record in group if not extract_target_blocked(record)]
        blocked_target = round(cap * (len(blocked) / len(group))) if group else 0
        safe_target = cap - blocked_target

        blocked_sample = rng.sample(blocked, min(blocked_target, len(blocked)))
        safe_sample = rng.sample(safe, min(safe_target, len(safe)))

        remaining = cap - len(blocked_sample) - len(safe_sample)
        if remaining > 0:
            blocked_left = [record for record in blocked if record not in blocked_sample]
            safe_left = [record for record in safe if record not in safe_sample]
            filler_pool = blocked_left + safe_left
            if filler_pool:
                filler = rng.sample(filler_pool, min(remaining, len(filler_pool)))
                blocked_sample.extend(record for record in filler if extract_target_blocked(record))
                safe_sample.extend(record for record in filler if not extract_target_blocked(record))

        sampled_records.extend(blocked_sample)
        sampled_records.extend(safe_sample)

    rng.shuffle(sampled_records)
    return sampled_records


def prepare_text_dataset(
    path,
    chat_processor,
    text_tokenizer,
    max_seq_length,
    max_train_samples,
    think_mode="required",
    think_ratio=0.2,
    type_caps=None,
    seed=42,
):
    """Load JSONL conversations, apply chat template, and filter by token length.

    think_mode:
        "off"      — all records without thinking
        "required" — all records with Gemma thinking channel
        "mixed"    — think_ratio % of hard cases get thinking, rest don't
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if max_train_samples > 0 and len(records) >= max_train_samples:
                break

    print(f"Loaded {len(records)} records from {path}")
    summarize_records(records)

    records = apply_type_caps(records, type_caps or {}, seed)
    if type_caps:
        print("After type caps:")
        summarize_records(records)

    # For mixed mode, select which records get thinking
    thinking_flags = {}
    if think_mode == "mixed":
        rng = random.Random(seed)
        candidates = [i for i, rec in enumerate(records) if is_thinking_candidate(rec)]
        n_thinking = int(len(records) * think_ratio)
        n_thinking = min(n_thinking, len(candidates))
        thinking_indices = set(rng.sample(candidates, n_thinking))
        thinking_flags = {i: True for i in thinking_indices}
        print(f"Mixed thinking: {len(thinking_indices)} thinking / {len(records)} total ({100*len(thinking_indices)/len(records):.1f}%)")

    texts = []
    n_think_on = 0
    n_think_off = 0
    for i, rec in enumerate(records):
        if think_mode == "mixed":
            use_thinking = thinking_flags.get(i, False)
            if use_thinking:
                convos = maybe_strip_think(rec, "required")
                n_think_on += 1
            else:
                convos = maybe_strip_think(rec, "off")
                n_think_off += 1
        else:
            convos = maybe_strip_think(rec, think_mode)

        text = chat_processor.apply_chat_template(
            convos,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    if think_mode == "mixed":
        print(f"  thinking-on: {n_think_on}, thinking-off: {n_think_off}")

    dataset = Dataset.from_dict({"text": texts})

    before = len(dataset)
    dataset = dataset.filter(
        lambda examples: [
            len(tokenize_text(text_tokenizer, t)) <= max_seq_length
            for t in examples["text"]
        ],
        batched=True,
    )
    print(f"After length filter ({max_seq_length} tokens): {len(dataset)} / {before}")

    return dataset


def save_prepared_dataset(dataset, output_dir):
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            f"Prepared dataset already exists: {output_path}. Remove it or choose a new --output-dir."
        )
    dataset.save_to_disk(str(output_path))
    print(f"Saved prepared dataset to {output_path}")


def load_prepared_dataset(path, max_train_samples):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_path}. Run prepare_dataset.py first."
        )

    # Keep the prepared dataset in memory so later tokenization/cache steps do not
    # try to create temporary files inside the read-only mounted dataset directory.
    dataset = load_from_disk(str(dataset_path), keep_in_memory=True)
    print(f"Loaded prepared dataset from {dataset_path}: {len(dataset)} rows")

    if max_train_samples > 0:
        take = min(max_train_samples, len(dataset))
        dataset = dataset.select(range(take))
        print(f"Using first {take} prepared rows for this run")

    return dataset
