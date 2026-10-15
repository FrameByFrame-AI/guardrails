"""One definition per dataset: where its training and benchmark prompts come from, and its own block/allow label.

`train` and `bench` are lists of (config, split). A dataset with no `train` splits is benchmark-only: it is never
trained on, so it measures generalisation. `gold(row, split)` maps the dataset's own label to "should a guardrail
block this prompt?" (True / False), or None when the row has no usable label.
Columns and splits were checked against the cached dataset_info.json files.
"""

MODERATION_FLAGS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")


def _openai_moderation(row, split):
    if all(row[f] is None for f in MODERATION_FLAGS):
        return None
    return any(row[f] == 1 for f in MODERATION_FLAGS)


def _label_one(row, split):
    return row["label"] == 1


def _always(row, split):
    return True


DATASETS = {
    # mixed safety: harmful and benign requests, vanilla and adversarial
    "wildguard": dict(hf="allenai/wildguardmix", text="prompt", group="mixed_safety",
                      train=[("wildguardtrain", "train")], bench=[("wildguardtest", "test")],
                      gold=lambda r, s: {"harmful": True, "unharmful": False}.get(r["prompt_harm_label"])),
    "aegis": dict(hf="nvidia/aegis-ai-content-safety-dataset-2.0", text="prompt", group="mixed_safety",
                  train=[(None, "train")], bench=[(None, "test")],
                  gold=lambda r, s: {"unsafe": True, "safe": False}.get(r["prompt_label"])),
    "openai_moderation": dict(hf="walledai/openai-moderation-dataset", text="prompt", group="toxic_content",
                              train=[], bench=[(None, "train")], gold=_openai_moderation),
    # harmful requests (benchmark-only)
    "harmbench": dict(hf="walledai/harm_bench", text="prompt", group="harmful_requests",
                      train=[], bench=[("standard", "train")], gold=_always),
    "jbb": dict(hf="JailbreakBench/jbb-behaviors", text="Goal", group="harmful_requests",
                train=[], bench=[("behaviors", "harmful"), ("behaviors", "benign")],
                gold=lambda r, s: s == "harmful"),
    "jbb_paraphrase": dict(hf="DhruvTre/jailbreakbench-paraphrase-2025-08", text="prompt", group="harmful_requests",
                           train=[], bench=[(None, "test")], gold=_label_one),
    "do_not_answer": dict(hf="LibrAI/do-not-answer", text="question", group="harmful_requests",
                          train=[], bench=[(None, "train")], gold=_always),
    # over-refusal: safe prompts that sound unsafe, with unsafe contrasts (benchmark-only)
    "xs_test": dict(hf="walledai/xs_test", text="prompt", group="over_refusal",
                    train=[], bench=[(None, "test")], gold=lambda r, s: r["label"] == "unsafe"),
    # jailbreak templates: the mini set is the benchmark, the rest is trainable
    "jailbreakv_28k": dict(hf="JailbreakV-28K/jail_break_v-28k", text="jailbreak_query", group="jailbreak_templates",
                           train=[("JailBreakV_28K", "JailBreakV_28K")],
                           bench=[("JailBreakV_28K", "mini_JailBreakV_28K")], gold=_always),
    # direct prompt injection
    "deepset_injection": dict(hf="deepset/prompt-injections", text="text", group="direct_injection",
                              train=[(None, "train")], bench=[(None, "test")], gold=_label_one),
    "safeguard_injection": dict(hf="xTRam1/safe-guard-prompt-injection", text="text", group="direct_injection",
                                train=[(None, "train")], bench=[(None, "test")], gold=_label_one),
    "prompt_shield": dict(hf="hendzh/prompt_shield", text="prompt", group="direct_injection",
                          train=[(None, "train")], bench=[(None, "test")], gold=_label_one),
    # indirect injection: every LLMail entry is a submitted attack email (train Phase1, benchmark Phase2)
    "llmail_inject": dict(hf="microsoft/llmail-inject-challenge", text="body", group="indirect_injection",
                          train=[(None, "Phase1")], bench=[(None, "Phase2")], gold=_always),
    # beavertails' is_safe rates the response, not the prompt: no usable prompt label, so not benchmarked
    "beavertails": dict(hf="PKU-Alignment/beaver_tails", text="prompt", group="mixed_safety",
                        train=[(None, "30k_train")], bench=[], gold=lambda r, s: None),
    # Added after the first benchmark build; kept last so earlier datasets are sampled exactly as before.
    # Benign contrasts for jailbreak templates: ordinary prompts, including benign role-play, from the same sources.
    "jailbreak_classification": dict(hf="jackhhao/jailbreak-classification", text="prompt", group="jailbreak_templates",
                                     train=[(None, "train")], bench=[(None, "test")],
                                     gold=lambda r, s: {"jailbreak": True, "benign": False}.get(r["type"])),
    "in_the_wild": dict(hf="TrustAIRLab/in-the-wild-jailbreak-prompts", text="prompt", group="jailbreak_templates",
                        train=[], bench=[("jailbreak_2023_12_25", "train"), ("regular_2023_12_25", "train")],
                        gold=lambda r, s: bool(r["jailbreak"])),
    "chatgpt_jailbreaks": dict(hf="rubend18/ChatGPT-Jailbreak-Prompts", text="Prompt", group="jailbreak_templates",
                               train=[], bench=[(None, "train")], gold=_always),
    # Benign contrasts for indirect injection: ordinary business emails (Enron). Different source and era from the
    # LLMail attack emails, so a model could learn style instead of injection; watch for that in results.
    "aeslc_emails": dict(hf="Yale-LILY/aeslc", text="email_body", group="indirect_injection",
                         train=[(None, "train")], bench=[(None, "test")], gold=lambda r, s: False),
    "gandalf": dict(hf="Lakera/gandalf_ignore_instructions", text="text", group="direct_injection",
                    train=[(None, "train")], bench=[(None, "test")], gold=_always),
    # Added for run 3: benign prompts that sound sensitive, so the model learns where the line is.
    # OR-Bench: 80k "seemingly toxic" safe prompts to train on; the hard 1k (safe prompts most LLMs refuse) to benchmark.
    "or_bench": dict(hf="bench-llm/or-bench", text="prompt", group="over_refusal",
                     train=[("or-bench-80k", "train")], bench=[("or-bench-hard-1k", "train")], gold=lambda r, s: False),
    "or_bench_toxic": dict(hf="bench-llm/or-bench", text="prompt", group="over_refusal",
                           train=[("or-bench-toxic", "train")], bench=[], gold=_always),
    # CoCoNot: "safety concerns" requests are harmful; its contrast set (test split only) holds look-alike safe requests.
    # Its other noncompliance categories (incomplete, unsupported, humanizing) are not guardrail decisions.
    "coconot": dict(hf="allenai/coconot", text="prompt", group="over_refusal",
                    train=[("original", "train"), ("contrast", "test")], bench=[],
                    gold=lambda r, s: False if s == "test" else
                    (True if r["category"] == "Requests with safety concerns" else None)),
    # WildGuard's adversarial benign prompts: jailbreak-shaped wording around harmless requests.
    "wildguard_adv_benign": dict(hf="allenai/wildguardmix", text="prompt", group="mixed_safety",
                                 train=[("wildguardtrain", "train")], bench=[],
                                 gold=lambda r, s: False if r["adversarial"] and r["prompt_harm_label"] == "unharmful" else None),
    # Real user prompts (lmsys toxic-chat): toxic or jailbreaking = block.
    "toxic_chat": dict(hf="lmsys/toxic-chat", text="user_input", group="mixed_safety",
                       train=[("toxicchat0124", "train")], bench=[("toxicchat0124", "test")],
                       gold=lambda r, s: bool(r["toxicity"] or r["jailbreaking"])),
    # Synthetic benign emails with LLMail attacks' surface features (gen_benign_emails.py), split 80/20 by hash.
    "hard_benign_emails": dict(hf="json", text="text", group="indirect_injection",
                               files={"train": "data/hard_benign_emails_train.jsonl",
                                      "bench": "data/hard_benign_emails_bench.jsonl"},
                               train=[(None, "train")], bench=[(None, "bench")], gold=lambda r, s: False),
}


def load_rows(name: str, which: str, n: int, rng) -> list[dict]:
    """Up to n random labelled rows from a dataset's `train` or `bench` splits."""
    import datasets

    spec = DATASETS[name]
    rows = []
    for config, split in spec[which]:
        if "files" in spec:  # local jsonl, one file per split
            ds = datasets.load_dataset("json", data_files=spec["files"][split], split="train")
        else:
            ds = datasets.load_dataset(spec["hf"], config, split=split)
        for i in rng.sample(range(len(ds)), min(len(ds), n * 3)):  # oversample; empties / unlabelled / dupes trimmed
            r = ds[i]
            text = (r.get(spec["text"]) or "").strip()
            gold = spec["gold"](r, split)
            if text and gold is not None:
                rows.append({"text": text, "source": name, "group": spec["group"], "split": split, "gold_block": gold})
    rng.shuffle(rows)
    return rows
