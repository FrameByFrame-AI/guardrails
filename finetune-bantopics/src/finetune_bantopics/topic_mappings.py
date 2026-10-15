"""Per-dataset category → production topic mappings.

Each public safety dataset uses its own category vocabulary. We collapse all of
them into the same production-aligned topic set so the encoder sees consistent
topic strings during training. Categories that don't map cleanly (e.g.
out-of-scope CBRN, copyright, mental-health-specific) are intentionally
omitted from the mapping — the prep adapters skip rows with no mapped topic.

Where a single source category carries signal for more than one production
topic (e.g. ``hate_speech`` is both ``hatespeech`` and ``hate speech`` to teach
the encoder both surface forms), the value is a *list*.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVIDIA Aegis 2.0 — comma-separated string of "violated_categories"
# ---------------------------------------------------------------------------
AEGIS_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "Violence": ["violence"],
    "Hate/Identity Hate": ["hatespeech", "hate speech"],
    "Harassment": ["harassment", "hatespeech"],
    "Malware": ["malware"],
    "Fraud/Deception": ["fraud", "phishing"],
    "Guns and Illegal Weapons": ["weapons"],
    "Controlled/Regulated Substances": ["drugs"],
    "Sexual": ["sexual content"],
    "Suicide and Self Harm": ["self harm"],
    "PII/Privacy": ["privacy violation"],
    "Criminal Planning/Confessions": ["illegal activity"],
    "Illegal Activity": ["illegal activity"],
    "Political/Misinformation/Conspiracy": ["misinformation"],
    "Immoral/Unethical": ["unethical behavior"],
    "Threat": ["violence"],
}

# ---------------------------------------------------------------------------
# PKU-Alignment/BeaverTails — dict of 14 boolean flags per row
# ---------------------------------------------------------------------------
BEAVERTAILS_FLAG_TO_TOPICS: dict[str, list[str]] = {
    "discrimination,stereotype,injustice": ["discrimination", "hatespeech"],
    "drug_abuse,weapons,banned_substance": ["drugs", "weapons"],
    "financial_crime,property_crime,theft": ["fraud", "financial crime"],
    "hate_speech,offensive_language": ["hatespeech", "hate speech"],
    "misinformation_regarding_ethics,laws_and_safety": ["misinformation"],
    "non_violent_unethical_behavior": ["unethical behavior"],
    "privacy_violation": ["privacy violation"],
    "self_harm": ["self harm"],
    "sexually_explicit,adult_content": ["sexual content"],
    "terrorism,organized_crime": ["terrorism", "violence"],
    "violence,aiding_and_abetting,incitement": ["violence"],
    # Skipped (out of scope or too narrow): animal_abuse, child_abuse,
    # controversial_topics,politics.
}

# ---------------------------------------------------------------------------
# walledai/HarmBench — single "category" field on each behavior
# ---------------------------------------------------------------------------
HARMBENCH_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "cybercrime_intrusion": ["cyber attack", "exploit", "malware"],
    "harassment_bullying": ["harassment", "hatespeech"],
    "illegal": ["illegal activity"],
    "misinformation_disinformation": ["misinformation"],
    "harmful": ["unethical behavior"],
    # Skipped: chemical_biological (CBRN, narrow), copyright.
}

# ---------------------------------------------------------------------------
# JailbreakBench/JBB-Behaviors — "Category" field, OpenAI usage policies
# ---------------------------------------------------------------------------
JBB_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "Harassment/Discrimination": ["harassment", "discrimination", "hatespeech"],
    "Malware/Hacking": ["malware", "cyber attack"],
    "Physical harm": ["violence"],
    "Economic harm": ["fraud", "financial crime"],
    "Fraud/Deception": ["fraud", "phishing"],
    "Disinformation": ["misinformation"],
    "Sexual/Adult content": ["sexual content"],
    "Privacy": ["privacy violation"],
    "Government decision-making": ["misinformation"],
    # Skipped: Expert advice (medical/legal — different scanner concern).
}

# ---------------------------------------------------------------------------
# LibrAI/do-not-answer — hierarchical, mapped via "types_of_harm" string
# ---------------------------------------------------------------------------
DNA_TYPES_OF_HARM_TO_TOPICS: dict[str, list[str]] = {
    "Compromise privacy by leaking or inferring private information (person/individual)":
        ["privacy violation"],
    "Risks from leaking or inferring sensitive information (organization/government)":
        ["privacy violation"],
    "Reducing the cost of disinformation campaigns": ["misinformation"],
    "Causing material harm by disseminating misinformation e.g. in medicine or law":
        ["misinformation"],
    "Disseminating false or misleading information": ["misinformation"],
    "Adult content": ["sexual content"],
    "Nudging or advising users to perform unethical or unsafe actions":
        ["unethical behavior"],
    "Social stereotypes and unfair discrimination": ["discrimination", "hatespeech"],
    "Toxic language (hate speech)": ["hatespeech", "hate speech"],
    "Assisting illegal activities": ["illegal activity"],
    # Skipped: Mental Health or Overreliance Crisis, Treat Chatbot as a Human.
}

# ---------------------------------------------------------------------------
# allenai/wildguardmix — "prompt_harm_category" subcategory string
# ---------------------------------------------------------------------------
WILDGUARD_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "social_stereotypes_and_unfair_discrimination": ["discrimination", "hatespeech"],
    "violence_and_physical_harm": ["violence"],
    "toxic_language_hate_speech": ["hatespeech", "hate speech"],
    "sexual_content": ["sexual content"],
    "defamation_encouraging_unethical_or_unsafe_actions":
        ["defamation", "unethical behavior"],
    "causing_material_harm_by_disseminating_misinformation": ["misinformation"],
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns":
        ["misinformation"],
    "private_information_individual": ["privacy violation"],
    "sensitive_information_organization_government": ["privacy violation"],
    "fraud_assisting_illegal_activities": ["fraud", "illegal activity"],
    "cyberattack": ["cyber attack", "exploit"],
    # Skipped: mental_health_over_reliance_crisis, copyright_violations, others.
}

# ---------------------------------------------------------------------------
# walledai/openai-moderation-dataset — boolean flags per row
# ---------------------------------------------------------------------------
OPENAI_MOD_FLAG_TO_TOPICS: dict[str, list[str]] = {
    "S": ["sexual content"],
    "H": ["hatespeech", "hate speech"],
    "V": ["violence"],
    "HR": ["harassment"],
    "SH": ["self harm"],
    "S3": [],  # sexual/minors — skip; should never appear in training data
    "H2": ["hatespeech", "violence"],
    "V2": ["violence"],
}

# ---------------------------------------------------------------------------
# walledai/XSTest — eval-only. "category" identifies the topic the prompt
# *appears* to be about. Used to configure the scanner during eval.
# ---------------------------------------------------------------------------
# Keyed by the XSTest ``type`` column (not ``category`` — that field doesn't
# exist in the actual schema; mine in earlier iterations had this wrong). Each
# safe-looking type has a ``contrast_*`` variant which is the genuinely-unsafe
# prompt, mapped to the same topic so the scanner sees the same configuration.
XSTEST_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "homonyms":                       ["violence"],
    "contrast_homonyms":              ["violence"],
    "figurative_language":            ["violence"],
    "contrast_figurative_language":   ["violence"],
    "safe_targets":                   ["violence"],
    "contrast_safe_targets":          ["violence"],
    "safe_contexts":                  ["violence", "drugs", "weapons"],
    "contrast_safe_contexts":         ["violence", "drugs", "weapons"],
    "definitions":                    ["violence", "drugs", "weapons", "sexual content"],
    "contrast_definitions":           ["violence", "drugs", "weapons", "sexual content"],
    "real_group_nons_discr":          ["discrimination", "hatespeech"],
    "nons_group_real_discr":          ["discrimination", "hatespeech"],
    "contrast_discr":                 ["discrimination", "hatespeech"],
    "historical_events":              ["violence"],
    "contrast_historical_events":     ["violence"],
    "privacy_public":                 ["privacy violation"],
    "privacy_fictional":              ["privacy violation"],
    "contrast_privacy":               ["privacy violation"],
}

# ---------------------------------------------------------------------------
# thu-coai/SafetyBench — multilingual, used for Chinese eval section.
# Categories are the same in English and Chinese splits.
# ---------------------------------------------------------------------------
SAFETYBENCH_CATEGORY_TO_TOPICS: dict[str, list[str]] = {
    "Offensiveness": ["hatespeech", "hate speech"],
    "Unfairness and Bias": ["discrimination", "hatespeech"],
    "Physical Health": ["violence", "self harm"],
    "Mental Health": ["self harm"],
    "Illegal Activities": ["illegal activity"],
    "Ethics and Morality": ["unethical behavior"],
    "Privacy and Property": ["privacy violation"],
}
