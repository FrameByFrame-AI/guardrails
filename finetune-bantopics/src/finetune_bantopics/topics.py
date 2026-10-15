"""Production-aligned topic taxonomy for BanTopics training.

Sources:
- ``policy_definations.json`` — UI defaults: ``violence``, ``hatespeech``;
  threshold options Low=0.3 / Med=0.5 / High=0.7.
- ``request_payloads.json`` — one customer config: ``ransomware``, ``phishing``,
  ``malware``, ``cyber attack``.

Topics are open-ended at request time (``text-chips``) — the goal is
generalisation across topic strings, not memorisation of a fixed head. The
expanded vocabulary below adds high-frequency customer topics (discrimination,
terrorism, defamation, financial crime) covered by the Tier-1 datasets.
"""

from __future__ import annotations

PRODUCTION_DEFAULT_TOPICS = ["violence", "hatespeech"]

CUSTOMER_SECURITY_TOPICS = ["ransomware", "phishing", "malware", "cyber attack"]

# Topics added because the new datasets cover them well and they are
# realistic customer-supplied topic strings.
EXPANDED_TOPICS = [
    "discrimination",
    "terrorism",
    "defamation",
    "financial crime",
    "fraud",
    "harassment",
    "weapons",
    "drugs",
    "sexual content",
    "self harm",
    "privacy violation",
    "illegal activity",
    "misinformation",
    "unethical behavior",
    "hate speech",  # surface variant of hatespeech
]

# Topics that need synthesis because they're under-represented across all
# public safety datasets we ingest.
SECURITY_TOPICS_NEEDING_SYNTHESIS = [
    "ransomware",
    "cyber attack",
    "social engineering",
    "exploit",
    "vulnerability",
]


def all_topics() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for src in (
        PRODUCTION_DEFAULT_TOPICS,
        CUSTOMER_SECURITY_TOPICS,
        EXPANDED_TOPICS,
        SECURITY_TOPICS_NEEDING_SYNTHESIS,
    ):
        for t in src:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
