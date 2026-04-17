#!/usr/bin/env python3
"""
Inference script for Korean PII detection using fine-tuned GLiNER2.

Usage:
    python inference.py --text "김민수의 전화번호는 010-1234-5678입니다."
    python inference.py --base-model fastino/gliner2-multi-v1 --adapter /models/gliner2-multi-korean-pii-lora/best --text "..."
"""

import json
import argparse
from gliner2 import GLiNER2

PII_LABELS = [
    "person", "phone number", "email address", "street address",
    "credit card number", "bank account number", "identity number",
    "social security number", "passport number", "driver license number",
    "ip address", "url", "username", "password", "date of birth",
    "organization",
]


def main():
    parser = argparse.ArgumentParser(description="Korean PII detection")
    parser.add_argument("--base-model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--adapter", default="/models/gliner2-multi-korean-pii-lora/best")
    parser.add_argument("--text", default="김민수의 전화번호는 010-1234-5678이고 이메일은 minsu@example.com입니다.")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = GLiNER2.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.adapter}")
    model.load_adapter(args.adapter)

    result = model.extract_entities(args.text, PII_LABELS)

    # Filter empty labels
    if isinstance(result, dict) and "entities" in result:
        entities = {k: v for k, v in result["entities"].items() if v}
    elif isinstance(result, dict):
        entities = {k: v for k, v in result.items() if v}
    else:
        entities = result

    output = {
        "input": args.text,
        "blocked": bool(entities),
        "entities": entities,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
