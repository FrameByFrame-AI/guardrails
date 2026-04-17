#!/usr/bin/env python3
"""Quick debug: see what GLiNER2 actually returns for Korean PII text."""
import json
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

labels = ["person", "phone number", "email address", "street address",
          "credit card number", "bank account number", "identity number",
          "organization", "url", "date of birth"]

tests = [
    "김민수의 전화번호는 010-1234-5678이고, 이메일은 minsu@example.com입니다.",
    "인스타 rainbow879612 여기서 만드는 거.",
    "https://www.electricrusticlandscape.or.kr",
    "John Smith works at OpenAI. His email is john@openai.com and SSN is 123-45-6789.",
]

for text in tests:
    print(f"\nINPUT: {text[:100]}")
    result = model.extract_entities(text, labels)
    print(f"TYPE: {type(result)}")
    print(f"RESULT: {json.dumps(result, indent=2, ensure_ascii=False, default=str)[:500]}")
    print("---")
