#!/usr/bin/env python3
"""Analyze actual training sequence lengths to understand OOM."""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('fastino/gliner2-multi-v1')
path = '/data/pii_gliner2_train.jsonl'

total_lens = []
input_lens = []
max_record = None

with open(path) as f:
    for line in f:
        r = json.loads(line)
        input_toks = len(tok.encode(r['input'], add_special_tokens=False))
        input_lens.append(input_toks)

        schema_parts = []
        for lbl in r['output'].get('entities', {}).keys():
            schema_parts.append(lbl)
        for lbl, desc in r['output'].get('entity_descriptions', {}).items():
            schema_parts.append(f'{lbl}: {desc}')
        for cls in r['output'].get('classifications', []):
            schema_parts.append(cls['task'])
            schema_parts.extend(cls['labels'])

        schema_text = ' '.join(schema_parts)
        schema_toks = len(tok.encode(schema_text, add_special_tokens=False))
        total = input_toks + schema_toks + 20  # +20 for special tokens
        total_lens.append(total)

        if max_record is None or total > max_record[0]:
            max_record = (total, input_toks, schema_toks, r['input'][:200])

import statistics

print(f"Records: {len(total_lens)}")
print()
print(f"INPUT tokens only:")
print(f"  max={max(input_lens)}  p99={sorted(input_lens)[int(len(input_lens)*0.99)]}  mean={statistics.mean(input_lens):.0f}")
print()
print(f"TOTAL sequence (input + schema + special):")
print(f"  max={max(total_lens)}")
print(f"  p99={sorted(total_lens)[int(len(total_lens)*0.99)]}")
print(f"  p99.9={sorted(total_lens)[int(len(total_lens)*0.999)]}")
print(f"  mean={statistics.mean(total_lens):.0f}")
print()
print(f">512: {sum(1 for t in total_lens if t > 512)}")
print(f">1024: {sum(1 for t in total_lens if t > 1024)}")
print(f">2048: {sum(1 for t in total_lens if t > 2048)}")
print()
print(f"Longest sample:")
print(f"  total={max_record[0]}, input={max_record[1]}, schema={max_record[2]}")
print(f"  text: {max_record[3]}...")
