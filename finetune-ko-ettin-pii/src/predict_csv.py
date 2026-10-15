#!/usr/bin/env python3
"""Run a ko PII span tagger over the 50-case eval CSV; render redactions, compare."""
import csv, sys
from benchmark_pii import load_model, predict_spans, _normalize_span_set

def render(text, spans):
    out = text
    for label, s, e in sorted(spans, key=lambda x: -x[1]):
        out = out[:s] + f"[{label.upper()}]" + out[e:]
    return out

def main(model_path, csv_path):
    tok, model, id2label, cats, info = load_model(model_path, device="auto")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    exact = 0
    for r in rows:
        text = r["Input_Text"]
        spans = _normalize_span_set(text, predict_spans(
            tok, model, id2label, text, max_length=256,
            allowed_labels=cats, model_info=info))
        rendered = render(text, spans)
        expected = r["Expected_Output"]
        ok = rendered.strip() == expected.strip()
        exact += ok
        tags = ",".join(sorted({l for l, _, _ in spans})) or "-"
        print(f"{r['Test_Case_ID']}  EXACT={'Y' if ok else 'n'}  detected=[{tags}]")
        if not ok:
            print(f"    OUT: {rendered}")
            print(f"    EXP: {expected}")
    print(f"\nExact-match vs Expected_Output: {exact}/{len(rows)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
