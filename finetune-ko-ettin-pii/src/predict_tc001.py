#!/usr/bin/env python3
"""Ad-hoc: run the ko-ettin-pii span tagger on a single text and render redaction."""
import sys
from benchmark_pii import load_model, predict_spans, _normalize_span_set

TEXT = ("라굴의 고정 IP 주소는 192.168.1.1이었으며, 동일한 주소를 통해 "
        "http://raghul-store.com에 접속하곤 했습니다.")
EXPECTED = ("[PRIVATE_PERSON]의 고정 IP 주소는 [IP_ADDRESS]이었으며, 동일한 주소를 통해 "
            "[PRIVATE_URL]에 접속하곤 했습니다.")

def render(text, spans):
    # non-overlapping, replace right-to-left
    spans = sorted(spans, key=lambda s: s[1])
    out, last_end, keep = text, None, []
    for label, s, e in sorted(spans, key=lambda s: s[1]):
        keep.append((label, s, e))
    for label, s, e in sorted(keep, key=lambda s: -s[1]):
        out = out[:s] + f"[{label.upper()}]" + out[e:]
    return out

def main(paths):
    for p in paths:
        print("=" * 80)
        print("MODEL:", p)
        tok, model, id2label, cats, info = load_model(p, device="auto")
        spans = predict_spans(tok, model, id2label, TEXT, max_length=256,
                              allowed_labels=cats, model_info=info)
        spans = _normalize_span_set(TEXT, spans)
        print("  raw spans:")
        for label, s, e in sorted(spans, key=lambda x: x[1]):
            print(f"    {label:<16} [{s:>3}:{e:<3}] {TEXT[s:e]!r}")
        print("  RENDERED:", render(TEXT, spans))
        print("  EXPECTED:", EXPECTED)
        print("  MATCH:", render(TEXT, spans) == EXPECTED)

if __name__ == "__main__":
    main(sys.argv[1:])
