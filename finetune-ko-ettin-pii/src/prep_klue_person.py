"""Extract KLUE-NER (news) person spans -> private_person training examples."""
import json
from datasets import load_dataset
ds = load_dataset("klue", "ner", split="train")
names = ds.features["ner_tags"].feature.names
out = open("/models/outputs/klue_person.jsonl", "w", encoding="utf-8")
n = 0
for r in ds:
    tokens, tags = r["tokens"], r["ner_tags"]; text = "".join(tokens)
    sp = []; cur = st = None
    for i, t in enumerate(tags):
        nm = names[t]
        if nm == "O" or nm.startswith("B-"):
            if cur: sp.append((cur, st, i)); cur = None
        if nm.startswith("B-"): cur, st = nm[2:], i
        elif nm.startswith("I-") and cur is None: cur, st = nm[2:], i
    if cur: sp.append((cur, st, len(tags)))
    labels = [{"category": "private_person", "start": s, "end": e} for c, s, e in sp if c == "PS"]
    if labels:
        out.write(json.dumps({"text": text, "label": labels, "info": {"source": "klue_ner"}}, ensure_ascii=False) + "\n"); n += 1
print("wrote", n, "KLUE-train person sentences")
