import itertools, json
from datasets import load_dataset
from collections import Counter
ds = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test", streaming=True)
ex = next(iter(ds))
print("COLUMNS:", list(ex.keys()))
for k, v in ex.items():
    print(f"  {k}: {str(v)[:160]}")
ds2 = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test", streaming=True)
langs = Counter(); labels = Counter(); ko = 0
for r in itertools.islice(ds2, 8000):
    lang = str(r.get("language"))
    langs[lang] += 1
    if "korean" in lang.lower() or lang == "ko":
        ko += 1
        sp = r.get("pii_spans")
        if isinstance(sp, str):
            try: sp = json.loads(sp)
            except: sp = []
        for s in (sp or []):
            labels[s.get("label")] += 1
print("LANGS:", langs.most_common(10))
print("KO_in_8k:", ko)
print("KO_LABELS:", labels.most_common(40))
