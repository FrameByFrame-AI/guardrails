import itertools, json
from collections import Counter
from datasets import load_dataset
ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
ex = next(iter(ds))
print("COLUMNS:", list(ex.keys()))
for k, v in ex.items():
    print(f"  {k}: {str(v)[:140]}")
ds2 = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
langs = Counter(); locales = Counter(); ko = 0; ko_labels = Counter(); ko_doc = Counter()
for r in itertools.islice(ds2, 30000):
    lang = str(r.get("language") or r.get("locale") or "")
    langs[lang] += 1
    if "ko" in lang.lower() or "korea" in lang.lower():
        ko += 1
        # try label field
        for fld in ("mbert_bio_labels","privacy_mask","span_labels","labels"):
            v=r.get(fld)
            if v: 
                if isinstance(v,str):
                    try: v=json.loads(v)
                    except: pass
                if isinstance(v,list):
                    for it in v:
                        lab=it.get("label") if isinstance(it,dict) else it
                        ko_labels[lab]+=1
                break
        ko_doc[str(r.get("document_type") or r.get("subdomain") or r.get("domain") or "")]+=1
print("LANGS(top):", langs.most_common(12))
print("KO_in_30k:", ko)
print("KO_doc_types:", ko_doc.most_common(10))
print("KO_labels:", ko_labels.most_common(20))
