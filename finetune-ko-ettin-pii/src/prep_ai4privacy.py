"""Map ai4privacy-400k (multi-domain, EU langs) -> our 9-class PII schema for cross-lingual transfer."""
import argparse, json, itertools
from collections import Counter
from datasets import load_dataset

ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100000)
ap.add_argument("--out",default="/models/outputs/ai4privacy_mapped.jsonl"); a=ap.parse_args()

def maplab(L):
    u=L.upper()
    if u=="USERNAME": return "personal_handle"
    if "EMAIL" in u: return "private_email"
    if "URL" in u: return "private_url"
    if u.startswith("IP") or "IPV" in u: return "ip_address"
    if "PHONE" in u or "TELEPHONE" in u: return "private_phone"
    if "DATEOFBIRTH" in u or u=="DOB": return "private_date"
    if any(k in u for k in ("STREET","CITY","ZIP","BUILDING","ADDRESS","STATE","COUNTY","SECONDARYADDRESS")): return "private_address"
    if any(k in u for k in ("ACCOUNT","CREDITCARD","IBAN","BIC","SSN","SOCIAL","PASSPORT","DRIVER","VEHICLE","TAXNUM","CURRENCYCODE" if False else "IBAN")): return "account_number"
    if "NAME" in u:  # after USERNAME handled
        return "private_person"
    return None

ds=load_dataset("ai4privacy/pii-masking-400k",split="train",streaming=True)
n=0; kept=0; labc=Counter()
with open(a.out,"w",encoding="utf-8") as fh:
    for r in ds:
        if kept>=a.n: break
        n+=1
        text=r.get("source_text") or ""
        spans=r.get("privacy_mask") or []
        if isinstance(spans,str):
            try: spans=json.loads(spans)
            except: spans=[]
        out=[]
        for sp in spans:
            cat=maplab(sp.get("label",""))
            if not cat: continue
            s,e=int(sp["start"]),int(sp["end"])
            if 0<=s<e<=len(text) and text[s:e].strip():
                out.append({"category":cat,"start":s,"end":e}); labc[cat]+=1
        if out:
            # drop overlapping spans (keep first)
            out.sort(key=lambda x:(x["start"],x["end"])); clean=[]; last=-1
            for sp in out:
                if sp["start"]>=last: clean.append(sp); last=sp["end"]
            fh.write(json.dumps({"text":text,"label":clean,"info":{"source":"ai4privacy"}},ensure_ascii=False)+"\n")
            kept+=1
print(f"scanned {n}, kept {kept} -> {a.out}")
print("label counts:",labc.most_common())
