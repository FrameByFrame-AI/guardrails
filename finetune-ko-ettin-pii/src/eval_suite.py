"""Multi-domain eval: KDPII (in-domain, all 9 classes) + KLUE-NER (cross-domain, zero-shot)."""
import os, re, json, torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

M = os.environ["KOE_DIAG_MODEL"]
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForTokenClassification.from_pretrained(M, torch_dtype=torch.bfloat16).eval().cuda()

_J = ["이에요","이라고","입니다","이야","이랑","한테","에게","으로","이가","이는","에서","이고","예요",
      "씨","님","이","가","은","는","을","를","야","아","에","의","랑","께","고"]
_DATE = re.compile(r".*(?:일|[0-9])", re.S)
def _norm(t,l,s,e):
    while s<e and t[s] in " .,\t\n": s+=1
    while e>s and t[e-1] in " .,\t\n": e-=1
    if l=="private_date":
        m=_DATE.match(t[s:e]);  e=s+m.end() if (m and m.end()>0) else e
    elif l in ("private_person","personal_handle","private_address"):
        for _ in range(2):
            seg=t[s:e]
            for j in _J:
                if seg.endswith(j) and (e-s)-len(j)>=2: e-=len(j); break
            else: break
    return s,e
def extract(text):
    enc=tok(text,truncation=True,max_length=256,return_offsets_mapping=True,return_tensors="pt")
    off=enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad(): pred=model(**{k:v.cuda() for k,v in enc.items()}).logits.argmax(-1)[0].tolist()
    id2l=model.config.id2label; spans=[]; act=None
    for i,lid in enumerate(pred):
        lab=id2l[int(lid)]; cs,ce=off[i]
        if cs==ce:
            if act: spans.append(act); act=None
            continue
        if lab=="O":
            if act: spans.append(act); act=None
            continue
        pre,cat=lab.split("-",1)
        if pre in ("B","S") or not act or act[0]!=cat:
            if act: spans.append(act)
            act=[cat,cs,ce]
        else: act[2]=ce
    if act: spans.append(act)
    out=set()
    for cat,s,e in spans:
        s,e=_norm(text,cat,s,e)
        if text[s:e].strip(): out.add((cat,s,e))
    return out
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return (2*P*R/(P+R) if P+R else 0),P,R

# --- KDPII in-domain ---
tp=defaultdict(int);fp=defaultdict(int);fn=defaultdict(int); T=F=N=0
for l in open("/gen/ko_pii_opf_v4/test.jsonl",encoding="utf-8"):
    r=json.loads(l); text=r["text"]
    gold={(_norm(text,sp["category"],sp["start"],sp["end"])[0:1][0] if False else sp["category"],)+_norm(text,sp["category"],sp["start"],sp["end"]) for sp in (r.get("label") or [])}
    gold={(sp["category"],*_norm(text,sp["category"],sp["start"],sp["end"])) for sp in (r.get("label") or [])}
    pred=extract(text)
    cats=set(c for c,_,_ in gold)|set(c for c,_,_ in pred)
    for c in cats:
        g={x for x in gold if x[0]==c}; p={x for x in pred if x[0]==c}
        tp[c]+=len(g&p);fp[c]+=len(p-g);fn[c]+=len(g-p)
print("=== KDPII (in-domain) ===")
gt=sum(tp.values());gf=sum(fp.values());gn=sum(fn.values())
print("OVERALL F1=%.3f P=%.3f R=%.3f"%prf(gt,gf,gn))
for c in sorted(tp):
    f,P,R=prf(tp[c],fp[c],fn[c]); print(f"  {c:18} F1={f:.3f}")

# --- held-out UNSEEN-domain set (optional; {text,label} schema like KDPII) ---
HO=os.environ.get("KOE_HELDOUT")
if HO and os.path.exists(HO):
    htp=defaultdict(int);hfp=defaultdict(int);hfn=defaultdict(int)
    for l in open(HO,encoding="utf-8"):
        r=json.loads(l); text=r["text"]
        gold={(sp["category"],*_norm(text,sp["category"],sp["start"],sp["end"])) for sp in (r.get("label") or [])}
        pred=extract(text)
        for c in set(c for c,_,_ in gold)|set(c for c,_,_ in pred):
            g={x for x in gold if x[0]==c}; p={x for x in pred if x[0]==c}
            htp[c]+=len(g&p);hfp[c]+=len(p-g);hfn[c]+=len(g-p)
    print("=== SPY held-out (UNSEEN domains) ===")
    print("OVERALL F1=%.3f P=%.3f R=%.3f"%prf(sum(htp.values()),sum(hfp.values()),sum(hfn.values())))
    for c in sorted(htp):
        f,P,R=prf(htp[c],hfp[c],hfn[c]); print(f"  {c:18} F1={f:.3f}")

# --- KLUE cross-domain (zero-shot) ---
ds=load_dataset("klue","ner",split="validation"); names=ds.features["ner_tags"].feature.names
MAP={"PS":"private_person","DT":"private_date","LC":"private_address"}
ktp=defaultdict(int);kfp=defaultdict(int);kfn=defaultdict(int)
for r in ds:
    tokens,tags=r["tokens"],r["ner_tags"]; text="".join(tokens)
    sp=set();cur=st=None
    for i,t in enumerate(tags):
        n=names[t]
        if n=="O" or n.startswith("B-"):
            if cur: sp.add((cur,st,i));cur=None
        if n.startswith("B-"): cur,st=n[2:],i
        elif n.startswith("I-") and cur is None: cur,st=n[2:],i
    if cur: sp.add((cur,st,len(tags)))
    gold={(MAP[c],*_norm(text,MAP[c],s,e)) for c,s,e in sp if c in MAP}
    pred={x for x in extract(text) if x[0] in MAP.values()}
    for c in MAP.values():
        g={x for x in gold if x[0]==c}; p={x for x in pred if x[0]==c}
        ktp[c]+=len(g&p);kfp[c]+=len(p-g);kfn[c]+=len(g-p)
print("=== KLUE-NER (cross-domain, zero-shot) ===")
for c in ("private_person","private_date","private_address"):
    f,P,R=prf(ktp[c],kfp[c],kfn[c]); print(f"  {c:18} F1={f:.3f} R={R:.3f}")
