import re, torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

M = "/models/FrameByFrame/korean-pii-e5-base"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForTokenClassification.from_pretrained(M, torch_dtype=torch.bfloat16).eval().cuda()

_J = ["이에요","이라고","입니다","이야","이랑","한테","에게","으로","이가","이는","에서","이고","예요",
      "씨","님","이","가","은","는","을","를","야","아","에","의","랑","께","고"]
_DATE = re.compile(r".*(?:일|[0-9])", re.S)
def _norm(t,l,s,e):
    while s<e and t[s] in " .,\t\n": s+=1
    while e>s and t[e-1] in " .,\t\n": e-=1
    if l=="private_date":
        m=_DATE.match(t[s:e])
        if m and m.end()>0: e=s+m.end()
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

ds=load_dataset("klue","ner",split="validation")
names=ds.features["ner_tags"].feature.names
MAP={"PS":"private_person","DT":"private_date","LC":"private_address"}
def gold(tokens,tags):
    text="".join(tokens); sp=set(); cur=st=None
    for i,t in enumerate(tags):
        n=names[t]
        if n=="O" or n.startswith("B-"):
            if cur: sp.add((cur,st,i)); cur=None
        if n.startswith("B-"): cur,st=n[2:],i
        elif n.startswith("I-") and cur is None: cur,st=n[2:],i
    if cur: sp.add((cur,st,len(tags)))
    return text,{(MAP[c],s,e) for c,s,e in sp if c in MAP}

tp=defaultdict(int);fp=defaultdict(int);fn=defaultdict(int)
for r in ds:
    text,g=gold(r["tokens"],r["ner_tags"])
    p={x for x in extract(text) if x[0] in MAP.values()}
    for c in MAP.values():
        gg={x for x in g if x[0]==c}; pp={x for x in p if x[0]==c}
        tp[c]+=len(gg&pp); fp[c]+=len(pp-gg); fn[c]+=len(gg-pp)
print("=== ZERO-SHOT transfer: KDPII-trained e5-base PII -> KLUE-NER (news), exact span F1 ===")
for c in ("private_person","private_date","private_address"):
    P=tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0; R=tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
    F=2*P*R/(P+R) if P+R else 0
    print(f"{c:18} F1={F:.3f} P={P:.3f} R={R:.3f} gold={tp[c]+fn[c]}")
