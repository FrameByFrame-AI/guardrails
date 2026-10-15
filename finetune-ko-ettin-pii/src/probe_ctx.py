import sys, torch
from benchmark_pii import load_model, predict_spans, _normalize_span_set
tok, model, id2label, cats, info = load_model(sys.argv[1], device="auto")
cases = {
 "A full TC_001":          "라굴의 고정 IP 주소는 192.168.1.1이었으며, 동일한 주소를 통해 http://raghul-store.com에 접속하곤 했습니다.",
 "B drop URL":             "라굴의 고정 IP 주소는 192.168.1.1이었으며, 동일한 주소를 통해 접속하곤 했습니다.",
 "C drop person(라굴)":     "고정 IP 주소는 192.168.1.1이었으며, 동일한 주소를 통해 http://raghul-store.com에 접속하곤 했습니다.",
 "D drop both":            "고정 IP 주소는 192.168.1.1이었으며 접속하곤 했습니다.",
 "E trailing 이었으며 only":  "제 서버 주소는 192.168.1.1이었으며 접속합니다.",
 "F trailing 입니다":        "제 서버 주소는 192.168.1.1입니다.",
}
for k,t in cases.items():
    sp = _normalize_span_set(t, predict_spans(tok,model,id2label,t,max_length=256,allowed_labels=cats,model_info=info))
    hit = [t[a:b] for l,a,b in sp if l=="ip_address"]
    other = sorted((l,t[a:b]) for l,a,b in sp if l!="ip_address")
    print(f"{k:<24} IP={'HIT '+str(hit) if hit else 'MISS':<22} other={other}")
