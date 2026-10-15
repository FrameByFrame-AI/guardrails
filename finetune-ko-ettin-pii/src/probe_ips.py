import sys, torch
from benchmark_pii import load_model, predict_spans, _normalize_span_set
tok, model, id2label, cats, info = load_model(sys.argv[1], device="auto")
ips = ["192.168.1.1","10.0.0.1","8.8.8.8","203.0.113.45","157.49.139.167","172.16.254.1"]
for ip in ips:
    t = f"제 서버 주소는 {ip}입니다."
    sp = _normalize_span_set(t, predict_spans(tok,model,id2label,t,max_length=128,allowed_labels=cats,model_info=info))
    hit = [s for s in sp if s[0]=="ip_address"]
    print(f"{ip:<18} {'DETECTED' if hit else 'MISSED  '}  spans={sorted((l,t[a:b]) for l,a,b in sp)}")
