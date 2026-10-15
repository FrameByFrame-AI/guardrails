#!/usr/bin/env python3
"""Per-token diagnostic for why the IP in TC_001 is/ isn't detected."""
import sys, torch
from benchmark_pii import load_model

TEXT = ("라굴의 고정 IP 주소는 192.168.1.1이었으며, 동일한 주소를 통해 "
        "http://raghul-store.com에 접속하곤 했습니다.")

def main(model_path):
    tok, model, id2label, cats, info = load_model(model_path, device="auto")
    enc = tok(TEXT, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offs = enc.pop("offset_mapping")[0].tolist()
    dev = model.device if hasattr(model, "device") else "cpu"
    if not info.get("onnx"):
        enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.float()[0]
    probs = torch.softmax(logits, dim=-1)
    argmax = logits.argmax(-1).tolist()
    print("MODEL:", model_path)
    print(f"{'tok':<14}{'span':<10}{'pred':<18}{'p':<7}top-3 (label:prob)")
    for i, (s, e) in enumerate(offs):
        piece = TEXT[s:e].replace("\n", "")
        # focus near the IP region (chars 12-26) and a couple neighbors
        if not (10 <= s <= 28):
            continue
        top = torch.topk(probs[i], 3)
        top3 = "  ".join(f"{id2label[j.item()]}:{p.item():.2f}" for p, j in zip(top.values, top.indices))
        print(f"{piece!r:<14}{f'[{s}:{e}]':<10}{id2label[argmax[i]]:<18}{probs[i][argmax[i]].item():<7.2f}{top3}")

if __name__ == "__main__":
    main(sys.argv[1])
