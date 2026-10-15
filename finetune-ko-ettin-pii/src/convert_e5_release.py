import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
SRC="/models/outputs/e5base-korean-pii"
DST="/models/FrameByFrame/korean-pii-e5-base"
tok=AutoTokenizer.from_pretrained(SRC)
m=AutoModelForTokenClassification.from_pretrained(SRC, torch_dtype=torch.bfloat16)
m.config.torch_dtype="bfloat16"
m.save_pretrained(DST, safe_serialization=True); tok.save_pretrained(DST)
print("saved bf16 ->", DST, "| arch:", m.config.architectures, "| labels:", m.config.num_labels)
