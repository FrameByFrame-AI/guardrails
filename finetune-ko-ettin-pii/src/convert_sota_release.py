import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
SRC="/models/outputs/e5base-korean-pii-sota"        # sota1 (locked)
DST="/models/FrameByFrame/korean-pii-e5-base"
tok=AutoTokenizer.from_pretrained(SRC)
m=AutoModelForTokenClassification.from_pretrained(SRC, torch_dtype=torch.bfloat16)
m.config.torch_dtype="bfloat16"
m.save_pretrained(DST, safe_serialization=True); tok.save_pretrained(DST)
print("repackaged sota1 -> release; arch",m.config.architectures,"labels",m.config.num_labels)
