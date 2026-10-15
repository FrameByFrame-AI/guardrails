#!/usr/bin/env python3
"""Package the canonical ko-ettin PII model into the FrameByFrame release layout (bf16)."""
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SRC = "/models/outputs/ko-ettin-encoder-150m-ko-pii"          # canonical (= aug2)
DST = "/models/FrameByFrame/ko-ettin-privacy-filter"

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForTokenClassification.from_pretrained(SRC, torch_dtype=torch.bfloat16)
if hasattr(model.config, "reference_compile"):
    model.config.reference_compile = False
model.config.torch_dtype = "bfloat16"
model.save_pretrained(DST, safe_serialization=True)
tok.save_pretrained(DST)
print("saved bf16 release ->", DST)
print("num_labels:", model.config.num_labels)
print("labels:", list(model.config.id2label.values()))
