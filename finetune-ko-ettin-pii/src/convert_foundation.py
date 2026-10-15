#!/usr/bin/env python3
"""Package the ko-ettin foundation encoder (stage-3 continued MLM) into the FrameByFrame layout (bf16)."""
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

SRC = "/models/outputs/ko-ettin-encoder-150m-stage_3_full_korean_continued_mlm"
DST = "/models/FrameByFrame/ko-ettin-encoder-150m"

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForMaskedLM.from_pretrained(SRC, torch_dtype=torch.bfloat16)
if hasattr(model.config, "reference_compile"):
    model.config.reference_compile = False
model.config.torch_dtype = "bfloat16"
model.save_pretrained(DST, safe_serialization=True)
tok.save_pretrained(DST)
print("saved bf16 foundation ->", DST)
print("arch:", model.config.architectures, "| vocab:", model.config.vocab_size)
