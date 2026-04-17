#!/usr/bin/env python3
"""
Prepare GLiNER2 with Korean DeBERTa encoder.
Swaps the English encoder, saves the combined model to disk.

Usage:
    python prepare_model.py
    python prepare_model.py --output-dir /models/gliner2-korean-base
"""

import argparse
from gliner2 import GLiNER2
from transformers import DebertaV2Model, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gliner2-model", default="fastino/gliner2-base-v1")
    parser.add_argument("--korean-encoder", default="team-lucid/deberta-v3-base-korean")
    parser.add_argument("--output-dir", default="/models/gliner2-korean-base")
    args = parser.parse_args()

    print(f"Loading GLiNER2: {args.gliner2_model}")
    model = GLiNER2.from_pretrained(args.gliner2_model)
    print(f"  Encoder: vocab={model.encoder.config.vocab_size}, hidden={model.encoder.config.hidden_size}")

    print(f"\nLoading Korean encoder: {args.korean_encoder}")
    kr_encoder = DebertaV2Model.from_pretrained(args.korean_encoder)
    print(f"  Korean encoder: vocab={kr_encoder.config.vocab_size}, hidden={kr_encoder.config.hidden_size}")

    tok_path = hf_hub_download(args.korean_encoder, "tokenizer.json")
    kr_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    kr_tokenizer.pad_token = "[PAD]"
    kr_tokenizer.cls_token = "[CLS]"
    kr_tokenizer.sep_token = "[SEP]"
    kr_tokenizer.unk_token = "[UNK]"
    kr_tokenizer.mask_token = "[MASK]"
    print(f"  Korean tokenizer: vocab={kr_tokenizer.vocab_size}")

    assert model.encoder.config.hidden_size == kr_encoder.config.hidden_size, "Hidden size mismatch"
    assert model.encoder.config.num_hidden_layers == kr_encoder.config.num_hidden_layers, "Layer count mismatch"

    # Swap tokenizer first — GLiNER2 adds special tokens internally
    model.processor.tokenizer = kr_tokenizer

    # Add GLiNER2's special tokens to Korean tokenizer so vocab aligns
    special_tokens = model.processor.SPECIAL_TOKENS
    num_added = kr_tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"  Added {num_added} GLiNER2 special tokens, new vocab: {len(kr_tokenizer)}")

    # Swap encoder and resize its embeddings to match tokenizer
    model.encoder = kr_encoder
    model.encoder.resize_token_embeddings(len(kr_tokenizer))
    print(f"  Encoder embeddings resized to: {model.encoder.embeddings.word_embeddings.weight.shape[0]}")

    test = "김민수의 전화번호는 010-1234-5678입니다"
    tokens = kr_tokenizer.tokenize(test)
    print(f"\n  Test: '{test}' -> {tokens}")

    # Update config to reference Korean encoder so loading works correctly
    model.config.model_name = args.korean_encoder

    print(f"\nSaving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
