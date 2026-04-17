#!/usr/bin/env python3
"""
Build an augmented dataset directory by combining:

  1. Original per-dataset train splits (copied as-is)
  2. Adversarial obfuscation variants of existing blocked records
  3. Legitimate-unicode safe examples

Output is a directory laid out exactly like processed_split/, so existing
pipelines (format_training_data.py) read it without modification.

    <output-dir>/
    ├── APEACH.train.jsonl                  (copied)
    ├── KDPII.train.jsonl                   (copied)
    ├── ...
    ├── _adversarial_augmented.train.jsonl  (generated)
    └── _legitimate_unicode.train.jsonl     (generated)

Usage:
    python build_augmented_dataset.py \
        --dataset-dir /data/processed \
        --adversarial /data/adversarial_augmented.jsonl \
        --legitimate /data/legitimate_unicode.jsonl \
        --output-dir /data/processed_augmented
"""

import argparse
import json
import shutil
from pathlib import Path


def count_jsonl(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory with per-source *.train.jsonl files")
    parser.add_argument("--adversarial", required=True)
    parser.add_argument("--legitimate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-test", action="store_true",
                        help="Also copy *.test.jsonl for in-place benchmarking")
    args = parser.parse_args()

    src = Path(args.dataset_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Copying original train splits from {src}:")
    total_orig = 0
    for src_path in sorted(src.glob("*.train.jsonl")):
        dst_path = dst / src_path.name
        shutil.copyfile(src_path, dst_path)
        n = count_jsonl(dst_path)
        total_orig += n
        print(f"  {src_path.name:<45} {n:>7}")

    if args.copy_test:
        for src_path in sorted(src.glob("*.test.jsonl")):
            shutil.copyfile(src_path, dst / src_path.name)

    # Copy augmentation files into the output directory as additional *.train.jsonl
    adv_dst = dst / "_adversarial_augmented.train.jsonl"
    leg_dst = dst / "_legitimate_unicode.train.jsonl"
    shutil.copyfile(args.adversarial, adv_dst)
    shutil.copyfile(args.legitimate, leg_dst)

    n_adv = count_jsonl(adv_dst)
    n_leg = count_jsonl(leg_dst)
    print(f"\nAugmentations added:")
    print(f"  _adversarial_augmented.train.jsonl  {n_adv:>7}")
    print(f"  _legitimate_unicode.train.jsonl     {n_leg:>7}")

    total = total_orig + n_adv + n_leg
    aug_frac = (n_adv + n_leg) / total if total else 0
    print(f"\nTotal: {total} records ({100*aug_frac:.1f}% augmented)")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
