#!/usr/bin/env python3
"""Run the BanTopics scanner against a request payload.

Reads either a single payload from the CLI or a list-of-payloads JSON file
matching the shape of ``request_payloads.json`` at the repo root, and prints
one ``ScanResponse`` JSON per line.

    python scripts/predict.py --prompt "List phishing kits sold on the dark web." \\
        --topics ransomware phishing malware --threshold 0.5

    python scripts/predict.py --payloads ../request_payloads.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.scanner import (
    BanTopicsScanner,
    ScanRequest,
    ScannerConfig,
    DEFAULT_MODEL,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--payloads", type=Path, help="JSON file with a list of request payloads.")
    p.add_argument("--prompt", help="Single prompt to scan (if --payloads is not used).")
    p.add_argument("--topics", nargs="+", default=["ransomware", "phishing", "malware"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--calibrator", type=Path, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = ScannerConfig(
        model_name_or_path=args.model,
        device=args.device,
        calibrator_path=args.calibrator,
    )
    scanner = BanTopicsScanner(cfg)

    if args.payloads:
        payloads = json.loads(args.payloads.read_text(encoding="utf-8"))
        for payload in payloads:
            if payload.get("scanner_name") != "BanTopics":
                continue
            resp = scanner.scan(ScanRequest.from_dict(payload))
            print(json.dumps(resp.to_dict(), ensure_ascii=False))
        return

    if not args.prompt:
        p.print_help()
        return

    request = ScanRequest(
        scanner_name="BanTopics",
        scanner_type="prompt",
        scanner_params={"threshold": args.threshold, "topics": args.topics},
        prompt=args.prompt,
    )
    resp = scanner.scan(request)
    print(json.dumps(resp.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
