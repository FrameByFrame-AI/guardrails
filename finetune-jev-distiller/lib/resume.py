"""Crash-safe jsonl output: keep finished rows, drop a torn last line, append the rest.

The GPU server has crashed mid-run (Xid 79), so every long job writes one row per finished item, flushed at once,
and a rerun skips what is already written.
"""

import json
import os


def open_resumable(path: str, key: str):
    """(keys already written, file handle opened for append). A torn last line from a crash is removed."""
    done, good = set(), []
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                break  # only the last line can be torn
            good.append(line if line.endswith("\n") else line + "\n")
        with open(path, "w") as fh:
            fh.writelines(good)
    return done, open(path, "a")


def write_row(fh, row: dict):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
