"""Pick a fixed 300-utterance subset of LibriSpeech test-other -> data/subset.csv.

The list is seeded, so anyone with the same tar gets the same 300 files.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "LibriSpeech" / "test-other"
OUT = Path(__file__).resolve().parents[1] / "data" / "subset.csv"
N = 300

rows = []
for trans in sorted(ROOT.rglob("*.trans.txt")):
    for line in trans.read_text().splitlines():
        uid, text = line.split(" ", 1)
        rows.append((uid, str(trans.parent / f"{uid}.flac"), text))
rows.sort()
random.Random(2026).shuffle(rows)
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "path", "reference"])
    w.writerows(rows[:N])
print(f"wrote {N} of {len(rows)} utterances to {OUT}")
