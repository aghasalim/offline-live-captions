"""Re-derive every number in README.md from results/*.csv.

README tables sit between <!-- table:NAME --> and <!-- /table --> markers and
are regenerated here; a mismatch fails. Inline claims are listed in CLAIMS as
(regex, derived value) pairs and must match too. `--write` rewrites the tables.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MODELS = ["tiny", "base", "small", "large-v3-turbo"]
COND = {"clean": "clean", "room_snr15": "room, babble 15 dB", "room_snr10": "room, babble 10 dB", "room_snr5": "room, babble 5 dB"}


def load_wer():
    d = {}
    for r in csv.DictReader((R / "wer_summary.csv").open()):
        d[(r["model"], r["condition"])] = r
    return d


def load_latency():
    d = defaultdict(list)
    for r in csv.DictReader((R / "latency.csv").open()):
        d[(r["model"], float(r["chunk_seconds"]))].append(float(r["latency_seconds"]))
    return d


def load_accent():
    d = defaultdict(dict)
    for r in csv.DictReader((R / "accent_summary.csv").open()):
        d[r["accent"]][r["model"]] = r
    return d


def pct(x):
    return f"{100 * float(x):.1f}%"


def table_wer():
    w = load_wer()
    n = next(iter(w.values()))["n"]
    lines = [f"| model | {' | '.join(COND.values())} | RTF (clean) |", "|---|" + "---|" * (len(COND) + 1)]
    for m in MODELS:
        if (m, "clean") not in w:
            continue
        cells = [pct(w[(m, c)]["wer"]) if (m, c) in w else "" for c in COND]
        lines.append(f"| {m} | {' | '.join(cells)} | {float(w[(m, 'clean')]['rtf']):.3f} |")
    lines.append(f"\nWER on {n} LibriSpeech test-other utterances. RTF is processing time divided by audio duration on the M4.")
    return "\n".join(lines)


def table_latency():
    lat = load_latency()
    chunks = sorted({c for _, c in lat})
    lines = ["| model | " + " | ".join(f"{c:g} s chunk" for c in chunks) + " |", "|---|" + "---|" * len(chunks)]
    for m in MODELS:
        if (m, chunks[0]) not in lat:
            continue
        cells = [f"{np.median(lat[(m, c)]):.2f} / {np.percentile(lat[(m, c)], 90):.2f}" for c in chunks]
        lines.append(f"| {m} | {' | '.join(cells)} |")
    n = len(lat[(MODELS[0], chunks[0])])
    lines.append(f"\nMedian / 90th percentile end-of-speech to caption latency in seconds, {n} utterances per cell. Includes the {0.4:g} s pause the chunker waits for.")
    return "\n".join(lines)


def table_accent():
    a = load_accent()
    models = [m for m in MODELS if m in next(iter(a.values()))]
    lines = ["| accent (self-reported) | clips | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    order = sorted((k for k in a if k != "all"), key=lambda k: -int(a[k][models[0]]["n"])) + ["all"]
    for k in order:
        lines.append(f"| {k} | {a[k][models[0]]['n']} | " + " | ".join(pct(a[k][m]["wer"]) for m in models) + " |")
    return "\n".join(lines)


TABLES = {"wer": table_wer, "latency": table_latency, "accent": table_accent}


def claims():
    """Inline numbers in README prose: (regex with one group, expected string)."""
    w = load_wer()
    lat = load_latency()
    a = load_accent()
    out = [
        (r"the small model gets\s+\*\*([\d.]+%)\*\*", pct(w[("small", "clean")]["wer"])),
        (r"large-v3-turbo gets \*\*([\d.]+%)\*\*", pct(w[("large-v3-turbo", "clean")]["wer"])),
        (r"tiny gets\s+\*\*([\d.]+%)\*\*", pct(w[("tiny", "clean")]["wer"])),
        (r"5 dB the small model\s+is at \*\*([\d.]+%)\*\*", pct(w[("small", "room_snr5")]["wer"])),
        (r"small runs at \*\*([\d.]+)\*\*", f"{float(w[('small', 'clean')]['rtf']):.3f}"),
        (r"the small model shows[^*]*\*\*([\d.]+) s\*\*", f"{np.median(lat[('small', 4.0)]):.2f}"),
        (r"large-v3-turbo a median latency of \*\*([\d.]+) s\*\*", f"{np.median(lat[('large-v3-turbo', 4.0)]):.2f}"),
        (r"India and South Asia[^*]*\*\*([\d.]+%)\*\*", pct(a["India and South Asia"]["small"]["wer"])),
    ]
    return out


def main():
    readme = (ROOT / "README.md").read_text()
    new = readme
    bad = 0
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n)(.*?)(\n<!-- /table -->)", re.S)
        m = pat.search(new)
        if not m:
            print(f"missing table marker: {name}")
            bad += 1
            continue
        want = fn()
        if m.group(2) != want:
            bad += 1
            print(f"table {name} differs from results/")
            new = new[: m.start(2)] + want + new[m.end(2):]
    for pat, want in claims():
        m = re.search(pat, readme, re.S)
        if not m:
            print(f"claim not found in README: {pat}")
            bad += 1
        elif m.group(1) != want:
            print(f"claim mismatch: README says {m.group(1)}, results say {want} ({pat})")
            bad += 1
    if "--write" in sys.argv:
        (ROOT / "README.md").write_text(new)
        print("tables rewritten")
        return 0
    print("all README numbers match results/" if not bad else f"{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
