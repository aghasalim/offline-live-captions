"""Figures in results/ from the CSVs."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"
MODELS = ["tiny", "base", "small", "large-v3-turbo"]
COND = {"clean": "clean", "room_snr10": "simulated room, babble 10 dB", "room_snr5": "simulated room, babble 5 dB"}

w = {(r["model"], r["condition"]): r for r in csv.DictReader((R / "wer_summary.csv").open())}
models = [m for m in MODELS if (m, "clean") in w]
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(7, 4))
for i, (c, label) in enumerate(COND.items()):
    ax.bar(x + (i - 1) * 0.26, [100 * float(w[(m, c)]["wer"]) for m in models], 0.26, label=label)
ax.set_xticks(x, models)
ax.set_ylabel("word error rate (%)")
ax.set_title("WER on 300 LibriSpeech test-other utterances")
ax.legend()
fig.tight_layout()
fig.savefig(R / "wer_vs_model.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(models, [float(w[(m, "clean")]["rtf"]) for m in models])
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_ylabel("real time factor (processing s / audio s)")
ax.set_title("RTF on Apple M4, mlx-whisper, clean audio")
fig.tight_layout()
fig.savefig(R / "rtf_per_model.png", dpi=150)

lat = defaultdict(list)
for r in csv.DictReader((R / "latency.csv").open()):
    lat[(r["model"], float(r["chunk_seconds"]))].append(float(r["latency_seconds"]))
chunks = sorted({c for _, c in lat})
fig, ax = plt.subplots(figsize=(7, 4))
for m in models:
    if (m, chunks[0]) in lat:
        ax.plot(chunks, [np.median(lat[(m, c)]) for c in chunks], marker="o", label=m)
ax.set_xlabel("max chunk length (s)")
ax.set_ylabel("median end-of-speech to caption (s)")
ax.set_title("Caption latency vs chunk size")
ax.legend()
fig.tight_layout()
fig.savefig(R / "latency_vs_chunk.png", dpi=150)

acc = defaultdict(dict)
for r in csv.DictReader((R / "accent_summary.csv").open()):
    acc[r["accent"]][r["model"]] = float(r["wer"])
if acc:
    groups = [g for g in acc if g != "all"]
    fig, ax = plt.subplots(figsize=(8, 4))
    xg = np.arange(len(groups))
    for i, m in enumerate(models):
        ax.bar(xg + (i - 1.5) * 0.2, [100 * acc[g][m] for g in groups], 0.2, label=m)
    ax.set_xticks(xg, groups, rotation=15, ha="right")
    ax.set_ylabel("word error rate (%)")
    ax.set_title("WER by self-reported accent (Common Voice clips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "wer_by_accent.png", dpi=150)
print("figures written")
