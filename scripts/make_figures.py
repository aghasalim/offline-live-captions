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
COND = {"clean": "clean", "room_snr15": "room, 15 dB", "room_snr10": "room, 10 dB", "room_snr5": "room, 5 dB"}

w = {(r["model"], r["condition"]): r for r in csv.DictReader((R / "wer_summary.csv").open())}
models = [m for m in MODELS if (m, "clean") in w]
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(7, 4))
for i, (c, label) in enumerate(COND.items()):
    ax.bar(x + (i - 1.5) * 0.2, [100 * float(w[(m, c)]["wer"]) for m in models], 0.2, label=label)
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
nclips = {}
for r in csv.DictReader((R / "accent_summary.csv").open()):
    acc[r["accent"]][r["model"]] = float(r["wer"])
    nclips[r["accent"]] = int(r["n"])
if acc:
    groups = sorted((g for g in acc if g != "all"), key=lambda g: -nclips[g])
    fig, ax = plt.subplots(figsize=(8, 4))
    xg = np.arange(len(groups))
    for i, m in enumerate(models):
        ax.bar(xg + (i - 1.5) * 0.2, [100 * acc[g][m] for g in groups], 0.2, label=m)
    ax.set_xticks(xg, [f"{g}\n({nclips[g]} clips)" for g in groups])
    ax.set_ylabel("word error rate (%)")
    ax.set_title("WER by self-reported accent (Common Voice clips)")
    ax.set_ylim(0, 55)
    ax.legend(ncol=4, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(R / "wer_by_accent.png", dpi=150)
# accuracy vs speed: each model's WER against its median caption latency at the default 4 s chunk
if (models[0], 4.0) in lat:
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, label, mk in [("clean", "clean", "o"), ("room_snr10", "room, babble 10 dB", "s")]:
        xs = [np.median(lat[(m, 4.0)]) for m in models]
        ys = [100 * float(w[(m, c)]["wer"]) for m in models]
        ax.plot(xs, ys, marker=mk, ls=":", label=label)
        for m, xv, yv in zip(models, xs, ys):
            ax.annotate(m, (xv, yv), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("median end-of-speech to caption latency, 4 s chunk (s)")
    ax.set_ylabel("word error rate (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Accuracy vs caption delay per model (Apple M4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "wer_vs_latency.png", dpi=150)
print("figures written")
