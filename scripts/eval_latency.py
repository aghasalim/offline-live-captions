"""End-of-speech to caption latency per chunk size and model.

Each subset utterance is replayed through the same Chunker the live tool uses.
For the chunk that contains the end of the utterance, latency is the time the
chunker waits to confirm the pause (silence_seconds) plus the ASR wall time
for that chunk. Terminal and browser rendering add a few milliseconds and are
not counted. Audio is not played in real time; the ASR wall time is real.
Writes results/latency.csv with one row per (model, chunk, utterance).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from livecap.asr import load
from livecap.chunker import Chunker

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny,base,small,large-v3-turbo")
    ap.add_argument("--chunks", default="2,3,4,6,8")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / "data" / "subset.csv").open()))[: args.n]
    audio = [sf.read(r["path"], dtype="float32")[0] for r in rows]
    out = ROOT / "results" / "latency.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "chunk_seconds", "id", "n_chunks", "last_chunk_seconds",
                    "pause_wait_seconds", "asr_seconds", "latency_seconds"])
        for model in args.models.split(","):
            asr = load(model)
            asr(audio[0])  # warm
            for cs in [float(c) for c in args.chunks.split(",")]:
                lat = []
                for r, x in zip(rows, audio):
                    ch = Chunker(max_seconds=cs)
                    chunks = []
                    for i in range(0, len(x), 1600):
                        chunks += ch.push(x[i:i + 1600])
                    tail = ch.flush()
                    # the end of speech sits in the flushed tail (silence shorter than
                    # silence_seconds) or in the last emitted chunk
                    last = tail if tail is not None and len(tail) >= 1600 else chunks[-1]
                    for c in chunks[:-1]:
                        asr(c)
                    t = time.perf_counter()
                    asr(last)
                    dt = time.perf_counter() - t
                    pause = ch.sil_n / ch.sr
                    w.writerow([model, cs, r["id"], len(chunks) + (tail is not None),
                                f"{len(last) / 16000:.2f}", pause, f"{dt:.4f}", f"{pause + dt:.4f}"])
                    lat.append(pause + dt)
                print(model, cs, f"median {np.median(lat):.3f}s p90 {np.percentile(lat, 90):.3f}s", flush=True)
                f.flush()


if __name__ == "__main__":
    main()
