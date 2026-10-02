"""WER and real time factor for each model under clean and simulated far-field audio.

Writes results/transcripts/<model>_<condition>.csv (one row per utterance) and
results/wer_summary.csv. Run with --models tiny,base to do fewer.
"""
import argparse
import csv
import time
from pathlib import Path

import soundfile as sf

from livecap.asr import load
from livecap.simulate import babble, far_field, synthetic_rir
from livecap.text import wer

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"clean": None, "room_snr15": 15, "room_snr10": 10, "room_snr5": 5}


def read16k(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000, sr
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny,base,small,large-v3-turbo")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / "data" / "subset.csv").open()))[: args.limit]
    audio = {r["id"]: read16k(r["path"]) for r in rows}
    # babble bed: six test-other speakers that are NOT in the subset
    subset_ids = {r["id"] for r in rows}
    others = [p for p in sorted((ROOT / "data/LibriSpeech/test-other").rglob("*.flac"))
              if p.stem not in subset_ids][:6]
    bed = babble([read16k(p) for p in others], seed=1)
    rir = synthetic_rir(rt60=0.5, seed=1)

    out_dir = ROOT / "results" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / "wer_summary.csv"
    summary = {}
    if summary_path.exists():
        for r in csv.DictReader(summary_path.open()):
            summary[(r["model"], r["condition"])] = r

    for model in args.models.split(","):
        asr = load(model)
        for cond in args.conditions.split(","):
            snr = CONDITIONS[cond]
            hyps, secs, wall = [], 0.0, 0.0
            for i, r in enumerate(rows):
                x = audio[r["id"]]
                if snr is not None:
                    x = far_field(x, rir, bed, snr, seed=i)
                t = time.perf_counter()
                hyps.append(asr(x))
                wall += time.perf_counter() - t
                secs += len(x) / 16000
            w = wer([r["reference"] for r in rows], hyps)
            with (out_dir / f"{model}_{cond}.csv").open("w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["id", "reference", "hypothesis"])
                cw.writerows((r["id"], r["reference"], h) for r, h in zip(rows, hyps))
            summary[(model, cond)] = dict(model=model, condition=cond, n=len(rows),
                                          wer=f"{w:.4f}", audio_seconds=f"{secs:.1f}",
                                          wall_seconds=f"{wall:.1f}", rtf=f"{wall / secs:.4f}")
            print(summary[(model, cond)], flush=True)
            with summary_path.open("w", newline="") as f:
                cw = csv.DictWriter(f, fieldnames=["model", "condition", "n", "wer",
                                                   "audio_seconds", "wall_seconds", "rtf"])
                cw.writeheader()
                cw.writerows(summary[k] for k in sorted(summary))


if __name__ == "__main__":
    main()
