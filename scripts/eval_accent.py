"""WER per self-reported accent on the DTU54DL/common-accent test split.

The split is 451 Common Voice clips with the speaker's self-reported accent.
Only accent groups with at least 10 clips are reported. Writes
results/accent_summary.csv and results/transcripts/accent_<model>.csv.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from livecap.asr import load
from livecap.text import wer

ROOT = Path(__file__).resolve().parents[1]
MIN_CLIPS = 10
SHORT = {
    "India and South Asia (India, Pakistan, Sri Lanka)": "India and South Asia",
    "Southern African (South Africa, Zimbabwe, Namibia)": "Southern African",
    "Hong Kong English": "Hong Kong",
    "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)": "West Indies and Bermuda",
    "Malaysian English": "Malaysian",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny,base,small,large-v3-turbo")
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "data" / "common-accent-test.parquet")
    df = df[df.accent.isin(SHORT)].reset_index(drop=True)
    clips = []
    for a in df.audio:
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            x = resample_poly(x, 16000, sr).astype(np.float32)
        clips.append(x)
    out = []
    (ROOT / "results" / "transcripts").mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        asr = load(model)
        hyps = [asr(x) for x in clips]
        pd.DataFrame({"accent": df.accent.map(SHORT), "reference": df.sentence, "hypothesis": hyps}) \
            .to_csv(ROOT / "results" / "transcripts" / f"accent_{model}.csv", index=False)
        for acc, g in df.groupby("accent"):
            idx = g.index.tolist()
            out.append(dict(model=model, accent=SHORT[acc], n=len(idx),
                            wer=f"{wer(df.sentence[idx].tolist(), [hyps[i] for i in idx]):.4f}"))
            print(out[-1], flush=True)
        out.append(dict(model=model, accent="all", n=len(df),
                        wer=f"{wer(df.sentence.tolist(), hyps):.4f}"))
    pd.DataFrame(out).to_csv(ROOT / "results" / "accent_summary.csv", index=False)


if __name__ == "__main__":
    main()
