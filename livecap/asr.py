"""On-device speech recognition with whichever backend is installed.

Preference order on Apple Silicon: mlx-whisper, then pywhispercpp, then
faster-whisper (CPU, int8). Every backend takes 16 kHz float32 mono and
returns plain text. Model weights are read from the local Hugging Face cache;
nothing here opens a network connection once the weights are present.
"""
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def load(model="base", backend=None):
    """Return a callable text = f(audio_float32_16k)."""
    backends = [backend] if backend else ["mlx", "whispercpp", "faster"]
    errors = []
    for b in backends:
        try:
            return {"mlx": _mlx, "whispercpp": _whispercpp, "faster": _faster}[b](model)
        except ImportError as e:
            errors.append(f"{b}: {e}")
    raise RuntimeError("no ASR backend available: " + "; ".join(errors))


def _mlx(model):
    import mlx_whisper
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never go online at runtime
    repo = MLX_REPOS.get(model, model)
    local = MODELS_DIR / repo.split("/")[-1]
    if local.is_dir():  # weights fetched with curl (see README), no hub lookup at all
        repo = str(local)
    mlx_whisper.load_models.load_model(repo)  # warm up and fail fast

    def run(audio):
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language="en",
                                   fp16=True, condition_on_previous_text=False)
        return r["text"].strip()
    run.name = f"mlx:{model}"
    return run


def _whispercpp(model):
    from pywhispercpp.model import Model
    m = Model(model, print_progress=False, print_realtime=False)

    def run(audio):
        return " ".join(s.text for s in m.transcribe(audio, language="en")).strip()
    run.name = f"whispercpp:{model}"
    return run


def _faster(model):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")

    def run(audio):
        segs, _ = m.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text for s in segs).strip()
    run.name = f"faster:{model}"
    return run
