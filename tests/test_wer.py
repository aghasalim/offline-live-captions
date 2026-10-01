import csv
from pathlib import Path

from livecap.text import normalize, wer
from livecap.simulate import add_noise, babble, far_field, synthetic_rir
import numpy as np

FIX = Path(__file__).parent / "fixtures" / "tiny_transcripts.csv"


def test_normalize_strips_case_and_punctuation():
    assert normalize("Hello, World!  it's") == "hello world its"  # jiwer drops apostrophes too


def test_wer_on_fixture():
    rows = list(csv.DictReader(FIX.open()))
    w = wer([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
    # 3 references, 16 words, 2 errors (one substitution, one deletion)
    assert abs(w - 2 / 16) < 1e-9


def test_wer_empty_hypothesis_counts_as_all_deleted():
    assert wer(["one two"], [""]) == 1.0


def test_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    n = rng.standard_normal(16000).astype(np.float32)
    y = add_noise(x, n, 10.0)
    added = y - x
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean(added ** 2))
    assert abs(snr - 10.0) < 0.05


def test_far_field_is_deterministic_and_same_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    bed = babble([rng.standard_normal(8000).astype(np.float32) for _ in range(6)])
    rir = synthetic_rir(seed=1)
    a = far_field(x, rir, bed, 5, seed=3)
    b = far_field(x, rir, bed, 5, seed=3)
    assert len(a) == len(x) and np.array_equal(a, b)
