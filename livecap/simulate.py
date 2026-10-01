"""Simulated far-field room: synthetic reverb plus babble at a fixed SNR.

This is not a recording of a real room. The impulse response is exponentially
decaying Gaussian noise (a standard textbook approximation of late reverb)
with a direct path in front of it. Babble is a sum of other speakers from the
same corpus. Everything is seeded so a rerun gives identical audio.
"""
import numpy as np

SR = 16000


def synthetic_rir(rt60=0.5, direct_delay_ms=8.0, direct_to_reverb_db=0.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / rt60)  # 60 dB down at rt60 (ln(1000) = 6.908)
    tail = rng.standard_normal(n) * decay
    tail /= np.sqrt(np.sum(tail ** 2))
    tail *= 10 ** (-direct_to_reverb_db / 20)
    d = int(direct_delay_ms * sr / 1000)
    rir = np.zeros(d + n, dtype=np.float32)
    rir[0] = 1.0
    rir[d:] += tail
    return rir


def reverberate(x, rir):
    from scipy.signal import fftconvolve
    y = fftconvolve(x, rir)[: len(x)]
    return (y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))).astype(np.float32)


def add_noise(x, noise, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = rng.integers(0, len(noise) - len(x) + 1)
    n = noise[start:start + len(x)]
    px, pn = np.mean(x ** 2), np.mean(n ** 2) + 1e-12
    n = n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return (x + n).astype(np.float32)


def babble(clips, n_speakers=6, seed=0):
    """Overlay n_speakers clips (loudness matched) into one noise bed."""
    rng = np.random.default_rng(seed)
    L = max(len(c) for c in clips[:n_speakers])
    out = np.zeros(L, dtype=np.float32)
    for c in clips[:n_speakers]:
        c = c / (np.sqrt(np.mean(c ** 2)) + 1e-9)
        reps = int(np.ceil(L / len(c)))
        c = np.tile(c, reps)[:L]
        out += np.roll(c, rng.integers(0, L))
    return out / n_speakers


def far_field(x, rir, noise, snr_db, seed=0):
    return add_noise(reverberate(x, rir), noise, snr_db, seed=seed)
