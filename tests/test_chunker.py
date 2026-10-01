import numpy as np

from livecap.chunker import Chunker


def tone(seconds, amp=0.3, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_cuts_at_max_seconds():
    ch = Chunker(max_seconds=2.0, min_seconds=1.0)
    out = ch.push(tone(5.0))
    assert [len(c) for c in out] == [32000, 32000]
    assert len(ch.buf) == 16000


def test_cuts_at_pause_after_min_seconds():
    ch = Chunker(max_seconds=10.0, min_seconds=1.0, silence_seconds=0.4)
    assert ch.push(tone(1.5)) == []
    out = ch.push(np.zeros(int(0.5 * 16000), dtype=np.float32))
    assert len(out) == 1 and len(out[0]) == 32000


def test_pure_silence_is_dropped():
    ch = Chunker(max_seconds=1.0)
    assert ch.push(np.zeros(32000, dtype=np.float32)) == []


def test_flush_returns_remainder():
    ch = Chunker(max_seconds=2.0)
    ch.push(tone(0.5))
    assert len(ch.flush()) == 8000
    assert ch.flush() is None
