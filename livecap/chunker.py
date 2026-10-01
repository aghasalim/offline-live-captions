"""Turn a stream of audio frames into chunks that end at a pause.

A chunk is emitted either when the running buffer hits `max_seconds`, or when
`min_seconds` of audio has accumulated and the last `silence_seconds` of it is
below `silence_rms`. Pure numpy so it is testable without a microphone.
"""
import numpy as np

SAMPLE_RATE = 16000


class Chunker:
    def __init__(self, max_seconds=4.0, min_seconds=1.0, silence_seconds=0.4,
                 silence_rms=0.01, sample_rate=SAMPLE_RATE):
        self.sr = sample_rate
        self.max_n = int(max_seconds * sample_rate)
        self.min_n = int(min_seconds * sample_rate)
        self.sil_n = int(silence_seconds * sample_rate)
        self.silence_rms = silence_rms
        self.buf = np.zeros(0, dtype=np.float32)

    def push(self, frames):
        """Append frames; return a list of completed chunks (usually 0 or 1)."""
        self.buf = np.concatenate([self.buf, np.asarray(frames, dtype=np.float32).ravel()])
        out = []
        while True:
            n = len(self.buf)
            if n >= self.max_n:
                cut = self.max_n
            elif n >= self.min_n and self._tail_is_silent():
                cut = n
            else:
                break
            chunk, self.buf = self.buf[:cut], self.buf[cut:]
            if np.sqrt(np.mean(chunk ** 2)) >= self.silence_rms:
                out.append(chunk)  # drop chunks that are silence end to end
            # ponytail: a silent chunk is dropped, not merged; good enough for speech
        return out

    def flush(self):
        chunk, self.buf = self.buf, np.zeros(0, dtype=np.float32)
        return chunk if len(chunk) else None

    def _tail_is_silent(self):
        tail = self.buf[-self.sil_n:]
        return np.sqrt(np.mean(tail ** 2)) < self.silence_rms
