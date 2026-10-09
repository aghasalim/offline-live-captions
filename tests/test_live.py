"""Ctrl-C must not drop the last chunk: main() has to wait for the worker."""
import sys
import threading
import types

import numpy as np

from livecap import live


class _Ctx:
    def __init__(self, *a, callback=None, **k):
        self.callback = callback

    def __enter__(self):
        if self.callback:  # one second of speech, below the 4 s chunk limit
            self.callback(np.full((16000, 1), 0.5, dtype=np.float32), 16000, None, None)
        return self

    def __exit__(self, *a):
        return False

    def update(self, *a):
        pass

    def status(self, *a):
        return _Ctx()


def test_last_chunk_is_transcribed_before_exit(tmp_path, monkeypatch):
    fake = {
        "sounddevice": types.SimpleNamespace(InputStream=_Ctx),
        "rich.console": types.SimpleNamespace(Console=_Ctx),
        "rich.live": types.SimpleNamespace(Live=_Ctx),
        "rich.panel": types.SimpleNamespace(Panel=lambda *a, **k: None),
        "rich.text": types.SimpleNamespace(Text=lambda *a, **k: None),
    }
    for name, mod in fake.items():
        monkeypatch.setitem(sys.modules, name, mod)

    def slow_asr(audio):
        threading.Event().wait(0.3)  # time.sleep is patched below
        return "last words"
    slow_asr.name = "fake"
    monkeypatch.setattr(live, "load", lambda *a: slow_asr)

    def ctrl_c(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(live.time, "sleep", ctrl_c)

    out = tmp_path / "t.txt"
    assert live.main(["--save", str(out)]) == 0
    assert out.read_text() == "last words\n"
