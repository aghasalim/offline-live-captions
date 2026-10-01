"""Nothing in the runtime path may open a socket to anything but localhost.

Monkeypatches socket.socket.connect so any non-loopback connection raises.
Then exercises the chunker, the web server and (when weights are cached and
not in CI) a real transcription through the ASR backend.
"""
import os
import socket

import numpy as np
import pytest

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    attempts = []

    def guarded(self, addr):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in LOOPBACK:
            attempts.append(addr)
            raise AssertionError(f"network connection attempted: {addr}")
        return real_connect(self, addr)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return attempts


def test_web_and_chunker_open_only_loopback(no_network):
    from livecap.chunker import Chunker
    from livecap.web import CaptionServer
    Chunker().push(np.zeros(16000, dtype=np.float32))
    srv = CaptionServer(port=0)
    try:
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
        s.close()
    finally:
        srv.close()
    assert no_network == []


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="model weights are not downloaded in CI")
def test_transcribe_opens_no_socket(no_network):
    pytest.importorskip("mlx_whisper")
    from livecap.asr import load
    asr = load("tiny")
    t = np.arange(16000) / 16000
    asr((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    assert no_network == []
