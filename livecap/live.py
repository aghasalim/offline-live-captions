"""Microphone -> chunker -> ASR -> terminal (rich) and optional local web page."""
import argparse
import queue
import sys
import threading
import time

import numpy as np

from .asr import load
from .chunker import SAMPLE_RATE, Chunker


def main(argv=None):
    ap = argparse.ArgumentParser(prog="livecap", description="offline live captions")
    ap.add_argument("--model", default="base", help="tiny, base, small, large-v3-turbo")
    ap.add_argument("--backend", default=None, help="mlx, whispercpp or faster")
    ap.add_argument("--chunk", type=float, default=4.0, help="max seconds per chunk")
    ap.add_argument("--web", action="store_true", help="serve captions on http://127.0.0.1:8765")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--save", default=None, help="append transcript lines to this file")
    args = ap.parse_args(argv)

    import sounddevice as sd
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    with console.status(f"loading {args.model}"):
        asr = load(args.model, args.backend)
    server = None
    if args.web:
        from .web import CaptionServer
        server = CaptionServer(port=args.port)
        console.print(f"[bold]captions at http://127.0.0.1:{server.port}/[/bold]")

    chunker = Chunker(max_seconds=args.chunk)
    chunks = queue.Queue()
    lines = []
    outfile = open(args.save, "a") if args.save else None

    def on_audio(indata, frames, t, status):
        for c in chunker.push(indata[:, 0]):
            chunks.put((c, time.monotonic()))

    def worker():
        while True:
            item = chunks.get()
            if item is None:
                return
            audio, t_end = item
            text = asr(audio)
            if not text:
                continue
            lag = time.monotonic() - t_end
            lines.append(text)
            if outfile:
                outfile.write(text + "\n"); outfile.flush()
            if server:
                server.publish(text)
            live.update(render(lines, lag))

    def render(lines, lag):
        t = Text("\n".join(lines[-6:]), style="bold white")
        return Panel(t, title=f"{asr.name}  chunk {args.chunk:.0f}s  lag {lag:.2f}s",
                     subtitle="ctrl-c to stop", padding=(1, 2))

    threading.Thread(target=worker, daemon=True).start()
    with Live(render(lines, 0.0), console=console, refresh_per_second=8) as live:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=1600, callback=on_audio):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        tail = chunker.flush()
        if tail is not None:
            chunks.put((tail, time.monotonic()))
        chunks.put(None)
    if server:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
