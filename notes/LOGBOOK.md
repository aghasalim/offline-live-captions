# Logbook

## 2026-10-01

Started the repo. Goal: live captions that run fully offline on my M4, with
the accuracy and latency actually measured rather than quoted.

Python 3.14 is the system interpreter and mlx-whisper has no wheel for it, so
the project uses a 3.12 venv made with uv. mlx-whisper 0.4.3 installed first
try.

Picked LibriSpeech test-other as the test set and fixed a 300-utterance
subset with seed 2026 (`scripts/prepare_subset.py`). The tar is 328 MB.

Looked for an accented English set that needs no sign-up. L2-ARCTIC needs a
form, EdAcc on Hugging Face is gated. `DTU54DL/common-accent` is an ungated
Common Voice repackaging with a self-reported accent column; the test split
is 451 clips and 19 MB. The dataset card is a copy of an unrelated template,
which is sloppy, but the parquet has audio, sentence and accent columns and
the audio decodes. 346 of 451 clips are "India and South Asia" so the
breakdown is unbalanced. Using it with that caveat.

Failure: `huggingface_hub.snapshot_download` stalled at 15 MB on the tiny
model twice, with no error. `curl` from the same URL ran at about 1.4 MB/s,
so the weights are fetched with curl into `models/` and `livecap/asr.py`
prefers that directory when it exists. This also means the runtime never
touches the hub at all, which is what I wanted anyway.

Failure: first WER fixture test was wrong, I miscounted the reference words
(16, not 12). Also jiwer's RemovePunctuation strips apostrophes, so the
normaliser maps "it's" to "its". Kept that and documented it.

Another job (a local LLM benchmark) was running on this machine while I set
things up. Timing runs are done after it finished; see entries below.

Failure: the first full WER run was far slower than the smoke test
suggested. In the 5 dB room condition tiny was at RTF 0.75 instead of 0.03.
Cause: mlx-whisper's default temperature fallback. When the compression
ratio check fails on a hallucinated repeat, it re-decodes at five higher
temperatures. On noisy audio that happens constantly. Set `temperature=0.0`
in `livecap/asr.py` so decoding is purely greedy with no retries; the live
tool uses the same setting, so the evaluation measures what the user gets.
RTF in that condition dropped to 0.03. Restarted the grid.

Headless Chrome could not screenshot the live page because the SSE stream
never closes and the load event never fires. Screenshot is of a static copy
of the same HTML with three lines injected (`results/captions_page.png`).
