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

WER grid finished (4 models x 4 conditions x 300 utterances). The simulated
room is harsher than I expected: tiny goes from 17% clean to 50% at 15 dB
and above 100% at 5 dB (insertions from hallucinated text push WER past 1).
large-v3-turbo holds 5% clean, 11% at 15 dB, 43% at 5 dB. Checked a few 5 dB
transcripts by hand: the babble is six full-volume speakers plus a reverb
tail with as much energy as the direct sound, which is a loud bar, not a
quiet office. Kept the condition, said so in the README, and added the
15 dB column so there is a milder point.

Timing caveat: a local LLM benchmark (ollama, qwen3 8b) was running on this
machine through the whole WER grid, at 20 to 50 percent of a core and some
GPU. I could not stop it. RTF numbers from the grid are therefore an upper
bound; base at RTF 0.07 looks about twice what I would expect from the
tiny to base size ratio. The latency run was started after the benchmark
had dropped to a few percent CPU.

Latency finished. Result I did not expect: the maximum chunk length hardly
changes end-of-speech latency (small: 0.73 s at 2 s chunks, 0.77 s at 8 s).
The final words of an utterance nearly always sit in the short remainder
after the last cut, so the model only has to decode a second or two
whatever the maximum is. Chunk length decides how often mid-sentence
captions appear, not how late the last word is. Wrote that into the README
rather than the claim I had planned to make.

Failure: `check_numbers.py --write` with empty table bodies matched across
two markers and deleted a README paragraph. Fixed the regex (bodies now
always contain a newline) and retyped the paragraph.

Accent set: large-v3-turbo at 12.9% on India and South Asia (346 clips),
small at 17.3%. Tiny is above 35% everywhere, which for a Common Voice
sentence of ten words means three or four wrong.
