# Offline live captions

Live captions that run entirely on the laptop, with the word error rate and
latency measured on this machine instead of quoted from a model card.

![captions web page](results/captions_page.png)

## Why

I read through r/deaf and r/hardofhearing threads on caption apps and the
complaints repeat: a monthly minute cap ("I had to make a throwaway account
just to reset the clock"), apps that stop the moment wifi is off ("I turned
off wifi and data and transcribe stopped working"), lag that makes a
conversation awkward, and a company saying nobody would believe a "processing
is local" claim because they cannot check it. Open source with a test that
fails if the program opens a non-local socket answers that last one. The
other three are answered by running Whisper on the machine. What nobody
publishes is how accurate that actually is on a consumer laptop, in a room
that is not a studio, for speakers who do not sound like audiobook narrators.
So this repo measures it.

## Results (Apple M4, 24 GB, mlx-whisper 0.4.3, greedy decoding)

Word error rate on 300 LibriSpeech test-other utterances, clean and in a
simulated room (synthetic reverb plus six-speaker babble at a fixed SNR).
The room is simulated, not recorded; see METHODOLOGY.md for exactly how.

<!-- table:wer -->
| model | clean | room, babble 15 dB | room, babble 10 dB | room, babble 5 dB | RTF (clean) |
|---|---|---|---|---|---|
| tiny | 17.2% | 50.5% | 65.3% | 104.1% | 0.025 |
| base | 13.9% | 38.3% | 61.5% | 88.1% | 0.069 |
| small | 9.2% | 21.4% | 35.0% | 58.9% | 0.181 |
| large-v3-turbo | 5.1% | 11.0% | 17.7% | 43.3% | 0.427 |

WER on 300 LibriSpeech test-other utterances. RTF is processing time divided by audio duration on the M4.
<!-- /table -->

On clean read speech the small model gets
**9.2%** of words wrong and large-v3-turbo gets **5.1%**; tiny gets
**17.2%**, which is already enough that a sentence will often contain a
wrong word. Once you add reverb and competing talkers at 5 dB the small model
is at **58.9%**: in a loud room every model here is a rough aid, not a
transcript. Real time factor is well under 1 for all four, so none of them
falls behind live speech on this machine; small runs at **0.181**.

End-of-speech to caption latency, by maximum chunk length:

<!-- table:latency -->
| model | 2 s chunk | 3 s chunk | 4 s chunk | 6 s chunk | 8 s chunk |
|---|---|---|---|---|---|
| tiny | 0.47 / 0.49 | 0.47 / 0.51 | 0.48 / 0.51 | 0.48 / 0.54 | 0.49 / 0.57 |
| base | 0.52 / 0.55 | 0.52 / 0.57 | 0.54 / 0.59 | 0.53 / 0.61 | 0.53 / 0.63 |
| small | 0.73 / 0.80 | 0.75 / 0.82 | 0.76 / 0.87 | 0.76 / 0.92 | 0.77 / 0.97 |
| large-v3-turbo | 1.48 / 1.71 | 1.44 / 1.49 | 1.69 / 1.79 | 1.72 / 1.82 | 1.73 / 1.89 |

Median / 90th percentile end-of-speech to caption latency in seconds, 60 utterances per cell. Includes the 0.4 s pause the chunker waits for.
<!-- /table -->

With the default 4 s chunk, the small model shows the final words of a
sentence a median latency of **0.76 s** after the speaker stops, and
large-v3-turbo a median latency of **1.69 s**. For a conversation that
is the difference between the caption arriving while the other person is
still looking at you and arriving after they have already started the next
sentence. For tiny and base most of that wait is the fixed 0.4 s pause the chunker
needs before it decides the sentence ended, not the model. The maximum chunk
length barely moves the number, because the chunk that holds the last words
of a sentence is usually the short remainder after the previous cut, not a
full-length chunk; what the chunk length changes is how often captions
appear mid-sentence, not how late the last word is.

Word error rate by self-reported accent on the `DTU54DL/common-accent` test
split (Common Voice clips, crowd-recorded, so absolute numbers are not
comparable to the LibriSpeech table):

<!-- table:accent -->
| accent (self-reported) | clips | tiny | base | small | large-v3-turbo |
|---|---|---|---|---|---|
| India and South Asia | 346 | 35.2% | 26.9% | 17.3% | 12.9% |
| Southern African | 23 | 34.5% | 33.5% | 23.5% | 22.5% |
| Hong Kong | 16 | 42.8% | 31.7% | 24.1% | 9.0% |
| West Indies and Bermuda | 13 | 32.6% | 28.0% | 15.9% | 12.9% |
| Malaysian | 11 | 44.8% | 32.3% | 22.9% | 10.4% |
| all | 409 | 35.6% | 27.6% | 17.9% | 13.2% |
<!-- /table -->

For India and South Asia, the one group with a sample size that means
something, the small model is at **17.3%**. The other groups have 11 to
23 clips each and their numbers should be read as rough.

Figures: `results/wer_vs_model.png`, `results/latency_vs_chunk.png`,
`results/rtf_per_model.png`, `results/wer_by_accent.png`.

## Limitations

* The far-field condition is a simulation (synthetic RIR, babble from the
  same corpus). It shows the direction and rough size of the degradation,
  not what a specific café does.
* LibriSpeech is read audiobook speech and a public set that Whisper may
  have seen during training. Conversational speech will be worse.
* The accent set is unbalanced (346 of 451 clips are one group), the labels
  are self-reported, and there is no native speaker control from the same
  source, so it cannot say how much accented speech costs, only how the
  models rank within it.
* One machine, one run per cell, no confidence intervals. Differences under
  about a percentage point of WER are noise at this sample size.
* Latency is measured on the processing path by replaying audio, not with a
  microphone loop. Display cost is a few milliseconds and not counted.
* Greedy decoding only, with no temperature fallback (see the logbook for
  why: the fallback retries made noisy audio 20 times slower).
* RTF was measured while another process (a local LLM benchmark I could not
  stop) was using part of the GPU, so the RTF column is an upper bound. The
  latency run started after that process had gone quiet.

## Run it

Apple Silicon, Python 3.12 (mlx-whisper has no 3.14 wheel yet):

    uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt -e .
    mkdir -p models/whisper-small-mlx
    curl -L -o models/whisper-small-mlx/config.json https://huggingface.co/mlx-community/whisper-small-mlx/resolve/main/config.json
    curl -L -o models/whisper-small-mlx/weights.npz https://huggingface.co/mlx-community/whisper-small-mlx/resolve/main/weights.npz
    .venv/bin/python -m livecap --model small --web

Then open http://127.0.0.1:8765/ for the large caption page (search box,
save as .txt). The terminal shows the same captions. `--chunk 2` trades
accuracy for speed. Without mlx-whisper the tool falls back to
`pywhispercpp` or `faster-whisper` if either is installed (`--backend`).

Reproduce the numbers:

    .venv/bin/python scripts/prepare_subset.py       # needs data/LibriSpeech/test-other
    .venv/bin/python scripts/eval_wer.py
    .venv/bin/python scripts/eval_latency.py
    .venv/bin/python scripts/eval_accent.py          # needs data/common-accent-test.parquet
    .venv/bin/python scripts/make_figures.py
    .venv/bin/python scripts/check_numbers.py        # fails if README disagrees with results/

## Privacy, checkable

`tests/test_no_network.py` patches `socket.connect` so that any connection
to a non-loopback address raises, then runs the chunker, the caption web
server and a real transcription through the model. It passes. Model weights
are read from `models/` on disk; `HF_HUB_OFFLINE=1` is set so the Hugging
Face library cannot phone home either. The web page is a single file served
on 127.0.0.1 with no external assets. If you do not trust any of that,
unplug the network and run it.

## Layout

    livecap/        chunker, ASR backends, terminal UI, web page
    scripts/        subset selection, evaluations, figures, check_numbers
    results/        CSVs and figures that the README numbers come from
    tests/          pytest; CI skips the model test with CI=true
    notes/LOGBOOK.md  dated notes including what went wrong

MIT licence.
