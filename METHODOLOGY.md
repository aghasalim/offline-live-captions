# Methodology

## What is being measured

Two things matter for a live caption tool used in a conversation: how many
words it gets wrong, and how long after someone stops speaking the words
appear. Both are measured on this machine (Apple M4, 24 GB, macOS, mlx-whisper
0.4.3) and nothing is taken from model cards or papers.

## Test set

LibriSpeech test-other, the harder half of the standard LibriSpeech test data
(read audiobooks, speakers the models were not selected on). The full split is
2939 utterances; `scripts/prepare_subset.py` picks 300 with a fixed seed
(2026) and writes `data/subset.csv`, so the subset is reproducible from the
public tar without any file from this repo. 300 was chosen to keep the full
grid (4 models x 3 conditions) under an hour.

Whisper was trained on web audio and LibriSpeech is a well known public set,
so some contamination is possible. That is a limitation shared by every
Whisper evaluation on public data; the relative ranking between models and
conditions is what this repo is for.

## Scoring

Word error rate with `jiwer` after lowercasing, removing punctuation and
collapsing whitespace (`livecap/text.py`). Apostrophes are removed as well,
so "its" and "it's" count as a match. No number or spelling normalisation is
applied, so "twenty" versus "20" counts as an error; LibriSpeech references
spell numbers out and Whisper mostly does too, so this affects few words.
WER is total errors over total reference words across the subset, not an
average of per-utterance rates.

## Far-field condition (simulated)

There is no real café recording here. The "room" condition is built in
`livecap/simulate.py`:

1. Convolve with a synthetic room impulse response: a unit direct path, then
   after 8 ms an exponentially decaying Gaussian noise tail with RT60 = 0.5 s,
   at equal energy to the direct path. This is a textbook late-reverb model.
2. Add babble: six other test-other speakers not in the subset, loudness
   matched, summed, randomly offset. The babble is scaled to a fixed
   signal to noise ratio of 10 dB or 5 dB against the reverberated speech.

The RIR seed and the noise offsets are fixed per utterance index, so a rerun
produces bit-identical audio. Everything stays at 16 kHz mono.

What this does and does not tell you: it shows how each model degrades under
reverb plus competing speech at known SNRs. It does not reproduce microphone
response, distance-dependent level loss, or the spectral shape of a real
room. Expect real rooms to be somewhat worse than the 10 dB column and
plausibly comparable to the 5 dB column.

## Accent condition

`DTU54DL/common-accent` on Hugging Face is an ungated repackaging of Common
Voice English clips with the speaker's self-reported accent field. Its test
split has 451 clips. Only accent groups with at least 10 clips are scored:
India and South Asia, Southern African, Hong Kong, West Indies and Bermuda,
Malaysian. Caveats: the labels are self-reported free text, the groups are
very unbalanced (most clips are India and South Asia), Common Voice sentences
are short, and the clips are crowd-recorded on consumer microphones so they
are noisier than LibriSpeech. The column is therefore not comparable to the
LibriSpeech columns; compare accents against each other within the table.
There is no native English control group of the same source in this split,
so the table cannot say how much worse accented speech is than unaccented
speech, only how the models rank across these accents.

## Real time factor

Processing wall time divided by audio duration, summed over the subset, on
the clean condition, with the model already loaded. Below 1.0 means the
model keeps up with live speech on this machine.

## Latency

`scripts/eval_latency.py` replays each of the first 60 subset utterances
through the same `Chunker` class the live tool uses, with the given maximum
chunk length. The chunk that contains the end of the utterance is timed
through the ASR. Reported latency is:

    pause wait (0.4 s, the silence the chunker needs to see before cutting)
    + ASR wall time for that final chunk

Terminal and browser rendering are not counted; both are a few milliseconds.
Audio is not played in real time, so the measurement is of the processing
path, not of a microphone. Queueing is also not modelled: if RTF were above
1, chunks would back up and real latency would grow without bound, which is
why RTF is reported alongside.

Median and 90th percentile over the 60 utterances are reported. A longer
maximum chunk means the final chunk is on average longer, so the ASR step
takes longer; shorter chunks give the model less context and can split
words. The default of 4 s in the live tool is a compromise.

## What was not done

* No real room recordings, no real distance tests.
* No native English accent control from the same source as the accent set.
* Only greedy decoding; beam search would be slower and slightly more accurate.
* Latency measured on the processing path only, not with a microphone loop.
* One machine, one run per cell. No confidence intervals; with 300
  utterances, differences under about a percentage point of WER should not
  be read as meaningful.
