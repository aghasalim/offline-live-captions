"""Text normalisation and WER used in every evaluation."""
import re

import jiwer

_norm = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def normalize(s):
    s = re.sub(r"[‘’]", "'", s)
    return _norm(s)


def wer(refs, hyps):
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) if normalize(h) else "" for h in hyps]
    return jiwer.wer(refs, hyps)
