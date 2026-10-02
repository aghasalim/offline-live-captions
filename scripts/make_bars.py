"""Animated bar chart of the WER grid for the README. Reads results/wer_summary.csv."""
import csv, pathlib
W, BAR_H, GAP, LABEL_W, PAD = 720, 22, 10, 170, 16
FG, MUT = "#8b949e", "#6e7681"
COLORS = ["#2f81f7", "#3fb950", "#d29922", "#f85149", "#a371f7"]


def chart(path, title, rows, unit, vmax, legend=None, note=""):
    """rows: list of (label, [values per series]). Bars animate with SMIL so they
    play inside an <img>, which is how GitHub shows README images. The static
    attributes already hold the final value, so animations off still shows the bar."""
    nseries = len(rows[0][1])
    inner = BAR_H if nseries == 1 else int(BAR_H * 0.8)
    row_h = inner * nseries + GAP
    h = PAD + 30 + len(rows) * row_h + (26 if legend else 0) + (18 if note else 0) + PAD
    scale = (W - LABEL_W - PAD * 2 - 60) / vmax
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="13">',
         f'<text x="{PAD}" y="{PAD + 12}" fill="{FG}" font-size="15" font-weight="600">{title}</text>']
    y = PAD + 30
    for i, (label, vals) in enumerate(rows):
        s.append(f'<text x="{LABEL_W - 8}" y="{y + (inner * nseries) / 2 + 5}" fill="{FG}" text-anchor="end">{label}</text>')
        for j, v in enumerate(vals):
            by = y + j * inner
            wv = max(2, v * scale)
            delay = 0.08 * i + 0.05 * j
            s.append(f'<rect x="{LABEL_W}" y="{by}" width="{wv:.1f}" height="{inner - 3}" rx="3" fill="{COLORS[j]}">'
                     f'<animate attributeName="width" from="0" to="{wv:.1f}" begin="{delay:.2f}s" dur="1s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/></rect>')
            s.append(f'<text x="{LABEL_W + wv + 6:.1f}" y="{by + inner / 2 + 3}" fill="{MUT}" font-size="12">{v:g}{unit}'
                     f'<animate attributeName="opacity" from="0" to="1" begin="{delay + 0.9:.2f}s" dur="0.4s" fill="freeze"/></text>')
        y += row_h
    if legend:
        x = LABEL_W
        for j, name in enumerate(legend):
            s.append(f'<rect x="{x}" y="{y + 4}" width="12" height="12" rx="2" fill="{COLORS[j]}"/>')
            s.append(f'<text x="{x + 17}" y="{y + 14}" fill="{MUT}" font-size="12">{name}</text>')
            x += 17 + 7 * len(name) + 24
        y += 26
    if note:
        s.append(f'<text x="{PAD}" y="{y + 10}" fill="{MUT}" font-size="11">{note}</text>')
    s.append('</svg>')
    path.write_text("\n".join(s))

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = {}
for r in csv.DictReader(open(ROOT / "results/wer_summary.csv")):
    rows.setdefault(r["model"], {})[r["condition"]] = round(float(r["wer"]) * 100, 1)
order = ["tiny", "base", "small", "large-v3-turbo"]
data = [(m, [rows[m]["clean"], rows[m]["room_snr15"], rows[m]["room_snr5"]]) for m in order]
chart(ROOT / "results/wer_bars.svg", "Word error rate on 300 LibriSpeech test-other utterances, Apple M4",
      data, "%", 110, ["clean", "room, 15 dB babble", "room, 5 dB babble"],
      "Lower is better. tiny passes 100% at 5 dB because it inserts words that were never spoken.")
