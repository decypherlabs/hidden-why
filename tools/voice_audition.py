"""Male narrator audition for Devlok Katha (owner, 6 Nov 2026): Hindi male takes from AI4Bharat's Rasa and
IndicVoices-R (CC-BY 4.0) as references, IndicF5 (MIT) reading the same lines in each voice. Test branch only."""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out/voices")
OUT.mkdir(parents=True, exist_ok=True)
api = HfApi()
LINES = ["क्या आप जानते हैं, धृतराष्ट्र ने भीम को इतनी ज़ोर से भींचा कि वो चूर-चूर हो गया?",
         "लेकिन रुकिए, वो भीम था ही नहीं।",
         "सौ पुत्र खोकर अंधे राजा ने भीम को गले लगाने बुलाया।",
         "कृष्ण इरादा भाँप गए, और आगे कर दी लोहे की मूर्ति।"]


def pitch(a, sr):
    n = int(0.04 * sr)
    f0 = []
    for i in range(0, len(a) - n, n):
        x = a[i:i + n] - a[i:i + n].mean()
        if np.sqrt((x ** 2).mean()) < 0.02:
            continue
        c = np.correlate(x, x, "full")[n - 1:]
        lo, hi = int(sr / 300), int(sr / 60)
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            f0.append(sr / k)
    return float(np.median(f0)) if f0 else 999


def col(t, *names):
    for n in names:
        if n in t.column_names:
            return n
    return None


cands = []
for repo, shards in (("ai4bharat/Rasa", 3), ("ai4bharat/indicvoices_r", 3)):
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("Hindi/") and f.endswith(".parquet"))
    print(repo, len(files), "Hindi files", flush=True)
    for f in files[:shards]:
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
        print(f, t.column_names, t.num_rows, flush=True)
        g, a, tx = col(t, "gender", "Gender", "speaker_gender"), col(t, "audio", "audio_filepath"), col(t, "text", "transcript", "verbatim")
        sp, st = col(t, "speaker_id", "speaker", "spk_id"), col(t, "style", "emotion")
        rows = t.to_pylist()
        for r in rows:
            if g and str(r.get(g)).lower() not in ("male", "m"):
                continue
            au = r[a]
            try:
                y, sr = sf.read(io.BytesIO(au["bytes"]))
            except Exception:  # noqa: BLE001
                continue
            if y.ndim > 1:
                y = y.mean(1)
            d = len(y) / sr
            if not 5 <= d <= 12:
                continue
            if "snr" in r and (r.get("snr") or 0) < 30:
                continue  # studio-clean takes only
            spk = str(r.get(sp)) if sp else "_".join(str(r.get("filename", "")).split("_")[:2])
            cands.append({"repo": repo, "speaker": spk, "style": str(r.get(st)), "text": r[tx], "dur": d,
                          "y": y, "sr": sr})
print(len(cands), "male clips", flush=True)
for c in cands:
    c["f0"] = pitch(c["y"], c["sr"])
best = {}
for c in sorted(cands, key=lambda c: c["f0"]):  # deepest take of each speaker
    k = (c["repo"], c["speaker"])
    if k not in best and c["f0"] < 160:
        best[k] = c
picks = list(best.values())[:6]
for i, c in enumerate(picks, 1):
    sf.write(OUT / f"ref-{i}.wav", c["y"], c["sr"])
    print("ref", i, c["repo"], c["speaker"], c["style"], round(c["f0"]), flush=True)
from transformers import AutoModel  # noqa: E402

model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, low_cpu_mem_usage=False)
summary = []
for i, c in enumerate(picks, 1):
    ref = OUT / f"ref-{i}.wav"
    parts = []
    for line in LINES:
        y = model(line, ref_audio_path=str(ref), ref_text=c["text"])
        y = np.asarray(y, dtype=np.float32)
        if y.dtype == np.int16 or np.abs(y).max() > 1.5:
            y = y / 32768.0
        parts += [y, np.zeros(6000, dtype=np.float32)]
    sf.write(OUT / f"voice-{i}.wav", np.concatenate(parts), 24000)
    summary.append({k: c[k] for k in ("repo", "speaker", "style", "text", "dur", "f0")})
    print(i, summary[-1], flush=True)
(OUT / "voices.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
