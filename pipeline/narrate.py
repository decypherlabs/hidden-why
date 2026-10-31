"""Narration: Kokoro's British narrator (brand voice) reads every shot's line. A script's own "voice"
({"voice", "lang", "lang_code", "speed"}) wins: the Hindi editions use Kokoro's Hindi narrator.

    python pipeline/narrate.py episodes/<slug>/script.json out/<slug>/

Writes vo-NN.wav per shot (24 kHz mono) and vo.json ({"NN": seconds}). English phonemes come from
espeak-ng: misaki's default English G2P loads spaCy, which crashes on load on these runners.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brand"))
from brandcfg import B as BRAND  # noqa: E402  (HW_BRAND: another channel's voice)
SR = 24000


def lines(script):
    n = 0
    for sec in script["sections"]:
        for s in sec.get("shots") or []:
            n += 1
            yield n, s["vo"]


def parler(script, out, v):
    """Indic Parler-TTS (AI4Bharat, Apache-2.0): Devlok Katha's Hindi narrator, on a free CPU runner. The voice is
    set by a written description; one seed for every line keeps it the same person. Resampled to 24 kHz."""
    import torch
    import torchaudio.functional as AF
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(v["model"])
    tok, dtok = AutoTokenizer.from_pretrained(v["model"]), AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    d = dtok(v["description"], return_tensors="pt")
    durs = {}
    for n, text in lines(script):
        torch.manual_seed(7)
        p = tok(text, return_tensors="pt")
        a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                           prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask).cpu().float().squeeze()
        x = AF.resample(a.unsqueeze(0), model.config.sampling_rate, SR).squeeze().numpy()
        sf.write(out / f"vo-{n:02d}.wav", x, SR)
        durs[f"{n:02d}"] = round(len(x) / SR, 3)
        print(f"  vo {n:02d}: {durs[f'{n:02d}']:.1f}s", flush=True)
    (out / "vo.json").write_text(json.dumps(durs, indent=1))


def main(script_path, out_dir):
    script = json.loads(Path(script_path).read_text())
    v = {**BRAND["voice"], **(script.get("voice") or {})}
    if v.get("engine") == "parler":
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return parler(script, out, v)
    from kokoro import KPipeline
    from misaki.espeak import EspeakG2P
    tts = KPipeline(lang_code=v.get("lang_code", "e"), repo_id="hexgrad/Kokoro-82M")
    tts.g2p = EspeakG2P(language=v.get("lang", "en-gb"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    durs = {}
    speed = script.get("voice_speed") or v.get("speed", 0.95)  # Shorts: a little faster
    for n, text in lines(script):
        chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                  for _, _, a in tts(text, voice=v["voice"], speed=speed)]
        x = np.concatenate(chunks) if chunks else np.zeros(SR // 2)
        sf.write(out / f"vo-{n:02d}.wav", x, SR)
        durs[f"{n:02d}"] = round(len(x) / SR, 3)
        print(f"  vo {n:02d}: {durs[f'{n:02d}']:.1f}s", flush=True)
    (out / "vo.json").write_text(json.dumps(durs, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
