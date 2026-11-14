"""Narration: Kokoro's British narrator (brand voice) reads every shot's line. A script's own "voice"
({"voice", "lang", "lang_code", "speed"}) wins: the Hindi editions use Kokoro's Hindi narrator.

    python pipeline/narrate.py episodes/<slug>/script.json out/<slug>/

Writes vo-NN.wav per shot (24 kHz mono) and vo.json ({"NN": seconds}). English phonemes come from
espeak-ng: misaki's default English G2P loads spaCy, which crashes on load on these runners.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brand"))
from brandcfg import B as BRAND  # noqa: E402  (HW_BRAND: another channel's voice)
SR = 24000
SEEDS = (7, 11, 23)  # the first is the narrator's usual take; the others only for a line that failed the check


def checker():
    """Whisper (faster-whisper, small, CPU) to hear each Hindi line back (owner, 9 Nov 2026: "make sure that voiceover
    pronounce things correctly"); None when it isn't installed (the take is then kept as it is)."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("small", device="cpu", compute_type="int8")
    except Exception as e:  # noqa: BLE001
        print(f"voice check off: {e}", flush=True)
        return None


def heard_ok(model, x, text, lang="hi"):
    """(score 0-1, what Whisper heard): how close the spoken line is to the written one, letters only. x: the take at
    SR, handed over as 16 kHz samples (no file decoding: PyAV's API moved under faster-whisper, 9 Nov 2026)."""
    import difflib
    import re
    norm = lambda t: re.sub(r"[\s.,।॥!?…'\"\-–—:;()\[\]]+", "", t)  # punctuation and spaces only: \W would drop the matras
    a = np.interp(np.arange(0, len(x), SR / 16000), np.arange(len(x)), x).astype(np.float32)
    segs, _ = model.transcribe(a, language=lang, beam_size=1, vad_filter=False)
    got = "".join(x.text for x in segs)
    return difflib.SequenceMatcher(None, norm(text), norm(got)).ratio(), got.strip()


def pace_ok(x, text):
    """A take far too long or short for its words is garbled or cut (Parler sometimes rambles or stops early)."""
    per_word = len(x) / SR / max(1, len(text.split()))
    return 0.18 <= per_word <= 1.1


def lines(script):
    n = 0
    for sec in script["sections"]:
        for s in sec.get("shots") or []:
            n += 1
            yield n, s["vo"]


def shlokas(script):
    """{line number: the Sanskrit verse inside it}, from each shot's "shloka" (the writer marks it, 14 Nov 2026)."""
    n, out = 0, {}
    for sec in script["sections"]:
        for s in sec.get("shots") or []:
            n += 1
            if s.get("shloka") and s["shloka"] in s["vo"]:
                out[n] = s["shloka"]
    return out


def pieces(text, verse, conf):
    """A line cut where a narrator breathes (owner, 14 Nov 2026: "pauses are missing... majorly the shloks"): every
    sentence on its own, and a shloka recited half-verse by half-verse, slower, with air around it. Parler reads a
    whole line in one breath and runs straight through a "।". [(text, is_verse, silence after)]."""
    import re
    say = lambda t: [x for x in re.split(r"(?<=[।?!])\s+", t.strip()) if x.strip()]
    gap, around, pada = float(conf.get("sentence_pause", 0)), float(conf.get("verse_pause", 0.5)), float(conf.get("pada_pause", 0.4))
    if not gap and not verse:
        return [(text, False, 0.0)]
    before, after = text.split(verse, 1) if verse else (text, "")
    out = [(t, False, gap) for t in (say(before) if gap else [before.strip()] if before.strip() else [])]
    if verse:
        if out:
            out[-1] = (out[-1][0], False, around)
        out += [(t, True, pada) for t in re.split(r"(?<=[,।॥;])\s*", verse.strip()) if t.strip()]
        out[-1] = (out[-1][0], True, around)
    out += [(t, False, gap) for t in (say(after) if gap else [after.strip()] if after.strip() else [])]
    if not out[-1][1]:
        out[-1] = (out[-1][0], False, 0.0)  # the edit's own breath follows the line (a verse keeps its air too)
    return out


def trim(x, pad=0.06):
    """Parler pads a line with silence at both ends (owner, 7 Nov 2026: "a lot of pauses in the voiceover"): cut it
    to `pad` seconds, so the pause between lines is the edit's own breath, not the model's. Pauses inside a line
    (commas) stay."""
    hop = SR // 100
    rms = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(hop) / hop, mode="same"))
    on = np.where(rms > max(1e-4, 0.03 * rms.max()))[0]
    if not len(on):
        return x
    return x[max(0, on[0] - int(pad * SR)):min(len(x), on[-1] + int(pad * SR))]


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
    durs, check = {}, {}
    ear = checker() if v.get("check", True) else None
    floor = float(v.get("check_floor", 0.55))
    conf = BRAND.get("narration") or {}  # pauses: outside "voice", so older narrations keep their fingerprint
    verses = shlokas(script)

    def say(text, seed, slow=None, cut=False):
        torch.manual_seed(seed)
        p = tok(text, return_tensors="pt")
        a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                           prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask).cpu().float().squeeze()
        x = AF.resample(a.unsqueeze(0), model.config.sampling_rate, SR).squeeze().numpy()
        x = trim(x) if v.get("trim") or cut else x  # a piece of a line is always cut: the pauses are ours
        if slow:  # a shloka is recited, not read: slower, same voice (time-stretch keeps the pitch)
            r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                                "-af", f"atempo={slow}", "-f", "f32le", "pipe:1"], input=x.astype(np.float32).tobytes(),
                               capture_output=True, check=True)
            x = np.frombuffer(r.stdout, np.float32)
        return x

    for n, text in lines(script):
        best = None
        parts = pieces(text, verses.get(n), conf)
        for seed in SEEDS:  # the usual take; another only when it isn't heard right (pace, Whisper)
            if len(parts) == 1:
                x = say(text, seed)
            else:
                x = np.concatenate([np.concatenate([say(t, seed, float(conf.get("verse_tempo", 0.85)) if verse else None, cut=True),
                                                    np.zeros(int(pause * SR), np.float32)])
                                    for t, verse, pause in parts]).astype(np.float32)
            score, got = 1.0, ""
            if ear is not None:
                try:
                    score, got = heard_ok(ear, x, text)
                except Exception as e:  # noqa: BLE001 - the check never stops the narration
                    print(f"voice check off: {e}", flush=True)
                    ear = None
            score = score if pace_ok(x, text) else min(score, 0.3)
            if best is None or score > best[0]:
                best = (score, x, got, seed)
            if score >= floor:
                break
        score, x, got, seed = best
        check[f"{n:02d}"] = {"score": round(score, 2), "seed": seed, **({"heard": got} if score < floor else {})}
        if score < floor:
            print(f"  vo {n:02d}: still unclear after {len(SEEDS)} takes ({score:.2f}): heard {got!r}", flush=True)
        sf.write(out / f"vo-{n:02d}.wav", x, SR)
        if v.get("tempo", 1) != 1 or v.get("pitch", 1) != 1:  # faster delivery (owner, 6 Nov 2026: hits speak ~3 words/s, Divya ~2)
            tmp = out / f"vo-{n:02d}.raw.wav"  # "pitch" < 1 deepens the voice without slowing it (owner picked 0.92 + 1.15x)
            (out / f"vo-{n:02d}.wav").rename(tmp)
            p = v.get("pitch", 1)
            af = f"asetrate={int(SR * p)},aresample={SR},atempo={v.get('tempo', 1) / p:.4f}" if p != 1 else f"atempo={v['tempo']}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-af", af,
                            str(out / f"vo-{n:02d}.wav")], check=True)
            tmp.unlink()
            x, _ = sf.read(out / f"vo-{n:02d}.wav")
        durs[f"{n:02d}"] = round(len(x) / SR, 3)
        print(f"  vo {n:02d}: {durs[f'{n:02d}']:.1f}s", flush=True)
    (out / "vo.json").write_text(json.dumps(durs, indent=1))
    bad = [k for k, c in check.items() if c["score"] < floor]
    (out / "voice_check.json").write_text(json.dumps({"floor": floor, "bad": bad, "lines": check}, indent=1, ensure_ascii=False))


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
    if tts.lang_code in ("a", "b"):  # lang_code a/b makes Kokoro tokenize misaki-English tokens, which EspeakG2P returns as None
        tts.lang_code = "e"  # (the voice is picked by name, so only the phonemizer path changes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    durs = {}
    # never slower than the brand's own speed (owner, 13 Nov 2026: Shorts at the scripts' 1.05 sounded slower than the
    # chosen George at 1.15); a script may only ask for faster
    speed = max(float(script.get("voice_speed") or 0), float(v.get("speed", 0.95)))
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
