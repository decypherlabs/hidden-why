"""The narrator's sound treatment, applied to the finished narration (owner, 15 Nov 2026: picked "1-aman-now-heavy",
the niche's heavy voice with bass): pitch ~6% lower at the same speed, a bass shelf, a touch of presence, compression
and a short hall echo. brand.json "voice_fx": {"lanes": [...] or null for every episode, "filter": "<ffmpeg -af>"}.
Done at build time on the voice track, so voiceovers already made get it too and nothing is narrated again."""
import subprocess

import numpy as np

DEFAULT = ("asetrate={sr}*0.94,aresample={sr},atempo=1/0.94,bass=g=7:f=110:w=0.7,equalizer=f=3000:t=q:w=1:g=2,"
           "acompressor=threshold=-20dB:ratio=3:attack=10:release=200:makeup=3,aecho=0.8:0.6:45:0.18")


def wanted(brand, script):
    fx = (brand or {}).get("voice_fx")
    if not fx:
        return None
    lanes = fx.get("lanes")
    return fx if lanes is None or script.get("lane") in lanes else None


def apply(x, sr, brand, script):
    """x: mono float32 at sr; returned the same length (the echo tail past the end is cut)."""
    fx = wanted(brand, script)
    if fx is None or not len(x):
        return x
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-af", (fx.get("filter") or DEFAULT).format(sr=sr), "-f", "f32le", "-ar", str(sr), "pipe:1"],
                       input=np.asarray(x, np.float32).tobytes(), capture_output=True, check=True)
    y = np.frombuffer(r.stdout, np.float32)
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)].astype(np.float32)
