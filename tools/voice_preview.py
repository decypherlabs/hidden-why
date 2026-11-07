"""The narration as the viewer will hear it, for the owner to judge before painting (7 Nov 2026): every vo-NN.wav in
order with the edit's own gaps (build.py: 0.15 s lead-in + BREATH after each shot, 2.4 s for a chapter card) and
a soft beep where a chapter card falls, so the hook, pauses and pace can be checked. Writes preview.mp3.

    python tools/voice_preview.py episodes/<slug>/script.json out/<slug>/
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR, BREATH, CARD = 24000, 0.45, 2.4


def main(script_path, out_dir):
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    parts, n = [], 0
    t = np.arange(int(0.12 * SR)) / SR
    beep = (0.08 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)
    for sec in script["sections"]:
        if sec.get("card"):
            gap = np.zeros(int(CARD * SR), np.float32)
            gap[int(0.3 * SR):int(0.3 * SR) + len(beep)] = beep
            parts.append(gap)
        for _ in sec.get("shots") or []:
            n += 1
            x, sr = sf.read(out / f"vo-{n:02d}.wav", dtype="float32")
            if sr != SR:
                x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x).astype(np.float32)
            parts += [np.zeros(int(0.15 * SR), np.float32), x, np.zeros(int(BREATH * SR), np.float32)]
    sf.write(out / "preview.wav", np.concatenate(parts), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "preview.wav"), "-af", "loudnorm=I=-16:TP=-1.5",
                    "-b:a", "96k", str(out / "preview.mp3")], check=True)
    (out / "preview.wav").unlink()
    print(f"preview.mp3: {sum(len(p) for p in parts) / SR / 60:.1f} min, {n} lines")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
