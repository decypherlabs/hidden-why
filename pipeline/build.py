"""The edit: paintings + diagrams + narration + music -> the 1920x1080 episode.

    python pipeline/build.py episodes/<slug>/script.json out/<slug>/   ->  video.mp4, chapters.txt

Timeline per the brand: cold open, 2 s logo sting, then each section behind a chapter card (ink, yellow
Anton title, red underline). Every shot lasts its narration + a breath, with a slow push-in, pull-out or
pan over the painting (supersampled 2x, no shimmer), 0.4 s dissolves. Music by section mood (licensed library tracks
in brand/music), kept low under the voice. Ends on a 6 s end screen (logo) for YouTube's end elements.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brand"))
sys.path.insert(0, str(ROOT / "pipeline"))
import brand_kit as K  # noqa: E402
from diagram import DIAGRAMS, OVERLAYS  # noqa: E402

W, H, FPS, SR = 1920, 1080, 30, 48000
XF = 0.4  # dissolve seconds
BREATH = 0.45


def bgr(img):
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def card(title):
    img = Image.new("RGB", (W, H), K.INK)
    d = ImageDraw.Draw(img)
    f = K.display(150)
    d.text((W / 2, H / 2 - 20), title, font=f, fill=K.YEL, anchor="mm")
    tw = d.textlength(title, font=f)
    d.rectangle((W / 2 - tw * 0.18, H / 2 + 80, W / 2 + tw * 0.18, H / 2 + 92), fill=K.RED)
    return bgr(K.grain(img))


def sting_frames(n):
    logo = Image.open(ROOT / "brand/logo.png").convert("RGBA")
    out = []
    for k in range(n):
        u = k / max(1, n - 1)
        img = Image.new("RGB", (W, H), K.INK)
        s = int(H * 0.42 * (0.9 + 0.1 * u))
        lg = logo.resize((s, s), Image.LANCZOS)
        a = min(1, u / 0.25) * min(1, (1 - u) / 0.2)
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * a)))
        img.paste(lg, ((W - s) // 2, (H - s) // 2 - 40), lg)
        d = ImageDraw.Draw(img)
        K.wordmark(d, W / 2, H / 2 + s / 2 + 30, 64, anchor="mm") if a > 0.5 else None
        out.append(bgr(img))
    return out


SS = 2  # supersampling: frames are rendered at 2x and averaged down, so slow zooms/pans don't shimmer


class Shot:
    """A painting or diagram under a slow camera move. The source is upscaled once (Lanczos) to cover the
    2x canvas; every frame is warped with a cubic filter at 2x and area-averaged down to 1080p: sub-pixel
    motion stays smooth instead of crawling (the pixel jitter of a 1x linear warp)."""

    def __init__(self, img, move):
        h, w = img.shape[:2]
        f = max(SS * W / w, SS * H / h) * 1.12
        self.src = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_LANCZOS4)
        self.move = move

    def frame(self, u):
        h, w = self.src.shape[:2]
        CW, CH = W * SS, H * SS
        s0 = max(CW / w, CH / h) * 1.02
        z = {"push": 1.0 + 0.09 * u, "pull": 1.09 - 0.09 * u}.get(self.move, 1.05)
        s = s0 * z
        half_w, half_h = CW / (2 * s), CH / (2 * s)
        cx = w / 2
        if self.move in ("pan_left", "pan_right"):
            span = max(0.0, w / 2 - half_w)
            cx = w / 2 + span * (0.8 * (u - 0.5)) * (1 if self.move == "pan_right" else -1) * 2
        cx = float(np.clip(cx, half_w, w - half_w))
        cy = float(np.clip(h / 2, half_h, h - half_h))
        M = np.float32([[s, 0, CW / 2 - s * cx], [0, s, CH / 2 - s * cy]])
        big = cv2.warpAffine(self.src, M, (CW, CH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


class Still:
    def __init__(self, img):
        self.img = img

    def frame(self, u):
        return self.img


class Seq:
    def __init__(self, frames):
        self.frames = frames

    def frame(self, u):
        return self.frames[min(len(self.frames) - 1, int(u * len(self.frames)))]


def load_mp3(path, seconds):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    reps = int(np.ceil(seconds * SR / max(1, len(x))))
    return np.tile(x, reps)[:int(seconds * SR)]


def main(script_path, out_dir):
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    vo = json.loads((out / "vo.json").read_text())
    items, n = [], 0  # (start, dur, visual, vo_file, section_id)
    t = 0.0
    chapters = []
    for sec in script["sections"]:
        if sec.get("sting"):
            items.append((t, 2.2, Seq(sting_frames(int(2.2 * FPS))), None, sec["id"]))
            t += 2.2
            continue
        if sec.get("card"):
            chapters.append((t, sec["card"].title()))
            items.append((t, 2.4, Still(card(sec["card"])), None, sec["id"]))
            t += 2.4
        elif not chapters:
            chapters.append((0.0, "Intro"))
        for s in sec.get("shots") or []:
            n += 1
            d = vo[f"{n:02d}"] + BREATH
            if s.get("diagram"):
                vis = Shot(bgr(DIAGRAMS[s["diagram"]]()), "push")
            else:
                img = cv2.imread(str(out / f"shot-{n:02d}.png"))
                if s.get("overlay"):  # science drawn onto the painting (a crack, a hole) where SDXL can't
                    img = bgr(OVERLAYS[s["overlay"]](Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                vis = Shot(img, s.get("move", "push"))
            items.append((t, d, vis, out / f"vo-{n:02d}.wav", sec["id"]))
            t += d
    end_img = Image.new("RGB", (W, H), K.INK)
    lg = Image.open(ROOT / "brand/logo.png").convert("RGBA").resize((300, 300), Image.LANCZOS)
    end_img.paste(lg, ((W - 300) // 2, 300), lg)
    K.wordmark(ImageDraw.Draw(end_img), W / 2, 680, 72, anchor="mm")
    items.append((t, 6.0, Still(bgr(K.grain(end_img))), None, "outro"))
    total = t + 6.0

    # ---- audio: narration + music by section mood
    voice = np.zeros(int(total * SR) + SR, np.float32)
    for start, dur, _, f, _ in items:
        if f is None:
            continue
        import soundfile as sf
        x, sr = sf.read(f, dtype="float32")
        x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x).astype(np.float32)
        i = int((start + 0.15) * SR)
        voice[i:i + len(x)] += x[:len(voice) - i]
    mood_of = {sid: mood for mood, sids in (script.get("music") or {}).items() for sid in sids}
    tracks = {m: load_mp3(ROOT / "brand/music" / f"{m}.mp3", total + 1) for m in set(mood_of.values())}
    music = np.zeros_like(voice)
    env_len = int(1.5 * SR)
    for start, dur, _, _, sid in items:
        m = mood_of.get(sid)
        if not m:
            continue
        a, b = int(start * SR), int((start + dur) * SR)
        music[a:b] += tracks[m][a:b]
    music = np.convolve(music, np.ones(64) / 64, mode="same")  # soften hard track switches
    fade = np.minimum(1, np.arange(len(music)) / env_len) * np.minimum(1, (len(music) - np.arange(len(music))) / (3 * SR))
    mix = voice / (np.abs(voice).max() + 1e-9) * 0.92 + music / (np.abs(music).max() + 1e-9) * 0.13 * fade
    mix = np.clip(mix, -1, 1)
    import soundfile as sf
    sf.write(out / "mix.wav", mix[:int(total * SR)], SR)

    # ---- picture
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", str(out / "mix.wav"), "-map", "0:v", "-map", "1:a",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "film", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out / "video.mp4")],
                          stdin=subprocess.PIPE)
    k = 0
    for idx, (start, dur, vis, _, _) in enumerate(items):
        nf = int(round((start + dur) * FPS)) - int(round(start * FPS))
        nxt = items[idx + 1][2] if idx + 1 < len(items) else None
        for j in range(nf):
            u = j / max(1, nf - 1)
            fr = vis.frame(u)
            left = (nf - j) / FPS
            if nxt is not None and left < XF:  # dissolve into the next item
                fr = cv2.addWeighted(fr, left / XF, nxt.frame(0.0), 1 - left / XF, 0)
            ff.stdin.write(fr.tobytes())
            k += 1
    ff.stdin.close()
    ff.wait()
    (out / "chapters.txt").write_text("\n".join(f"{int(s // 60):02d}:{int(s % 60):02d} {name}" for s, name in chapters))
    print(f"video {total / 60:.1f} min, {k} frames -> {out / 'video.mp4'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
