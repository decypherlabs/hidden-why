"""The edit: paintings + diagrams + narration + music -> the 1920x1080 episode, or with "format": "short" the
1080x1920 Short (fast cuts, bold 2-3 word captions, no cards, no end screen: Shorts loop).

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


class Captions:
    """Shorts: 2-3 word groups in heavy white caps with a black outline, mid-lower, timed across each
    shot's narration by word length (the voice reads at a steady pace)."""

    def __init__(self, groups):
        self.groups, self.cache = groups, {}  # [(t0, t1, text)]

    @staticmethod
    def for_shot(text, start, dur, size=3):
        words = text.split()
        chunks = [words[i:i + size] for i in range(0, len(words), size)]
        total = sum(len(w) + 1 for w in words) or 1
        out, t = [], start
        for c in chunks:
            d = dur * sum(len(w) + 1 for w in c) / total
            out.append((t, t + d, " ".join(c)))
            t += d
        return out

    def draw(self, frame, t):
        g = next((g for g in self.groups if g[0] <= t < g[1]), None)
        if not g:
            return frame
        if g[2] not in self.cache:
            f = K.display(118)
            img = Image.new("RGBA", (W, 400), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            text = g[2].upper().replace(",", "").replace(".", "")
            while d.textlength(text, font=f) > W * 0.86 and f.size > 60:
                f = K.display(f.size - 6)
            d.text((W / 2, 200), text, font=f, anchor="mm", fill=(255, 255, 255, 255), stroke_width=10,
                   stroke_fill=(0, 0, 0, 255))
            a = np.asarray(img)
            self.cache[g[2]] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        rgb, al = self.cache[g[2]]
        y = int(H * 0.68) - 200
        roi = frame[y:y + 400].astype(np.float32)
        frame[y:y + 400] = (rgb * al + roi * (1 - al)).astype(np.uint8)
        return frame


def load_mp3(path, seconds):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    reps = int(np.ceil(seconds * SR / max(1, len(x))))
    return np.tile(x, reps)[:int(seconds * SR)]


def main(script_path, out_dir):
    global W, H, XF
    script = json.loads(Path(script_path).read_text())
    short = script.get("format") == "short"
    if short:
        W, H, XF = 1080, 1920, 0.15
    caps = []
    out = Path(out_dir)
    vo = json.loads((out / "vo.json").read_text())
    items, n = [], 0  # (start, dur, visual, vo_file, section_id)
    explainers = []  # time ranges of the science diagrams: the music sits a little higher under them
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
                explainers.append((t, t + d))
            else:
                img = cv2.imread(str(out / f"shot-{n:02d}.png"))
                if s.get("overlay"):  # science drawn onto the painting (a crack, a hole) where SDXL can't
                    img = bgr(OVERLAYS[s["overlay"]](Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                vis = Shot(img, s.get("move", "push"))
            items.append((t, d, vis, out / f"vo-{n:02d}.wav", sec["id"]))
            if short:
                caps += Captions.for_shot(s["vo"], t + 0.15, vo[f"{n:02d}"])
            t += d
    if short:  # no end screen: Shorts loop back to the start
        total = t
    else:
        end_img = Image.new("RGB", (W, H), K.INK)
        lg = Image.open(ROOT / "brand/logo.png").convert("RGBA").resize((300, 300), Image.LANCZOS)
        end_img.paste(lg, ((W - 300) // 2, 300), lg)
        K.wordmark(ImageDraw.Draw(end_img), W / 2, 680, 72, anchor="mm")
        items.append((t, 6.0, Still(bgr(K.grain(end_img))), None, "outro"))
        total = t + 6.0
    captions = Captions(caps) if short else None

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
    # each section's track, loudness-matched (RMS, not peak: a soft bed with a few hits would vanish),
    # with 1.2 s crossfades where the mood changes
    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9
    tracks = {m: t / rms(t) for m, t in tracks.items()}
    music = np.zeros_like(voice)
    xf = int(1.2 * SR)
    for start, dur, _, _, sid in items:
        m = mood_of.get(sid)
        if not m:
            continue
        a, b = max(0, int(start * SR) - xf // 2), min(len(music), int((start + dur) * SR) + xf // 2)
        seg = tracks[m][a:b].copy()
        ramp = np.minimum(1, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / (xf / 2))
        music[a:b] += seg * ramp
    # ducking: music ~12 dB under the narration while it speaks, up to ~4 dB under in pauses, cards,
    # the sting and the end screen; smooth attack (0.15 s) and slower release (0.6 s)
    hop = SR // 50
    env = np.sqrt(np.convolve(voice.astype(np.float64) ** 2, np.ones(hop) / hop, mode="same"))
    speaking = (env > 0.02 * (env.max() + 1e-9)).astype(np.float64)
    gain = np.empty_like(speaking)
    g, up, down = 0.0, 1 / (0.6 * SR), 1 / (0.15 * SR)
    for i in range(0, len(speaking), hop):  # per 20 ms block, then held across the block
        target = speaking[i]
        g = min(target, g + down * hop) if target > g else max(target, g - up * hop)
        gain[i:i + hop] = g
    voice_rms = rms(voice[speaking > 0]) if speaking.any() else 0.1
    under = np.full(len(gain), 10 ** (-12 / 20))
    for a, b in explainers:  # under the explainers: -8 dB (a clear, curious bed while the viewer reads)
        under[int(a * SR):int(b * SR)] = 10 ** (-8 / 20)
    under = np.convolve(under, np.ones(SR // 2) / (SR // 2), mode="same")  # ease in/out over 0.5 s
    level = voice_rms * (under * gain + 10 ** (-4 / 20) * (1 - gain))
    fade = np.minimum(1, np.arange(len(music)) / (1.5 * SR)) * np.minimum(1, (len(music) - np.arange(len(music))) / (3 * SR))
    mix = voice + music * level * fade
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.95
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
            if captions:
                fr = captions.draw(fr.copy(), k / FPS)
            ff.stdin.write(fr.tobytes())
            k += 1
    ff.stdin.close()
    ff.wait()
    (out / "chapters.txt").write_text("" if short else"\n".join(f"{int(s // 60):02d}:{int(s % 60):02d} {name}" for s, name in chapters))
    print(f"video {total / 60:.1f} min, {k} frames -> {out / 'video.mp4'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
