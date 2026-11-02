"""Soch Ke Batao quiz Short (owner, 2 Nov 2026; frame design approved the same day):

    python pipeline/quiz.py episodes/<slug>/script.json out/<slug>/

Needs narrate.py's vo-01..05.wav + vo.json (the script's five lines: question with options, "comment your
answer", the answer, its explanation, subscribe). Timeline: the question frame while the question is read ->
the screen blurs with "सोच के बताओ! अपना जवाब कमेंट में लिखो" and a ticking 5-4-3-2-1 countdown -> the right
option lights up green with a chime, the explanation shows -> a subscribe strip. Writes video.mp4 (1080x1920),
thumb.jpg (the question, never the answer) and an empty chapters.txt.

script.json: {"format": "quiz", "brand": "soch-ke-batao", "n": <question number>, "subject": "<Hindi>",
 "question": "<on screen>", "options": ["<A>", "<B>", "<C>", "<D>"], "answer": <0-3>, "explanation": "<on screen>",
 "sections": [{"id": "main", "shots": [{"vo": ...} x5]}]}
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
HIND = str(ROOT / "brand" / "devlok-katha" / "fonts" / "Hind-Bold.ttf")
ANTON = str(ROOT / "brand" / "fonts" / "Anton-Regular.ttf")
W, H, SR, FPS = 1080, 1920, 24000, 30
INK, INK2 = (9, 28, 38), (16, 58, 74)
YEL, CORAL, MINT, BONE, GREEN = (255, 210, 63), (255, 90, 95), (46, 196, 182), (246, 241, 230), (52, 199, 89)
OPT = [CORAL, (77, 150, 255), MINT, (255, 159, 28)]
X0, X1 = 70, 880  # content edges: YouTube's buttons sit right of ~900, its title over the bottom ~420


def font(n, latin=False):
    return ImageFont.truetype(ANTON if latin else HIND, n)


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def background():
    im = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(im)
    for r in range(1400, 0, -10):
        t = r / 1400
        d.ellipse([W / 2 - r, 760 - r * 1.1, W / 2 + r, 760 + r * 1.1],
                  fill=tuple(int(INK[i] + (INK2[i] - INK[i]) * (1 - t) ** 1.5) for i in range(3)))
    faint = Image.new("RGBA", (W, H))
    fd = ImageDraw.Draw(faint)
    for x, y, s in [(120, 330, 150), (960, 420, 110), (90, 1560, 120), (990, 1300, 160), (520, 1750, 90)]:
        fd.text((x, y), "?", font=font(s, True), fill=(*YEL, 28), anchor="mm")
    im.paste(faint, (0, 0), faint)
    return im


def header(d, n, subject):
    d.rounded_rectangle([X0, 170, X0 + 86, 240], radius=22, fill=CORAL)
    d.text((X0 + 43, 203), "?", font=font(54, True), fill=YEL, anchor="mm", stroke_width=3, stroke_fill=INK)
    d.text((X0 + 108, 206), "सोच के बताओ", font=font(46), fill=BONE, anchor="lm")
    chip = f"सवाल #{n}  ·  {subject}"
    tw = d.textlength(chip, font=font(34))
    d.rounded_rectangle([X1 - tw - 50, 178, X1, 234], radius=28, fill=YEL)
    d.text((X1 - 25 - tw / 2, 207), chip, font=font(34), fill=INK, anchor="mm")


def question_block(d, q, y=330):
    size = 70
    while True:  # long questions shrink to fit 4 lines
        f = font(size)
        lines = wrap(d, q, f, X1 - X0 - 40)
        if len(lines) <= 4 or size <= 50:
            break
        size -= 4
    lh = int(size * 1.32)
    h = len(lines) * lh + 60
    d.rounded_rectangle([X0, y, X1, y + h], radius=36, fill=BONE)
    for i, ln in enumerate(lines):
        d.text(((X0 + X1) / 2, y + 30 + lh / 2 + i * lh), ln, font=f, fill=INK, anchor="mm")
    return y + h


def options(d, opts, y, right=None):
    for i, o in enumerate(opts):
        top = y + 50 + i * 150
        good = right == i
        size = 58
        while d.textlength(o, font=font(size)) > X1 - X0 - 220 and size > 36:
            size -= 2
        fill = GREEN if good else ((40, 60, 70) if right is not None else (22, 46, 58))
        d.rounded_rectangle([X0, top, X1, top + 122], radius=61, fill=fill,
                            outline=OPT[i] if right is None else (GREEN if good else (70, 90, 100)), width=6)
        d.ellipse([X0 + 14, top + 14, X0 + 108, top + 108], fill=OPT[i] if (right is None or good) else (80, 100, 110))
        d.text((X0 + 61, top + 61), "ABCD"[i], font=font(52, True), fill=INK, anchor="mm")
        d.text((X0 + 140, top + 63), o, font=font(size), fill=INK if good else (BONE if right is None else (150, 165, 170)),
               anchor="lm")
        if good:  # a drawn tick (no font needed)
            cx, cy = X1 - 60, top + 61
            d.line([(cx - 24, cy), (cx - 6, cy + 20), (cx + 26, cy - 22)], fill=INK, width=12, joint="curve")
    return y + 50 + len(opts) * 150


def timer(d, cx, cy, r, sec, frac):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(22, 46, 58))
    d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * frac, fill=YEL, width=14)
    d.text((cx, cy + 4), str(sec), font=font(int(r * 1.05), True), fill=YEL, anchor="mm")


def frame_question(s):
    im = background()
    d = ImageDraw.Draw(im)
    header(d, s["n"], s["subject"])
    options(d, s["options"], question_block(d, s["question"]))
    return im


def frame_blur(q_img, sec, frac):
    im = Image.blend(q_img.filter(ImageFilter.GaussianBlur(18)), Image.new("RGB", (W, H), INK), 0.35)
    d = ImageDraw.Draw(im)
    cx, cy, s = (X0 + X1) / 2, 820, 120
    d.polygon([(cx - s * 0.55, cy - 260 + s * 0.5), (cx - s * 0.8, cy - 260 + s * 1.2), (cx - s * 0.05, cy - 260 + s * 0.6)], fill=CORAL)
    d.rounded_rectangle([cx - s, cy - 260 - s * 0.82, cx + s, cy - 260 + s * 0.82], radius=s * 0.42, fill=CORAL)
    d.text((cx, cy - 266), "?", font=font(170, True), fill=YEL, anchor="mm", stroke_width=6, stroke_fill=INK)
    d.text((cx, cy), "सोच के बताओ!", font=font(110), fill=YEL, anchor="mm", stroke_width=6, stroke_fill=INK)
    d.text((cx, cy + 120), "अपना जवाब कमेंट में लिखो", font=font(58), fill=BONE, anchor="mm")
    d.text((cx, cy + 200), "A, B, C या D?", font=font(54), fill=BONE, anchor="mm")
    timer(d, cx, cy + 380, 95, sec, frac)
    return im


def frame_answer(s, subscribe=False):
    im = background()
    d = ImageDraw.Draw(im)
    header(d, s["n"], s["subject"])
    y = options(d, s["options"], question_block(d, s["question"]), right=s["answer"])
    f = font(46)
    lines = wrap(d, s["explanation"], f, X1 - X0 - 60)[:4]
    top = y + 40
    d.rounded_rectangle([X0, top, X1, top + 90 + len(lines) * 62], radius=30, fill=(22, 46, 58), outline=YEL, width=4)
    d.text((X0 + 30, top + 42), "जवाब: " + "ABCD"[s["answer"]], font=font(48), fill=YEL, anchor="lm")
    for i, ln in enumerate(lines):
        d.text((X0 + 30, top + 110 + i * 62), ln, font=f, fill=BONE, anchor="lm")
    if subscribe:
        b = top + 90 + len(lines) * 62 + 30
        d.rounded_rectangle([X0, b, X1, b + 100], radius=50, fill=CORAL)
        d.text(((X0 + X1) / 2, b + 52), "रोज़ नए सवाल · सब्सक्राइब करें", font=font(50), fill=BONE, anchor="mm")
    return im


def tone(freq, dur, vol, decay):
    t = np.arange(int(SR * dur)) / SR
    return (vol * np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)).astype(np.float32)


def main(script_path, out_dir):
    s = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    vo = [sf.read(out / f"vo-{i:02d}.wav", dtype="float32")[0] for i in range(1, 6)]
    dur = [len(v) / SR for v in vo]
    # the timeline (seconds)
    t_q = 0.25 + dur[0] + 0.35                      # question frame
    t_b = max(5.0, dur[1] + 0.6)                    # blur + countdown (at least the 5 ticks)
    t_a = 0.3 + dur[2] + 0.25 + dur[3] + 0.5        # answer + explanation
    t_s = dur[4] + 0.9                              # subscribe
    total = t_q + t_b + t_a + t_s
    voice = np.zeros(int(SR * (total + 0.2)), np.float32)

    def put(x, at):
        a = int(SR * at)
        voice[a:a + len(x)] += x[:len(voice) - a]
    put(vo[0], 0.25)
    put(vo[1], t_q + 0.1)
    put(vo[2], t_q + t_b + 0.3)
    put(vo[3], t_q + t_b + 0.3 + dur[2] + 0.25)
    put(vo[4], t_q + t_b + t_a)
    fx = np.zeros_like(voice)
    for k in range(5):  # tick-tock under the countdown
        a = int(SR * (t_q + k * t_b / 5))
        tk = tone(1800 if k % 2 == 0 else 1400, 0.06, 0.25, 60)
        fx[a:a + len(tk)] += tk
    ding = tone(1320, 0.9, 0.3, 5) + tone(1980, 0.9, 0.15, 6)  # the reveal chime
    a = int(SR * (t_q + t_b))
    fx[a:a + len(ding)] += ding[:len(fx) - a]
    music = np.zeros_like(voice)
    try:  # a low bed under everything (the repo's licensed library)
        sys.path.insert(0, str(ROOT / "brand"))
        import brandcfg as K
        m = K.asset("music/intense.mp3")
        tmp = Path(tempfile.mkdtemp()) / "m.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(m), "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
        bed = sf.read(tmp, dtype="float32")[0]
        bed = np.tile(bed, int(np.ceil(len(music) / len(bed))))[:len(music)]
        music = bed * 0.08
    except Exception as e:  # noqa: BLE001 - no music is better than no video
        print(f"  music skipped ({e})")
    mix = voice + fx + music
    mix = mix / max(1.0, float(np.abs(mix).max()) / 0.95)
    work = Path(tempfile.mkdtemp())
    sf.write(work / "a.wav", mix, SR)
    # frames: the question, 5 countdown frames, the answer, the answer + subscribe
    q = frame_question(s)
    segs = [(q, t_q)] + [(frame_blur(q, 5 - k, (5 - k) / 5), t_b / 5) for k in range(5)] + \
           [(frame_answer(s), t_a), (frame_answer(s, subscribe=True), t_s)]
    lst = []
    for i, (im, d) in enumerate(segs):
        im.save(work / f"f{i}.png")
        lst += [f"file '{work}/f{i}.png'", f"duration {d:.3f}"]
    lst.append(f"file '{work}/f{len(segs) - 1}.png'")
    (work / "list.txt").write_text("\n".join(lst) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
                    "-i", str(work / "a.wav"), "-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                    str(out / "video.mp4")], check=True)
    q.convert("RGB").save(out / "thumb.jpg", quality=92)
    (out / "chapters.txt").write_text("")
    print(f"quiz: {total:.1f}s -> {out / 'video.mp4'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
