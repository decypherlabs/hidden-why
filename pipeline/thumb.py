"""The episode thumbnail: the whole airplane on the right, a circular zoom bubble magnifying one detail
(the window), a red arrow pointing at the bubble, and 2-4 words left (brand rule: open the question,
never answer it; the sting word in the red box).

    python pipeline/thumb.py episodes/<slug>/script.json out/<slug>/   ->  thumb.jpg (1280x720)

thumbnail.layout "face" (after The Infographics Show, the niche's top long-form page): the painting is one
person with an extreme emotion filling the frame, a 2-4 word statement in heavy white caps across the top
(the *word* in yellow), and the zoom bubble, if any, small at the bottom right.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brand"))
import brand_kit as K  # noqa: E402  (display font, grain, colours)
sys.path.insert(0, str(ROOT / "pipeline"))

W, H = 1280, 720


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img.crop(((img.width - w) // 2, (img.height - h) // 2, (img.width - w) // 2 + w, (img.height - h) // 2 + h))


def arrow(d, start, end, width=22, head=56):
    import math
    (x0, y0), (x1, y1) = start, end
    a = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * math.cos(a), y1 - head * math.sin(a)
    d.line((x0, y0, bx, by), fill=K.RED, width=width)
    d.polygon([(x1, y1), (bx + head * 0.6 * math.sin(a), by - head * 0.6 * math.cos(a)),
               (bx - head * 0.6 * math.sin(a), by + head * 0.6 * math.cos(a))], fill=K.RED)


def face_layout(script, out):
    th = script["thumbnail"]
    img = cover(Image.open(out / "thumb-plane.png").convert("RGB"), W, H)
    shade = Image.new("L", (W, H))
    sd = ImageDraw.Draw(shade)
    for y in range(H):  # darken the top for the words
        sd.line((0, y, W, y), fill=int(200 * max(0.0, 1 - y / (H * 0.42)) ** 1.6))
    img.paste(Image.new("RGB", (W, H), K.INK), mask=shade)
    d = ImageDraw.Draw(img)
    words = th["text"].replace("|", " ").split()
    f = K.display(170)
    sp = lambda f: d.textlength(" ", font=f)
    total = lambda f: sum(d.textlength(w.replace("*", "").upper(), font=f) for w in words) + sp(f) * (len(words) - 1)
    while total(f) > W - 80 and f.size > 70:
        f = K.display(f.size - 6)
    x, y = (W - total(f)) / 2, 30 + f.size * 0.55
    for w in words:
        t = w.replace("*", "").upper()
        d.text((x, y), t, font=f, anchor="lm", fill=K.YEL if "*" in w else K.BONE, stroke_width=10, stroke_fill=K.INK)
        x += d.textlength(t, font=f) + sp(f)
    if th.get("zoom") or th.get("zoom_diagram"):
        r, cx, cy = 120, W - 160, H - 150
        if th.get("zoom_diagram"):
            from diagram import DIAGRAMS
            dg = DIAGRAMS[th["zoom_diagram"]]()
            zx, zy, zr = th.get("zoom_crop", (960, 540, 330))
            zoom = cover(dg.crop((zx - zr, zy - zr, zx + zr, zy + zr)), 2 * r, 2 * r)
        else:
            zoom = cover(Image.open(out / "thumb-zoom.png").convert("RGB"), 2 * r, 2 * r)
        m = Image.new("L", (2 * r, 2 * r), 0)
        ImageDraw.Draw(m).ellipse((0, 0, 2 * r, 2 * r), fill=255)
        img.paste(zoom, (cx - r, cy - r), m)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=K.BONE, width=8)
        d.ellipse((cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8), outline=K.RED, width=8)
    K.grain(img).save(out / "thumb.jpg", quality=95)
    print(out / "thumb.jpg")


def short_cover(script, out):
    """Shorts cover 1080x1920 (owner, 29 Oct: "shocking and surprising, put a question in the viewer's
    brain"). After Zack D. Films' winners: one strange, bright picture fills the frame; ours adds the
    question in heavy caps in the top safe zone (the Shorts UI covers the bottom and right), the *key*
    word in a yellow box, and our tag. script["cover_shot"] picks the painting (default: the first)."""
    from PIL import ImageEnhance
    import re
    CW, CH = 1080, 1920
    n = int(script.get("cover_shot") or 1)
    img = cover(Image.open(out / f"shot-{n:02d}.png").convert("RGB"), CW, CH)
    img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.25)).enhance(1.12)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    shade = Image.new("L", (CW, CH))
    sd = ImageDraw.Draw(shade)
    for y in range(CH):  # dark at the top for the words, a light vignette at the bottom
        top = max(0.0, 1 - y / (CH * 0.40)) ** 1.5
        bottom = max(0.0, (y - CH * 0.82) / (CH * 0.18)) ** 1.5
        sd.line((0, y, CW, y), fill=int(215 * top + 140 * bottom))
    img.paste(Image.new("RGB", (CW, CH), K.INK), mask=shade)
    d = ImageDraw.Draw(img)
    text = re.sub(r"[^\w\s?!'’*.,:…\u0900-\u097F-]", "", script.get("cover_text") or script.get("hook_text") or script["title"]).strip().upper()
    if "*" not in text:  # no marked key word: the last word of the question
        ws = text.split()
        ws[-1] = "*" + ws[-1]
        text = " ".join(ws)
    words, lines, f = text.split(), [], K.display(150)
    width = lambda s: d.textlength(s.replace("*", ""), font=f)
    while True:
        lines = []
        for w in words:
            if lines and width(lines[-1] + " " + w) < CW * 0.86:
                lines[-1] += " " + w
            else:
                lines.append(w)
        if len(lines) <= 3 or f.size <= 96:
            break
        f = K.display(f.size - 8)
    y = 230
    for ln in lines:
        x = (CW - width(ln)) / 2
        for w in ln.split():
            t = w.replace("*", "")
            tw = d.textlength(t, font=f)
            if w.startswith("*"):
                bb = d.textbbox((x, y), t, font=f, anchor="lm")  # the real ink: Devanagari vowel signs rise above the caps height
                d.rectangle((x - 14, min(y - f.size * 0.56, bb[1] - 12), x + tw + 14, max(y + f.size * 0.52, bb[3] + 12)), fill=K.YEL)
                d.text((x, y), t, font=f, anchor="lm", fill=K.INK)
            else:
                d.text((x, y), t, font=f, anchor="lm", fill=K.BONE, stroke_width=10, stroke_fill=K.INK)
            x += tw + d.textlength(" ", font=f)
        y += f.size * 1.08
    K.wordmark(d, CW / 2, CH - 330, 44, anchor="mm")
    K.grain(img).save(out / "thumb.jpg", quality=95)
    print(out / "thumb.jpg")


def main(script_path, out_dir):
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    if script.get("format") == "short":
        return short_cover(script, out)
    if (script.get("thumbnail") or {}).get("layout") == "face":
        return face_layout(script, out)
    plane = cover(Image.open(out / "thumb-plane.png").convert("RGB"), W, H)
    img = plane.copy()
    # darken the left for the words
    shade = Image.new("L", (W, H))
    sd = ImageDraw.Draw(shade)
    for x in range(W):
        sd.line((x, 0, x, H), fill=int(225 * max(0.0, 1 - x / (W * 0.55)) ** 1.3))
    img.paste(Image.new("RGB", (W, H), K.INK), mask=shade)
    # zoom bubble: the window close-up in a circle, white rim + red ring, soft shadow
    r, cx, cy = 165, 1030, 215
    th = script.get("thumbnail") or {}
    if th.get("zoom_diagram"):  # numbers/labels can't be painted: the bubble shows a code-drawn close-up
        from diagram import DIAGRAMS
        dg = DIAGRAMS[th["zoom_diagram"]]()
        zx, zy, zr = th.get("zoom_crop", (960, 540, 330))
        zoom = cover(dg.crop((zx - zr, zy - zr, zx + zr, zy + zr)), 2 * r, 2 * r)
    else:
        zoom = cover(Image.open(out / "thumb-zoom.png").convert("RGB"), 2 * r, 2 * r)
    m = Image.new("L", (2 * r, 2 * r), 0)
    ImageDraw.Draw(m).ellipse((0, 0, 2 * r, 2 * r), fill=255)
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).ellipse((cx - r + 10, cy - r + 16, cx + r + 10, cy + r + 16), fill=190)
    img.paste((0, 0, 0), mask=sh.filter(ImageFilter.GaussianBlur(18)))
    img.paste(zoom, (cx - r, cy - r), m)
    d = ImageDraw.Draw(img)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=K.BONE, width=10)
    d.ellipse((cx - r - 9, cy - r - 9, cx + r + 9, cy + r + 9), outline=K.RED, width=9)
    # arrow from the words toward the bubble
    arrow(d, (560, 300), (cx - r - 22, cy + 30))
    # words (brand style: yellow Anton with ink stroke, sting word in a red box)
    lines = [t.strip() for t in script["thumbnail"]["text"].split("|")]
    size = 150 if len(lines) <= 2 else 120
    x, y = 56, (H - len(lines) * size * 1.02) / 2 + size * 0.5 + 40
    for t in lines:
        plain, _, boxed = t.upper().partition("*")
        boxed = boxed.replace("*", "")  # "*WORD*" and "*WORD" both mean: WORD in the red box
        f = K.display(size)
        while d.textlength(plain + boxed, font=f) + (28 if boxed else 0) > W * 0.46 and f.size > 70:
            f = K.display(f.size - 6)
        if plain:
            d.text((x, y), plain, font=f, anchor="lm", fill=K.YEL, stroke_width=8, stroke_fill=K.INK)
        if boxed:
            bx = x + d.textlength(plain, font=f) + (14 if plain else 0)
            w = d.textlength(boxed, font=f)
            d.rectangle((bx - 14, y - f.size * 0.56, bx + w + 14, y + f.size * 0.5), fill=K.RED)
            d.text((bx, y), boxed, font=f, anchor="lm", fill=K.BONE)
        y += f.size * 1.04
    K.grain(img).save(out / "thumb.jpg", quality=95)
    print(out / "thumb.jpg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
