"""The episode thumbnail: the whole airplane on the right, a circular zoom bubble magnifying one detail
(the window), a red arrow pointing at the bubble, and 2-4 words left (brand rule: open the question,
never answer it; the sting word in the red box).

    python pipeline/thumb.py episodes/<slug>/script.json out/<slug>/   ->  thumb.jpg (1280x720)
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


def main(script_path, out_dir):
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
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
