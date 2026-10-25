"""The Hidden Why: brand assets from brand.json (logo, banner, thumbnails).

    python3 channels/the-hidden-why/brand_kit.py                 # logo.png, banner.png, banner-preview.png
    python3 channels/the-hidden-why/brand_kit.py thumb ART.png OUT.png "SQUARE|WINDOWS|*KILLED" [cx,cy,r]

Thumbnail text: lines split by "|", the one starting with "*" goes in the red box. The optional circle
(in 1280x720 pixels) rings the hidden detail in the art.
"""
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
B = json.loads((HERE / "brand.json").read_text())
C = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in B["colors"].items()}
INK, YEL, RED, BONE, TEAL = C["ink"], C["signal_yellow"], C["alarm_red"], C["bone"], C["deep_teal"]


def display(size):
    return ImageFont.truetype(str(HERE / B["fonts"]["display"]), size)


def body(size, weight="SemiBold"):
    f = ImageFont.truetype(str(HERE / B["fonts"]["body"]), size)
    try:
        f.set_variation_by_name(weight)
    except Exception:  # noqa: BLE001
        pass
    return f


def glow_bg(w, h, cx, cy, r, col=TEAL):
    """Ink with a soft teal light: the brand's backdrop."""
    img = Image.new("RGB", (w, h), INK)
    g = Image.new("L", (w, h), 0)
    ImageDraw.Draw(g).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    img.paste(Image.new("RGB", (w, h), col), mask=g.filter(ImageFilter.GaussianBlur(r * 0.6)))
    return img


def grain(img, amount=10, seed=3):
    rnd = random.Random(seed)
    n = Image.effect_noise(img.size, 40).point(lambda v: 128 + (v - 128) * amount // 40)
    del rnd
    return Image.blend(img, Image.merge("RGB", (n, n, n)), 0.06)


def mark(size, ring=True):
    """The logo mark: a yellow disc with a black keyhole, a red light looking back through it.
    Drawn 4x and scaled down for clean edges. `ring` puts it on an ink circle (the profile picture)."""
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = S * 0.07 if ring else 0
    if ring:
        d.ellipse((0, 0, S - 1, S - 1), fill=INK + (255,))
    d.ellipse((pad, pad, S - pad, S - pad), fill=YEL + (255,))
    cx, r = S / 2, S * 0.115
    top = S * 0.25
    d.ellipse((cx - r, top, cx + r, top + 2 * r), fill=INK + (255,))
    d.polygon([(cx - r * 0.55, top + r * 1.6), (cx + r * 0.55, top + r * 1.6), (cx + r * 1.05, S * 0.75),
               (cx - r * 1.05, S * 0.75)], fill=INK + (255,))
    eye = r * 0.36
    g = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse((cx - eye * 2, top + r - eye * 2, cx + eye * 2, top + r + eye * 2), fill=RED + (150,))
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(eye * 0.8)))
    d.ellipse((cx - eye, top + r - eye, cx + eye, top + r + eye), fill=RED + (255,))
    return im.resize((size, size), Image.LANCZOS)


def logo():
    img = Image.new("RGBA", (800, 800), INK + (255,))
    img.alpha_composite(mark(800))
    img.convert("RGB").save(HERE / "logo.png")


def wordmark(d, x, y, size, anchor="lm"):
    f = display(size)
    a = "THE HIDDEN "
    wa = d.textlength(a, font=f)
    wb = d.textlength("WHY", font=f)
    if anchor == "mm":
        x -= (wa + wb) / 2
    d.text((x, y), a, font=f, anchor="lm", fill=BONE)
    d.text((x + wa, y), "WHY", font=f, anchor="lm", fill=YEL)
    return wa + wb


def banner():
    W, H = 2560, 1440
    img = glow_bg(W, H, W // 2, H // 2, 700)
    d = ImageDraw.Draw(img)
    # faint blueprint grid: "how things really work"
    for x in range(0, W, 80):
        d.line((x, 0, x, H), fill=(22, 30, 36))
    for y in range(0, H, 80):
        d.line((0, y, W, y), fill=(22, 30, 36))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    ms = 300  # everything sits inside the 1546x423 box phones show
    m = mark(ms)
    f = display(150)
    wm = d.textlength("THE HIDDEN WHY", font=f)
    tf = body(32, "Medium")
    tag = B["tagline"].upper()
    while d.textlength(tag, font=tf) > wm and tf.size > 22:
        tf = body(tf.size - 1, "Medium")
    x0 = (W - (ms + 45 + wm)) / 2
    img.paste(m, (int(x0), H // 2 - ms // 2), m)
    d = ImageDraw.Draw(img)
    tx = x0 + ms + 45
    wordmark(d, tx, H / 2 - 30, 150)
    d.text((tx + 4, H / 2 + 75), tag, font=tf, anchor="lm", fill=BONE)
    d.rectangle((tx + 4, H / 2 + 105, tx + 204, H / 2 + 111), fill=RED)
    img = grain(img)
    img.save(HERE / "banner.png")
    pv = img.copy()
    d = ImageDraw.Draw(pv)
    d.rectangle(((W - 1546) // 2, (H - 423) // 2, (W + 1546) // 2, (H + 423) // 2), outline=(255, 255, 255), width=5)
    d.rectangle((0, (H - 423) // 2, W, (H + 423) // 2), outline=(255, 80, 80), width=5)
    pv.save(HERE / "banner-preview.png")


def thumb(art, out, text, circle=None):
    """The brand thumbnail: art right, huge yellow words left, one word in a red box, one red ring."""
    W, H = B["thumbnail"]["size"]
    img = Image.open(art).convert("RGB")
    s = max(W / img.width, H / img.height)
    img = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    img = img.crop(((img.width - W) // 2, (img.height - H) // 2, (img.width - W) // 2 + W, (img.height - H) // 2 + H))
    # left-side darkening so the words always read, plus a vignette
    shade = Image.new("L", (W, H))
    for x in range(W):
        v = int(235 * max(0.0, 1 - x / (W * 0.62)) ** 1.4)
        ImageDraw.Draw(shade).line((x, 0, x, H), fill=v)
    img.paste(Image.new("RGB", (W, H), INK), mask=shade)
    vig = Image.new("L", (W, H), 255)
    ImageDraw.Draw(vig).ellipse((-W * 0.15, -H * 0.25, W * 1.15, H * 1.25), fill=0)
    img.paste(Image.new("RGB", (W, H), INK), mask=vig.filter(ImageFilter.GaussianBlur(120)).point(lambda v: v * 0.8))
    d = ImageDraw.Draw(img)
    if circle:
        cx, cy, r = circle
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse((cx - r, cy - r, cx + r, cy + r), outline=RED + (200,), width=26)
        img.paste(g.filter(ImageFilter.GaussianBlur(14)), (0, 0), g.filter(ImageFilter.GaussianBlur(14)))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=RED, width=12)
    lines = [t.strip() for t in text.split("|")]
    size = 150 if len(lines) <= 3 else 120
    x, y = 56, (H - len(lines) * size * 1.02) / 2 + size * 0.5
    for t in lines:
        plain, _, boxed = t.upper().partition("*")  # "FOR *THIS": FOR in yellow, THIS in the red box
        f = display(size)
        while d.textlength(plain + boxed, font=f) + (28 if boxed else 0) > W * 0.52 and f.size > 70:
            f = display(f.size - 6)
        if plain:
            sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(sh).text((x + 6, y + 8), plain, font=f, anchor="lm", fill=(0, 0, 0, 200))
            img.paste(sh.filter(ImageFilter.GaussianBlur(6)), (0, 0), sh.filter(ImageFilter.GaussianBlur(6)))
            d = ImageDraw.Draw(img)
            d.text((x, y), plain, font=f, anchor="lm", fill=YEL, stroke_width=8, stroke_fill=INK)
        if boxed:
            bx = x + d.textlength(plain, font=f) + (14 if plain else 0)
            w = d.textlength(boxed, font=f)
            d.rectangle((bx - 14, y - f.size * 0.56, bx + w + 14, y + f.size * 0.5), fill=RED)
            d.text((bx, y), boxed, font=f, anchor="lm", fill=BONE)
        y += f.size * 1.04
    img = grain(img)
    img.save(out, quality=95)


if __name__ == "__main__":
    if sys.argv[1:2] == ["thumb"]:
        circ = tuple(int(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 else None
        thumb(sys.argv[2], sys.argv[3], sys.argv[4], circ)
    else:
        logo()
        banner()
        print("logo.png, banner.png, banner-preview.png")
