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
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

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


class Intro:
    """The show's title sequence after the cold open (Logos & Legends, owner 7 Nov 2026: "LL misses the
    introduction"): the brand crest scales in out of darkness, the episode title rises in under it and a gold rule
    sweeps out; a slow push throughout. brand "intro": "title"."""

    def __init__(self, title):
        self.title = title.upper()
        lg = Image.open(K.asset("logo.png")).convert("RGBA")
        m = Image.new("L", lg.size, 0)
        ImageDraw.Draw(m).ellipse((0, 0) + lg.size, fill=255)  # the round crest, without its square corners
        lg.putalpha(Image.composite(lg.getchannel("A"), Image.new("L", lg.size, 0), m))
        self.logo = lg
        self.bg = Image.new("RGB", (W, H), (10, 10, 14))
        g = Image.new("L", (W, H), 0)
        ImageDraw.Draw(g).ellipse((W * 0.2, H * 0.05, W * 0.8, H * 0.95), fill=70)
        from PIL import ImageFilter
        self.bg = Image.composite(Image.new("RGB", (W, H), (38, 32, 24)), self.bg, g.filter(ImageFilter.GaussianBlur(160)))

    def frame(self, u):
        ease = lambda x: 1 - (1 - min(1.0, max(0.0, x))) ** 3
        img = self.bg.copy()
        a = ease(u / 0.35)
        size = int(440 * (0.82 + 0.18 * a) * (1 + 0.04 * u))
        lg = self.logo.resize((size, size), Image.LANCZOS)
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * a)))
        img.paste(lg, (W // 2 - size // 2, int(H * 0.38) - size // 2), lg)
        d = ImageDraw.Draw(img)
        b = ease((u - 0.25) / 0.35)
        if b > 0:
            f = K.display(104)
            while d.textlength(self.title, font=f) > W * 0.85 and f.size > 40:
                f = K.display(f.size - 4)
            y = int(H * 0.74 + 30 * (1 - b))
            col = tuple(int(c * b + 10 * (1 - b)) for c in (248, 242, 230))
            d.text((W / 2, y), self.title, font=f, anchor="mm", fill=col)
            half = int(260 * ease((u - 0.4) / 0.3))
            if half > 0:
                d.rectangle((W // 2 - half, y + 78, W // 2 + half, y + 82), fill=K.YEL)
        fade = min(1.0, (1 - u) / 0.12)  # out to the first chapter
        out = bgr(img)
        return (out * fade).astype(np.uint8) if fade < 1 else out


# ---- documentary graphics (Logos & Legends, owner 7 Nov 2026, after The Quiet Story's on-screen text)

def numbered_card(title, num):
    """Chapter card like the reference: deep navy, the chapter number huge and faded, "CHAPTER 02" in small gold
    spaced capitals, the title in cream."""
    img = Image.new("RGB", (W, H), (12, 22, 36))
    d = ImageDraw.Draw(img)
    big = K.display(560)
    d.text((W / 2, H / 2 + 40), f"{num:02d}", font=big, anchor="mm", fill=(26, 40, 60))
    d.text((W / 2, H / 2 - 70), " ".join(f"CHAPTER {num:02d}"), font=K.body(28, "Bold"), fill=K.YEL, anchor="mm")
    f = K.display(96)
    while d.textlength(title, font=f) > W * 0.8 and f.size > 50:
        f = K.display(f.size - 4)
    d.text((W / 2, H / 2 + 30), title, font=f, fill=(244, 236, 220), anchor="mm")
    return bgr(K.grain(img))


def _paste(frame, rgba, x, y, alpha=1.0):
    arr = np.asarray(rgba)
    rgb, al = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), arr[..., 3:4] / 255.0 * max(0.0, alpha)
    h, w = rgb.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    sr, sa = rgb[y0 - y:y1 - y, x0 - x:x1 - x], al[y0 - y:y1 - y, x0 - x:x1 - x]
    frame[y0:y1, x0:x1] = (sr * sa + roi * (1 - sa)).astype(np.uint8)
    return frame


def _dim(frame, k):
    return (frame.astype(np.float32) * (1 - k)).astype(np.uint8)


def date_callout(frame, text, t, dur=2.0):
    """A big bold date over a softened, dimmed frame ("OCTOBER 1947"), underlined; a typewriter hit lands on it."""
    a = min(1.0, t / 0.2, (dur - t) / 0.3)
    frame = cv2.addWeighted(frame, 1 - 0.6 * a, cv2.GaussianBlur(frame, (0, 0), 9), 0.6 * a, 0)
    frame = _dim(frame, 0.25 * a)
    f = K.body(130, "Bold")
    img = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((W / 2, 110), text.upper(), font=f, anchor="mm", fill=(255, 255, 255, 255))
    tw = d.textlength(text.upper(), font=f)
    d.rectangle((W / 2 - tw / 2, 190, W / 2 - tw / 2 + tw * min(1.0, t / 0.6), 196), fill=(255, 255, 255, 255))
    return _paste(frame, img, 0, H / 2 - 130, a)


def stamp(frame, text, t, dur=2.4):
    """A red rubber stamp slammed onto the scene ("NO MONEY"): double border, grainy ink, tilted, a quick slam-in."""
    if t > dur:
        return frame
    key = ("stamp", text)
    if key not in _TITLES:
        f = K.body(110, "Bold")
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = int(d0.textlength(text.upper(), font=f))
        im = Image.new("RGBA", (tw + 120, 220), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        red = (200, 30, 36, 235)
        d.rounded_rectangle((8, 8, tw + 112, 212), 22, outline=red, width=12)
        d.rounded_rectangle((26, 26, tw + 94, 194), 16, outline=red, width=5)
        d.text(((tw + 120) / 2, 112), text.upper(), font=f, anchor="mm", fill=red)
        rng = np.random.default_rng(7)  # worn ink
        a = np.asarray(im).copy()
        holes = rng.random(a.shape[:2]) < 0.09
        a[..., 3][holes] = (a[..., 3][holes] * 0.25).astype(np.uint8)
        im = Image.fromarray(a).rotate(-11, expand=True, resample=Image.BICUBIC)
        _TITLES[key] = im
    im = _TITLES[key]
    sc = 1.35 - 0.35 * min(1.0, t / 0.12)
    w, h = int(im.width * sc), int(im.height * sc)
    a = min(1.0, (dur - t) / 0.35)
    return _paste(frame, im.resize((w, h)), W * 0.62 - w / 2, H * 0.4 - h / 2, a)


def keyword(frame, text, t, dur=2.6):
    """A spaced-capitals word over the dimmed scene ("LICENCE RAJ")."""
    a = min(1.0, t / 0.35, (dur - t) / 0.4)
    frame = _dim(frame, 0.45 * a)
    img = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((W / 2, 100), "  ".join(text.upper()), font=K.body(66), anchor="mm", fill=(250, 246, 238, 255))
    return _paste(frame, img, 0, H / 2 - 100, a)


def typecard(frame, text, t, dur=3.4):
    """Old paper with the words typed onto it, letter by letter (company names, documents)."""
    a = min(1.0, t / 0.3, (dur - t) / 0.35)
    frame = _dim(frame, 0.35 * a)
    f = ImageFont.truetype(str(K.HERE / "logos-and-legends" / "fonts" / "SpecialElite-Regular.ttf"), 76) \
        if (K.HERE / "logos-and-legends" / "fonts" / "SpecialElite-Regular.ttf").exists() else K.body(70)
    shown = text[:int(max(0.0, t - 0.25) / 0.07)]
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    cw = int(max(700, d0.textlength(text, font=f) + 160))
    card = Image.new("RGBA", (cw, 260), (236, 224, 196, 250))
    rng = np.random.default_rng(3)
    arr = np.asarray(card).copy()
    arr[..., :3] = np.clip(arr[..., :3].astype(np.int16) - rng.integers(0, 18, arr.shape[:2])[..., None], 0, 255).astype(np.uint8)
    card = Image.fromarray(arr)
    ImageDraw.Draw(card).text((cw / 2, 130), shown, font=f, anchor="mm", fill=(40, 34, 28, 255))
    return _paste(frame, card, W / 2 - cw / 2, H / 2 - 130, a)


_LOGOS = {}


def logo_image(name, out):
    """The brand's real logo from its Wikipedia page (editorial use; owner, 7 Nov 2026), cached in out/logos/."""
    if name in _LOGOS:
        return _LOGOS[name]
    import urllib.parse
    import urllib.request
    p = out / "logos" / (re.sub(r"[^a-z0-9]+", "-", name.lower()) + ".png")
    if not p.exists():
        try:
            q = urllib.parse.urlencode({"action": "query", "titles": name, "prop": "pageimages", "format": "json",
                                        "pithumbsize": 600, "redirects": 1})
            req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{q}", headers={"User-Agent": "LogosAndLegends/1.0"})
            pages = json.loads(urllib.request.urlopen(req, timeout=30).read())["query"]["pages"]
            src = next(iter(pages.values())).get("thumbnail", {}).get("source")
            if src:
                p.parent.mkdir(exist_ok=True)
                req = urllib.request.Request(src, headers={"User-Agent": "LogosAndLegends/1.0"})
                p.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:  # noqa: BLE001 - the brand's name in type instead
            print(f"logo {name}: {e}")
    _LOGOS[name] = Image.open(p).convert("RGBA") if p.exists() else None
    return _LOGOS[name]


def _logo_tile(name, out, size=230):
    """A white rounded card with the logo (or the name in type when there's none)."""
    card = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, size - 1, size - 1), 28, fill=(250, 250, 248, 245))
    lg = logo_image(name, out)
    if lg is not None:
        lg.thumbnail((int(size * 0.8), int(size * 0.8)))
        card.paste(lg, ((size - lg.width) // 2, (size - lg.height) // 2), lg)
    else:
        f = K.body(34, "Bold")
        d.multiline_text((size / 2, size / 2), name.replace(" ", "\n"), font=f, anchor="mm", fill=(30, 30, 30, 255), align="center")
    return card


def logo_card(frame, name, out, t, dur=2.8):
    a = min(1.0, t / 0.25, (dur - t) / 0.35)
    sc = 0.85 + 0.15 * min(1.0, t / 0.25)
    tile = _logo_tile(name, out)
    s2 = int(tile.width * sc)
    return _paste(frame, tile.resize((s2, s2)), W - s2 - 80, 80, a)


def merge_graphic(frame, spec, out, t, dur=4.2):
    """A business move in one line: logos with "+" and "->", the year over the arrow (Mahindra + Punjab Tractors
    -> 2007 -> Swaraj). spec: {"from": [...], "year": "2007", "to": "Swaraj"}; each piece appears in turn."""
    a = min(1.0, (dur - t) / 0.4)
    frame = _dim(frame, 0.5 * min(1.0, t / 0.3) * a)
    names = list(spec.get("from") or []) + ([spec["to"]] if spec.get("to") else [])
    n = len(names)
    tile_w, gap = 230, 110
    total = n * tile_w + (n - 1) * gap
    x = W / 2 - total / 2
    f = K.body(70, "Bold")
    for i, nm in enumerate(names):
        ti = 0.35 * i
        if t < ti:
            break
        al = min(1.0, (t - ti) / 0.25) * a
        frame = _paste(frame, _logo_tile(nm, out), x, H / 2 - tile_w / 2, al)
        if i < n - 1 and t > ti + 0.2:
            sym = "→" if (spec.get("to") and i == n - 2) else "+"
            img = Image.new("RGBA", (gap, 200), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((gap / 2, 100), sym, font=f, anchor="mm", fill=(255, 255, 255, 255))
            frame = _paste(frame, img, x + tile_w, H / 2 - 100, al)
            if sym == "→" and spec.get("year"):
                yi = Image.new("RGBA", (gap + 120, 60), (0, 0, 0, 0))
                ImageDraw.Draw(yi).text(((gap + 120) / 2, 30), str(spec["year"]), font=K.body(40, "Bold"), anchor="mm", fill=K.YEL + (255,))
                frame = _paste(frame, yi, x + tile_w - 60, H / 2 - 150, al)
        x += tile_w + gap
    return frame


def film_card(title, num, first_shot):
    """A documentary chapter card (Logos & Legends, after The Quiet Story; brand "chapter_style": "film"): the
    chapter's first picture darkened and softened, "CHAPTER N" in small spaced gold capitals, the title large,
    a thin gold rule."""
    from PIL import ImageFilter
    bg = Image.open(first_shot).convert("RGB").resize((W, H)) if first_shot and Path(first_shot).exists() \
        else Image.new("RGB", (W, H), K.INK)
    bg = Image.blend(bg.filter(ImageFilter.GaussianBlur(6)), Image.new("RGB", (W, H), (8, 8, 10)), 0.62)
    d = ImageDraw.Draw(bg)
    d.text((W / 2, H / 2 - 95), " ".join(f"CHAPTER {num}"), font=K.body(30, "Bold"), fill=K.YEL, anchor="mm")
    f = K.display(120)
    while d.textlength(title, font=f) > W * 0.8 and f.size > 60:
        f = K.display(f.size - 6)
    d.text((W / 2, H / 2 + 10), title, font=f, fill=(248, 242, 230), anchor="mm")
    tw = d.textlength(title, font=f)
    d.rectangle((W / 2 - min(tw, 600) * 0.3, H / 2 + 95, W / 2 + min(tw, 600) * 0.3, H / 2 + 98), fill=K.YEL)
    return bgr(K.grain(bg))


_TITLES = {}


def title_overlay(frame, text, alpha):
    """A big phrase over the picture (shot "title"; after The Quiet Story's "OR OUR OWN?"): centred, cream with a
    soft shadow, the *starred* word in the brand's gold, fading in and out."""
    from PIL import ImageFilter
    if text not in _TITLES:
        f = K.display(118)
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        plain = text.replace("*", "")
        while d0.textlength(plain, font=f) > W * 0.8 and f.size > 48:
            f = K.display(f.size - 4)
        img = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
        sh = Image.new("L", (W, 260), 0)
        ImageDraw.Draw(sh).text((W / 2, 133), plain, font=f, anchor="mm", fill=230, stroke_width=4, stroke_fill=230)
        img.paste(Image.new("RGBA", (W, 260), (0, 0, 0, 255)), mask=sh.filter(ImageFilter.GaussianBlur(9)))
        d = ImageDraw.Draw(img)
        x = W / 2 - d.textlength(plain, font=f) / 2
        for part in re.split(r"(\*[^*]+\*)", text):
            if not part:
                continue
            t = part.strip("*")
            d.text((x, 130), t, font=f, anchor="lm", fill=(K.YEL if part.startswith("*") else (250, 245, 235)) + (255,))
            x += d.textlength(t, font=f)
        arr = np.asarray(img)
        _TITLES[text] = (cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), arr[..., 3:4] / 255.0)
    rgb, al = _TITLES[text]
    al = al * max(0.0, alpha)
    y = int(H * 0.42) - 130
    roi = frame[y:y + 260].astype(np.float32)
    frame[y:y + 260] = (rgb * al + roi * (1 - al)).astype(np.uint8)
    return frame


def sting_frames(n):
    logo = Image.open(K.asset("logo.png")).convert("RGBA")
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

    def __init__(self, img, move, focus=None):
        h, w = img.shape[:2]
        f = max(SS * W / w, SS * H / h) * 1.12
        self.src = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_LANCZOS4)
        self.move = move
        self.focus = focus  # (x, y) 0-1: the zoom drifts toward the subject, not the centre (owner, 6 Nov 2026)

    def frame(self, u):
        h, w = self.src.shape[:2]
        CW, CH = W * SS, H * SS
        s0 = max(CW / w, CH / h) * 1.02
        k = 0.16 if W < H else 0.09  # Shorts move faster (a slow drift reads as a still on a phone)
        z = {"push": 1.0 + k * u, "pull": 1.0 + k - k * u}.get(self.move, 1.05)
        s = s0 * z
        half_w, half_h = CW / (2 * s), CH / (2 * s)
        cx, cy0 = w / 2, h / 2
        if self.focus is not None and self.move in ("push", "pull"):
            t = u if self.move == "push" else 1 - u  # tighter = nearer the subject
            cx = w / 2 + (self.focus[0] * w - w / 2) * t
            cy0 = h / 2 + (self.focus[1] * h - h / 2) * t
        if self.move in ("pan_left", "pan_right"):
            span = max(0.0, w / 2 - half_w)
            cx = w / 2 + span * (0.8 * (u - 0.5)) * (1 if self.move == "pan_right" else -1) * 2
        if FX:  # a slow handheld drift, so a still never feels frozen
            cx += w * 0.004 * np.sin(2 * np.pi * (0.9 * u + 0.13 * (hash(self.move) % 7)))
            cy0 += h * 0.003 * np.sin(2 * np.pi * (0.7 * u + 0.37))
        cx = float(np.clip(cx, half_w, w - half_w))
        cy = float(np.clip(cy0, half_h, h - half_h))
        M = np.float32([[s, 0, CW / 2 - s * cx], [0, s, CH / 2 - s * cy]])
        big = cv2.warpAffine(self.src, M, (CW, CH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


def subject(img):
    """Where the painting's subject is, as (x, y) in 0-1, plus the faces' centre (or None): faces first (OpenCV's
    frontal and profile detectors), else the most detailed region (blurred edge energy), which is where a figure,
    animal or object is and empty sky or floor isn't."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (w // 2, h // 2))
    found = []
    if hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 dropped the Haar detectors: then the detail map alone
        for xml in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
            det = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
            found += [tuple(f) for f in det.detectMultiScale(small, 1.1, 5, minSize=(24, 24))]
    faces = None
    if found:
        found = sorted(found, key=lambda f: -f[2] * f[3])[:3]
        faces = (float(np.mean([(x + fw / 2) * 2 / w for x, y, fw, fh in found])),
                 float(np.mean([(y + fh / 2) * 2 / h for x, y, fw, fh in found])))
    e = cv2.GaussianBlur(np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1)), (0, 0), small.shape[1] / 12)
    e = np.maximum(e - np.percentile(e, 60), 0)
    ys, xs = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    e = e * np.clip(1.25 - ys / e.shape[0], 0.25, 1.0)  # floors and rocks are detailed too: favour where heads are
    tot = e.sum() or 1.0
    ex, ey = float((xs * e).sum() / tot / e.shape[1]), float((ys * e).sum() / tot / e.shape[0])
    if faces:  # the body is below the face: aim a little lower than the faces, toward the detail
        return (0.7 * faces[0] + 0.3 * ex, min(0.85, 0.6 * (faces[1] + 0.12) + 0.4 * ey), faces)
    return (ex, 0.6 * ey + 0.4 * 0.42, None)


class Panel:
    """Logos & Legends Shorts (owner, 8 Nov 2026): the episode's 16:9 painting whole and sharp across the middle of
    the vertical frame under a slow camera move, on the brand's dark ground: the hook headline stays above it and
    the captions sit below it (a 9:16 crop of a landscape painting kept a third of it, blown up 2x)."""
    TOP = 0.36  # the picture's top edge (fraction of the height): headline above, captions below

    def __init__(self, img, move):
        self.pw, self.ph = W, round(W * 9 / 16)
        h, w = img.shape[:2]
        f = max(SS * self.pw / w, SS * self.ph / h) * 1.12
        self.src = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_LANCZOS4)
        self.move = move
        g = np.linspace(0, 1, H)[:, None, None]
        top, bot = np.array(K.INK[::-1], np.float32), np.array(K.INK[::-1], np.float32) * 0.45
        self.ground = (top * (1 - g) + bot * g).repeat(W, axis=1).astype(np.uint8)

    def frame(self, u):
        h, w = self.src.shape[:2]
        cw, ch = self.pw * SS, self.ph * SS
        z = {"push": 1.0 + 0.10 * u, "pull": 1.10 - 0.10 * u}.get(self.move, 1.04)
        s0 = max(cw / w, ch / h) * 1.02 * z
        hw, hh = cw / (2 * s0), ch / (2 * s0)
        dx = {"pan_left": 1, "pan_right": -1}.get(self.move, 0) * (w / 2 - hw) * (u - 0.5) * 0.8
        cx, cy = w / 2 + dx, h / 2
        crop = self.src[int(cy - hh):int(cy + hh), int(cx - hw):int(cx + hw)]
        pic = cv2.resize(crop, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        fr = self.ground.copy()
        y = int(H * self.TOP)
        fr[y:y + self.ph] = pic
        return fr


class Split:
    """Shorts: a long shot cut into parts (the painting, then tighter crops of it) so the picture changes every
    ~2 s (29 Oct: one painting held 4-5 s and viewers swiped after 13 s of 41; 31 Oct: every 2-3 s -> ~2 s).
    The second crop sits in the upper third, where faces are."""

    def __init__(self, img, parts=2):
        h, w = img.shape[:2]
        fx, fy, faces = subject(img)  # the crops follow the people (owner, 6 Nov 2026: zooms landed on empty areas)

        def around(cx, cy, frac):
            cw, ch = int(w * frac), int(h * frac)
            x0 = int(np.clip(cx * w - cw / 2, 0, w - cw))
            y0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
            return img[y0:y0 + ch, x0:x0 + cw]
        # mild crops (owner, 7 Nov 2026: pictures didn't match the line; a 2x crop aimed at the busiest texture
        # showed chariot wheels, not the person or the mountain the line was about): 0.64/0.5 -> 0.8/0.66
        crop = around(fx, fy, 0.8)
        face = around(faces[0], faces[1], 0.66) if faces else around(fx, fy - 0.1, 0.66)  # the upper body, not the feet
        self.parts = [Shot(img, "push", (fx, fy)), Shot(crop, "pull"), Shot(face, "push")][:parts]

    def frame(self, u):
        n = len(self.parts)
        i = min(n - 1, int(u * n))
        return self.parts[i].frame(u * n - i)


def hook_card(text):
    """Shorts: the question as a big headline over the first seconds, so the scroll stops on it."""
    img = Image.new("RGBA", (W, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    import re
    text = re.sub(r"[^\w\s?!'’.,:\u0900-\u097F-]", "", text).strip()  # the display font has no emoji (they drew as boxes)
    words, lines, f = text.upper().split(), [], K.display(112)
    for wd in words:  # wrap at ~86% width
        if lines and d.textlength(lines[-1] + " " + wd, font=f) < W * 0.86:
            lines[-1] += " " + wd
        else:
            lines.append(wd)
    y = 260 - 62 * (len(lines) - 1)
    for ln in lines[:3]:
        tw = d.textlength(ln, font=f)
        d.rectangle((W / 2 - tw / 2 - 24, y - 66, W / 2 + tw / 2 + 24, y + 62), fill=K.YEL + (255,))
        d.text((W / 2, y), ln, font=f, anchor="mm", fill=K.INK + (255,))
        y += 132
    a = np.asarray(img)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0


def cta_card(text):
    """The Short's last seconds: a bright bar sending viewers to the full episode (owner, 8 Nov 2026: "to know
    those, watch the full episode"), with a play triangle drawn (the fonts have no arrow glyph)."""
    f = K.display(64)
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    while d.textlength(text, font=f) > W * 0.72 and f.size > 40:
        f = K.display(f.size - 2)
    tw, bh = int(d.textlength(text, font=f)), 150
    img = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0 = (W - tw - 110) // 2
    d.rounded_rectangle((x0 - 30, 14, x0 + tw + 140, bh - 14), 26, fill=K.YEL + (255,))
    d.polygon([(x0 + 10, bh / 2 - 30), (x0 + 10, bh / 2 + 30), (x0 + 62, bh / 2)], fill=K.RED + (255,))
    d.text((x0 + 92, bh / 2), text, font=f, anchor="lm", fill=K.INK + (255,))
    a = np.asarray(img)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0


def video_mark():
    """brand.json "video_mark": the channel banner's name + tagline as a small panel at the top of every frame
    (owner, 1 Nov 2026: Devlok Katha's videos carry the banner's text, in the banner's colours)."""
    m = K.B.get("video_mark")
    if not m:
        return None
    # see-through text only, no panel (owner: "not in full opacity, nothing gets blocked")
    ft, fs = K.display(58), K.display(28)
    img = Image.new("RGBA", (W, 130), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((W / 2, 44), m["title"], font=ft, anchor="mm", fill=tuple(m["fg"]) + (255,), stroke_width=3, stroke_fill=tuple(m["bg"]) + (255,))
    d.text((W / 2, 98), m["tagline"], font=fs, anchor="mm", fill=tuple(m["bg"]) + (255,), stroke_width=2, stroke_fill=tuple(m["ink"]) + (255,))
    a = np.asarray(img)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0 * float(m.get("opacity", 0.6))


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
    """Shorts: 2-3 word groups in heavy white caps with a black outline, mid-lower. Long-forms: 7-word
    subtitles on a soft box near the bottom. Both timed across each
    shot's narration by word length (the voice reads at a steady pace)."""

    def __init__(self, groups, short=True):
        self.groups, self.cache, self.short = groups, {}, short  # [(t0, t1, text)]
        self.short_y = 0.68  # the caption band's centre (the panel layout puts it under the picture)
        self.soft = None if short else K.B.get("long_captions")  # owner, 7 Nov 2026: the aesthetic long-form style
        self.spans = []  # the faint bottom gradient stays on across short gaps (no flicker between groups)
        for t0, t1, _ in groups:
            if self.spans and t0 - self.spans[-1][1] < 1.0:
                self.spans[-1][1] = max(self.spans[-1][1], t1)
            else:
                self.spans.append([t0, t1])
        self.shade = None

    @staticmethod
    def for_shot(text, start, dur, size=3):
        words = text.split()
        # a group never runs across a sentence/clause end (29 Oct QC: "OPEN POP THE" read as nonsense)
        chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) == size or w[-1] in ".,!?;:।":
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        total = sum(len(w) + 1 for w in words) or 1
        out, t = [], start
        for c in chunks:
            d = dur * sum(len(w) + 1 for w in c) / total
            out.append((t, t + d, " ".join(c)))
            t += d
        return out

    def draw(self, frame, t):
        if self.soft:
            return self.draw_soft(frame, t)
        g = next((g for g in self.groups if g[0] <= t < g[1]), None)
        if not g:
            return frame
        if not self.short:
            return self.draw_long(frame, g[2])
        if g[2] not in self.cache:
            f = K.display(118)
            img = Image.new("RGBA", (W, 400), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            text = g[2].upper().replace(",", "").replace(".", "").replace("।", "")
            while d.textlength(text, font=f) > W * 0.86 and f.size > 60:
                f = K.display(f.size - 6)
            # the group's key word (its longest) in the brand colour (31 Oct: keeps the eye on the screen)
            words = text.split()
            key = max(range(len(words)), key=lambda i: len(words[i])) if len(words) > 1 else -1
            sp = d.textlength(" ", font=f)
            x = W / 2 - d.textlength(text, font=f) / 2
            for i, wd in enumerate(words):
                d.text((x, 200), wd, font=f, anchor="lm", fill=(K.YEL if i == key else (255, 255, 255)) + (255,),
                       stroke_width=10, stroke_fill=(0, 0, 0, 255))
                x += d.textlength(wd, font=f) + sp
            a = np.asarray(img)
            self.cache[g[2]] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        rgb, al = self.cache[g[2]]
        age = t - g[0]
        if age < 0.12:  # pop in: 82% -> 100% over 0.12 s
            sc = 0.82 + 0.18 * age / 0.12
            sw, sh = int(W * sc), int(400 * sc)
            ox, oy = (W - sw) // 2, (400 - sh) // 2
            r2, a2 = np.zeros_like(rgb), np.zeros_like(al)
            r2[oy:oy + sh, ox:ox + sw] = cv2.resize(rgb, (sw, sh))
            a2[oy:oy + sh, ox:ox + sw] = cv2.resize(al, (sw, sh))[..., None]
            rgb, al = r2, a2
        y = int(H * self.short_y) - 200
        roi = frame[y:y + 400].astype(np.float32)
        frame[y:y + 400] = (rgb * al + roi * (1 - al)).astype(np.uint8)
        return frame

    def draw_long(self, frame, text):
        """Long-form: subtitle-style, sentence case on a soft dark box near the bottom."""
        if text not in self.cache:
            f = K.display(50)
            d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            while d.textlength(text, font=f) > W * 0.8 and f.size > 34:
                f = K.display(f.size - 2)
            tw, bh = int(d.textlength(text, font=f)), 100
            img = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(((W - tw) // 2 - 28, 8, (W + tw) // 2 + 28, bh - 8), 18, fill=(0, 0, 0, 150))
            d.text((W / 2, bh / 2), text, font=f, anchor="mm", fill=(255, 255, 255, 255))
            a = np.asarray(img)
            self.cache[text] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        rgb, al = self.cache[text]
        y = int(H * 0.86) - rgb.shape[0] // 2
        roi = frame[y:y + rgb.shape[0]].astype(np.float32)
        frame[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
        return frame


    def draw_soft(self, frame, t):
        """Long-form with brand "long_captions" (owner, 7 Nov 2026: simpler font, medium size, the painting
        stays whole): no box, warm off-white text with a soft shadow over a faint dark gradient at the bottom,
        fading in and out (0.3 s gradient, 0.12 s text)."""
        span = next((s for s in self.spans if s[0] <= t < s[1]), None)
        if span is None:
            return frame
        pill = self.soft.get("style") == "pill"  # LL (owner, 7 Nov 2026): white text in a dark rounded box
        if not pill:
            if self.shade is None:
                y = np.arange(H, dtype=np.float32)
                self.shade = (0.68 * np.clip((y - H * 0.65) / (H * 0.35), 0, 1) ** 1.3)[:, None, None]
            fade = min(1.0, (t - span[0]) / 0.3, (span[1] - t) / 0.3)
            frame = (frame.astype(np.float32) * (1 - self.shade * fade)).astype(np.uint8)
        g = next((g for g in self.groups if g[0] <= t < g[1]), None)
        if not g:
            return frame
        text = g[2]
        if text not in self.cache:
            st = self.soft

            def font(size):
                f = ImageFont.truetype(str(K.HERE / st["font"]), size, layout_engine=ImageFont.Layout.RAQM
                                       if features.check("raqm") else ImageFont.Layout.BASIC)
                if st.get("weight"):  # a variable font's weight axis (Inter: optical size, weight)
                    axes = f.get_variation_axes()
                    f.set_variation_by_axes([st["weight"] if a["name"] in (b"Weight", "Weight") else a["default"] for a in axes])
                return f
            f = font(st.get("size", 44))
            d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            while d.textlength(text, font=f) > W * 0.8 and f.size > 32:
                f = font(f.size - 2)
            bh = 140
            if pill:
                img = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                tw, ph = d.textlength(text, font=f), f.size + 44
                d.rounded_rectangle(((W - tw) / 2 - 34, (bh - ph) / 2, (W + tw) / 2 + 34, (bh + ph) / 2),
                                    radius=ph // 2, fill=(10, 10, 14, 190))
                d.text((W / 2, bh / 2), text, font=f, anchor="mm", fill=(255, 255, 255, 255))
                a = np.asarray(img)
                self.cache[text] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        if text not in self.cache:
            st = self.soft
            bh = 140
            sh = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
            ImageDraw.Draw(sh).text((W / 2, bh / 2 + 3), text, font=f, anchor="mm", fill=(0, 0, 0, 255),
                                    stroke_width=3, stroke_fill=(0, 0, 0, 255))
            img = sh.filter(ImageFilter.GaussianBlur(4))
            ImageDraw.Draw(img).text((W / 2, bh / 2), text, font=f, anchor="mm", fill=(250, 243, 230, 255))
            a = np.asarray(img)
            self.cache[text] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        rgb, al = self.cache[text]
        # fade only at the edges of a run of groups: back-to-back groups swap without a dip
        after = any(abs(h[1] - g[0]) < 0.05 for h in self.groups)
        follows = any(abs(h[0] - g[1]) < 0.05 for h in self.groups)
        al = al * min(1.0, 1.0 if after else (t - g[0]) / 0.12, 1.0 if follows else (g[1] - t) / 0.12)
        y = int(H * 0.885) - rgb.shape[0] // 2
        roi = frame[y:y + rgb.shape[0]].astype(np.float32)
        frame[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
        return frame


_LT = {}
MOVES_CYCLE = ("pan_right", "pull", "pan_left", "push")
# a colour grade per episode (script "grade"; owner, 7 Nov 2026: Maruti 800 in Cross process), the same settings as the
# dashboard's filter previews (clipper/filters.json), applied while encoding
GRADES = {"cross-process": "eq=contrast=1.15:saturation=1.3,hue=h=-12",
          "cinematic": "eq=contrast=1.1:saturation=1.15,colorbalance=rs=0.05:bs=-0.05:rh=0.04:bh=-0.04",
          "warm": "colorbalance=rs=0.06:gs=0.02:bs=-0.06,eq=saturation=1.08"}
FX = False  # brand "motion_fx" (Logos & Legends, owner 7 Nov 2026: "make it more engaging, with some motion")
YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
AMOUNT = re.compile(r"₹\s?[\d,]+(?:\.\d+)?\s*(?:करोड़|लाख|crore|lakh)?|\b\d+(?:\.\d+)?\s?%")


def transition(a, b, p, kind):
    """a (outgoing) into b (incoming), p 0->1. dissolve / whip (a fast blurred slide) / zoom (a pushes through) /
    leak (a warm light leak at the crossover) / dip (through black, for chapters)."""
    if kind == "whip":
        sh = int(W * p)
        out = np.zeros_like(a)
        out[:, :W - sh] = a[:, sh:]
        out[:, W - sh:] = b[:, :sh]
        k = max(1, int(60 * np.sin(np.pi * p)))
        return cv2.blur(out, (k * 2 + 1, 1))
    if kind == "zoom":
        z = 1 + 0.35 * p
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        a2 = cv2.warpAffine(a, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(a2, 1 - p, b, p, 0)
    if kind == "dip":
        return (a * max(0.0, 1 - 2 * p) if p < 0.5 else b * min(1.0, 2 * p - 1)).astype(np.uint8)
    out = cv2.addWeighted(a, 1 - p, b, p, 0)
    if kind == "leak":
        glow = np.sin(np.pi * p) * 0.55
        g = np.linspace(1, 0, W, dtype=np.float32)[None, :, None] * np.array([90, 170, 255], np.float32)  # BGR warm
        out = np.clip(out.astype(np.float32) + g * glow, 0, 255).astype(np.uint8)
    return out


_CNT = {}


def counter(frame, text, t):
    """An amount counting up as it's said ("₹47,500", "60%"): big gold digits, top-left, ~2.8 s."""
    m = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not m:
        return frame
    val = float(m.group().replace(",", ""))
    k = min(1.0, t / 1.1)
    cur = val * (1 - (1 - k) ** 3)
    num = f"{cur:,.1f}" if "." in m.group() else f"{int(round(cur)):,}"
    shown = text[:m.start()] + num + text[m.end():]
    a = min(1.0, t / 0.25, max(0.0, (2.8 - t) / 0.4))
    f = K.body(110, "Bold")
    img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((86, 85), shown, font=f, anchor="lm", fill=K.YEL + (255,), stroke_width=6, stroke_fill=(10, 10, 10, 220))
    arr = np.asarray(img)
    rgb, al = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), arr[..., 3:4] / 255.0 * a
    y = int(H * 0.12)
    roi = frame[y:y + 170].astype(np.float32)
    frame[y:y + 170] = (rgb * al + roi * (1 - al)).astype(np.uint8)
    return frame


def riser(sec=1.4, seed=9):
    """A rising swell before a chapter card: noise through a band-pass sweeping upward, getting louder."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    x = rng.standard_normal(n).astype(np.float32)
    out, lp = np.zeros(n, np.float32), 0.0
    for i in range(n):
        a = 0.002 + 0.12 * (i / n) ** 2
        lp += a * (x[i] - lp)
        out[i] = lp
    return out * np.linspace(0.05, 1, n, dtype=np.float32) ** 2


def lower_third(frame, text, alpha):
    """A documentary lower-third (after The Quiet Story): the label in spaced capitals with a thin accent line,
    bottom-left above the captions, e.g. "J.C. MAHINDRA · INDIA'S FIRST STEEL CONTROLLER"."""
    if text not in _LT:
        main, _, sub = text.partition("·")
        f1, f2 = K.body(34, "Bold"), K.body(22)
        spaced = lambda t: " ".join(t.strip().upper())  # letter-spacing, like the reference
        a, b = spaced(main), spaced(sub) if sub.strip() else ""
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        w = int(max(d0.textlength(a, font=f1), d0.textlength(b, font=f2) if b else 0)) + 40
        img = Image.new("RGBA", (w, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 8, 4, 104 if b else 60), fill=K.YEL + (255,))
        d.text((20, 10), a, font=f1, fill=(250, 245, 235, 255), stroke_width=2, stroke_fill=(0, 0, 0, 160))
        if b:
            d.text((20, 62), b, font=f2, fill=(220, 210, 190, 255), stroke_width=2, stroke_fill=(0, 0, 0, 160))
        arr = np.asarray(img)
        _LT[text] = (cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), arr[..., 3:4] / 255.0)
    rgb, al = _LT[text]
    al = al * max(0.0, alpha)
    x, y = 70, int(H * 0.70)
    h, w = rgb.shape[:2]
    w = min(w, W - x)
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (rgb[:, :w] * al[:, :w] + roi * (1 - al[:, :w])).astype(np.uint8)
    return frame


def whoosh(sec=0.45, seed=5):
    """A soft air whoosh, made here (no sample, no licence): noise through a band-pass sweeping up then down."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    x = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(x)
    fq = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(spec * np.exp(-((fq - 1800) / 1400) ** 2), n).astype(np.float32)
    u = np.linspace(0, 1, n)
    return x * np.sin(np.pi * u) ** 2


def bell(sec=2.2):
    """A temple-bell strike, made here: inharmonic partials with long decays."""
    t = np.arange(int(sec * SR)) / SR
    parts = [(523, 1.0, 1.6), (1046 * 1.003, 0.5, 1.1), (1410, 0.35, 0.8), (2093, 0.2, 0.5), (2790, 0.12, 0.35)]
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, a, d in parts)
    return (x * np.minimum(1, t / 0.004)).astype(np.float32)


def load_sfx(name):
    if name == "bell":
        return bell()
    import soundfile as sf
    x, sr = sf.read(K.asset(f"sfx/{name}.wav"), dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    return np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x).astype(np.float32)


def load_mp3(path, seconds):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    reps = int(np.ceil(seconds * SR / max(1, len(x))))
    return np.tile(x, reps)[:int(seconds * SR)]


def main(script_path, out_dir):
    global W, H, XF, FX
    script = json.loads(Path(script_path).read_text())
    short = script.get("format") == "short"
    FX = not short and bool(K.B.get("motion_fx"))
    PANEL = short and K.B.get("short_layout") == "panel"  # Logos & Legends Shorts: the landscape painting whole (8 Nov 2026)
    if short:
        W, H, XF = 1080, 1920, 0.15
    caps = []
    out = Path(out_dir)
    vo = json.loads((out / "vo.json").read_text())
    items, n = [], 0  # (start, dur, visual, vo_file, section_id)
    cuts = []  # Shorts: every picture change (new shot or new crop), for the whoosh
    explainers = []  # time ranges of the science diagrams: the music sits a little higher under them
    t = 0.0
    chapters = []
    labels = []  # (t0, t1, text): lower-thirds
    titles = []  # (t0, t1, text): big phrases over the picture
    counters, years_seen, cards_at, last_counter = [], set(), [], -99.0  # FX: counting amounts, auto date labels
    gfx = []  # (t0, kind, payload): date callouts, stamps, keywords, typed cards, logos, merge graphics
    intro_at = None
    for si, sec in enumerate(script["sections"]):
        if FX and K.B.get("intro") == "title" and intro_at is None and si > 0 and sec.get("card"):
            # the show's title sequence between the cold open and the first chapter
            name = script.get("intro_title") or script.get("title", "").split("|")[0].strip()
            name = re.sub(r"[^\w\s&'.:-]", "", name).strip()
            intro_at = t
            items.append((t, 4.0, Intro(name), None, "intro"))
            t += 4.0
        if sec.get("sting"):
            items.append((t, 2.2, Seq(sting_frames(int(2.2 * FPS))), None, sec["id"]))
            t += 2.2
            continue
        if sec.get("card"):
            chapters.append((t, sec["card"].title()))
            if K.B.get("chapter_style") == "numbered" and not short:  # the reference's navy numbered card
                cards_at.append(t)
                num = len(chapters) - (1 if chapters[0][1] == "Intro" else 0)
                items.append((t, 3.0, Shot(numbered_card(sec["card"], num), "push"), None, sec["id"]))
                t += 3.0
            elif K.B.get("chapter_style") == "film" and not short:
                first = next((x for x in [out / f"shot-{n + 1:02d}.png"] if x.exists()), None)
                cards_at.append(t)
                items.append((t, 3.0, Shot(film_card(sec["card"], len(chapters) - (1 if chapters[0][1] == "Intro" else 0), first), "push"),
                              None, sec["id"]))
                t += 3.0
            else:
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
                src = s.get("from")  # an episode's own painting (Logos & Legends Shorts from a long-form, 8 Nov 2026)
                img = cv2.imread(str(out / (f"shot-{int(src.split('-')[0]):02d}{'-' + src.split('-')[1] if '-' in src else ''}.png"
                                            if src else f"shot-{n:02d}.png")))
                if s.get("overlay"):  # science drawn onto the painting (a crack, a hole) where SDXL can't
                    img = bgr(OVERLAYS[s["overlay"]](Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                parts = 3 if d > 5.4 else 2 if d > 2.6 else 1
                if short and PANEL:  # a landscape painting kept whole in a vertical frame
                    vis, parts = Panel(img, MOVES_CYCLE[n % 4]), 1
                else:
                    vis = Split(img, parts) if short and parts > 1 else Shot(img, s.get("move", "push"))
                if short:
                    cuts += [t + d * i / parts for i in range(1, parts)]
            extras = [] if short or s.get("diagram") else sorted(
                out.glob(f"shot-{n:02d}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
            if extras:  # a new picture every ~4 s (owner, 7 Nov 2026, after The Quiet Story): the line's pictures in turn
                pics = [vis] + [Shot(cv2.imread(str(x)), MOVES_CYCLE[(n + i) % 4]) for i, x in enumerate(extras)]
                part = d / len(pics)
                for i, v in enumerate(pics):
                    items.append((t + i * part, part, v, out / f"vo-{n:02d}.wav" if i == 0 else None, sec["id"]))
            else:
                items.append((t, d, vis, out / f"vo-{n:02d}.wav", sec["id"]))
            if s.get("title") and not short:  # a big phrase over the picture
                titles.append((t + 0.2, t + min(d, 3.2), s["title"]))
            if s.get("label") and not short:  # a lower-third (Logos & Legends, 7 Nov 2026: names, dates, places)
                labels.append((t + 0.3, t + min(d, 4.3), s["label"]))
                years_seen.update(YEAR.findall(s["label"]))
            line = s.get("caption") or s["vo"]
            if FX and s.get("date"):
                gfx.append((t + 0.2, "date", s["date"]))
                years_seen.update(YEAR.findall(s["date"]))
            elif FX and not s.get("label"):  # every first mention of a year: a big date callout (or a lower-third)
                y = next((y for y in YEAR.findall(line) if y not in years_seen), None)
                if y:
                    years_seen.add(y)
                    if K.B.get("date_style") == "callout":
                        gfx.append((t + 0.2, "date", y))
                    else:
                        labels.append((t + 0.3, t + min(d, 3.6), y))
            if FX:
                for kind in ("stamp", "keyword", "typecard", "logo", "merge"):
                    if s.get(kind):
                        gfx.append((t + (0.6 if kind == "stamp" else 0.25), kind, s[kind]))
            if FX and t - last_counter > 12:  # an amount or a percentage counts up as it's said
                m = AMOUNT.search(line)
                if m and not YEAR.fullmatch(m.group().strip()):
                    counters.append((t + 0.2, m.group().strip()))
                    last_counter = t
            if short:  # the very first words show at 0.0 s, not after a pause
                caps += Captions.for_shot(s.get("caption") or s["vo"], t + (0.0 if t == 0 else 0.15), vo[f"{n:02d}"])
            elif not s.get("diagram"):  # diagrams carry their own labels
                caps += Captions.for_shot(s.get("caption") or s["vo"], t + 0.15, vo[f"{n:02d}"], size=7)
            t += d
    if short:  # no end screen: Shorts loop back to the start
        total = t
    else:
        end_img = Image.new("RGB", (W, H), K.INK)
        lg = Image.open(K.asset("logo.png")).convert("RGBA").resize((300, 300), Image.LANCZOS)
        end_img.paste(lg, ((W - 300) // 2, 300), lg)
        K.wordmark(ImageDraw.Draw(end_img), W / 2, 680, 72, anchor="mm")
        items.append((t, 6.0, Still(bgr(K.grain(end_img))), None, "outro"))
        total = t + 6.0
    captions = Captions(caps, short) if caps else None
    if captions and PANEL:
        captions.short_y = Panel.TOP + 9 / 16 * W / H + 0.105

    # ---- audio: narration + music by section mood
    voice = np.zeros(int(total * SR) + SR, np.float32)
    for start, dur, _, f, _ in items:
        if f is None:
            continue
        import soundfile as sf
        x, sr = sf.read(f, dtype="float32")
        x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x).astype(np.float32)
        i = int((start + (0.0 if short and start == 0 else 0.15)) * SR)  # Shorts: the voice starts at once
        voice[i:i + len(x)] += x[:len(voice) - i]
    mood_of = {sid: mood for mood, sids in (script.get("music") or {}).items() for sid in sids}
    # a mood without a track falls back to reflective (31 Oct: "playful" has no track; s29-s31 crashed every run)
    track = lambda m: K.asset(f"music/{m}.mp3") if K.asset(f"music/{m}.mp3").exists() else K.asset("music/reflective.mp3")
    tracks = {m: load_mp3(track(m), total + 1) for m in set(mood_of.values())}
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
    # long-form (owner, 7 Nov 2026: "a good faint bg music"): -18 dB under the voice, -8 dB in pauses and cards
    talk_db, pause_db, read_db = (-12, -4, -8) if short else (-18, -8, -12)
    under = np.full(len(gain), 10 ** (talk_db / 20))
    for a, b in explainers:  # under the explainers: a clear, curious bed while the viewer reads
        under[int(a * SR):int(b * SR)] = 10 ** (read_db / 20)
    under = np.convolve(under, np.ones(SR // 2) / (SR // 2), mode="same")  # ease in/out over 0.5 s
    level = voice_rms * (under * gain + 10 ** (pause_db / 20) * (1 - gain))
    fade = np.minimum(1, np.arange(len(music)) / ((0.05 if short else 1.5) * SR)) * \
        np.minimum(1, (len(music) - np.arange(len(music))) / (3 * SR))
    mix = voice + music * level * fade
    if short:
        sfx = np.zeros_like(mix)

        def put(x, at, db):
            i = int(at * SR)
            if 0 <= i < len(sfx):
                sfx[i:i + len(x)] += x[:len(sfx) - i] * voice_rms * 10 ** (db / 20) / rms(x)

        style = K.B.get("sfx") or {}
        last = items[-1][0] if items else 0
        for c in cuts + [it[0] for it in items[1:]]:
            if abs(c - last) > 0.05:
                put(whoosh(), max(0, c - 0.18), -15)
        put(load_sfx(style.get("reveal", "impact")), last, -9)
        if style.get("open"):
            put(load_sfx(style["open"]), 0, -12)
        mix = mix + sfx
    elif FX:  # sound design (owner, 7 Nov 2026): a soft whoosh on new shots, a swell into chapters, a hit on titles
        sfx = np.zeros_like(mix)

        def put(x, at, db):
            i = int(at * SR)
            if 0 <= i < len(sfx):
                sfx[i:i + len(x)] += x[:len(sfx) - i] * voice_rms * 10 ** (db / 20) / rms(x)
        for it in items[1:]:
            if it[3] is not None:
                put(whoosh(0.35), max(0, it[0] - 0.2), -27)
        page = K.asset("sfx/page.wav")
        for c in cards_at:  # the brand's page sound on each chapter card, else a rising swell into it
            if page.exists():
                put(load_sfx("page"), c + 0.05, -14)
            else:
                put(riser(), max(0, c - 1.4), -21)
        hit = K.asset("sfx/typehit.wav").exists()
        for g0, kind, pay in gfx:
            if kind in ("date", "stamp"):
                put(load_sfx("typehit") if hit else load_sfx("impact"), g0 + (0.0 if kind == "date" else 0.1), -12 if kind == "stamp" else -15)
            elif kind == "typecard" and K.asset("sfx/typing.wav").exists():
                put(load_sfx("typing")[:int(SR * min(3.0, 0.07 * len(str(pay)) + 0.2))], g0 + 0.25, -20)
            elif kind in ("logo", "merge"):
                put(whoosh(0.3, seed=13), g0, -24)
        if intro_at is not None:
            put(riser(1.8), max(0, intro_at - 1.6), -17)
            put(load_sfx("impact"), intro_at + 0.15, -12)
        for t0, _, _ in titles:
            put(load_sfx("impact"), t0, -18)
        for t0, _ in counters:
            put(whoosh(0.25, seed=11), t0, -26)
        mix = mix + sfx
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.95
    mix = np.clip(mix, -1, 1)
    import soundfile as sf
    sf.write(out / "mix.wav", mix[:int(total * SR)], SR)

    # ---- picture
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", str(out / "mix.wav"), "-map", "0:v", "-map", "1:a",
                           *(["-vf", GRADES[script["grade"]]] if script.get("grade") in GRADES else []),
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "film", "-pix_fmt", "yuv420p",
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out / "video.mp4")],
                          stdin=subprocess.PIPE)
    hook = hook_card(script.get("hook_text") or script["title"]) if short else None
    cta = cta_card(script["cta"]) if short and script.get("cta") else None
    CTA_S = 3.0
    HOOK_S = 2.8
    mark = video_mark() if short else None
    k = 0
    for idx, (start, dur, vis, _, _) in enumerate(items):
        nf = int(round((start + dur) * FPS)) - int(round(start * FPS))
        nxt = items[idx + 1][2] if idx + 1 < len(items) else None
        for j in range(nf):
            u = j / max(1, nf - 1)
            fr = vis.frame(u)
            left = (nf - j) / FPS
            if nxt is not None and left < XF:  # into the next item
                if FX:
                    ns = items[idx + 1][0]
                    kind = "dip" if any(abs(ns - c) < 0.01 for c in cards_at) or any(abs(start - c) < 0.01 for c in cards_at) \
                        else (("whip", "zoom", "leak", "dissolve")[idx % 4] if items[idx + 1][3] is not None else "dissolve")
                    fr = transition(fr, nxt.frame(0.0), 1 - left / XF, kind)
                else:
                    fr = cv2.addWeighted(fr, left / XF, nxt.frame(0.0), 1 - left / XF, 0)
            for g0, kind, pay in gfx:
                gt = k / FPS - g0
                if gt < 0 or gt > {"date": 2.0, "stamp": 2.4, "keyword": 2.6, "typecard": 3.4, "logo": 2.8, "merge": 4.2}[kind]:
                    continue
                if kind == "date":
                    fr = date_callout(fr.copy(), pay, gt)
                elif kind == "stamp":
                    fr = stamp(fr.copy(), pay, gt)
                elif kind == "keyword":
                    fr = keyword(fr.copy(), pay, gt)
                elif kind == "typecard":
                    fr = typecard(fr.copy(), pay, gt)
                elif kind == "logo":
                    fr = logo_card(fr.copy(), pay, out, gt)
                elif kind == "merge":
                    fr = merge_graphic(fr.copy(), pay, out, gt)
            cnt = next((c for c in counters if 0 <= k / FPS - c[0] < 2.8), None)
            if cnt:
                fr = counter(fr.copy(), cnt[1], k / FPS - cnt[0])
            ttl = next((x for x in titles if x[0] <= k / FPS < x[1]), None)
            if ttl:
                fr = title_overlay(fr.copy(), ttl[2], min(1.0, (k / FPS - ttl[0]) / 0.35, (ttl[1] - k / FPS) / 0.35))
            lab = next((l for l in labels if l[0] <= k / FPS < l[1]), None)
            if lab:
                fr = lower_third(fr.copy(), lab[2], min(1.0, (k / FPS - lab[0]) / 0.4, (lab[1] - k / FPS) / 0.4))
            if captions:
                fr = captions.draw(fr.copy(), k / FPS)
            if hook is not None and (PANEL or k / FPS < HOOK_S):  # the question headline, fading out over its last 0.3 s
                rgb, al = hook                                     # (the panel layout keeps it above the picture)
                al = al if PANEL else al * min(1.0, (HOOK_S - k / FPS) / 0.3)
                y = int(H * Panel.TOP) - 540 if PANEL else int(H * 0.12)
                roi = fr[y:y + 520].astype(np.float32)
                fr[y:y + 520] = (rgb * al + roi * (1 - al)).astype(np.uint8)
            if cta is not None and k / FPS > total - CTA_S:  # "the answers: the full episode" (Logos & Legends Shorts)
                rgb, al = cta
                al = al * min(1.0, (k / FPS - (total - CTA_S)) / 0.3)
                y = int(H * 0.86) - rgb.shape[0] // 2
                roi = fr[y:y + rgb.shape[0]].astype(np.float32)
                fr[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
            if mark is not None:  # the banner's name + tagline at the top, above the hook (it starts at 12%)
                rgb, al = mark
                y = 40
                roi = fr[y:y + rgb.shape[0]].astype(np.float32)
                fr[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
            ff.stdin.write(fr.tobytes())
            k += 1
    ff.stdin.close()
    ff.wait()
    (out / "chapters.txt").write_text("" if short else"\n".join(f"{int(s // 60):02d}:{int(s % 60):02d} {name}" for s, name in chapters))
    print(f"video {total / 60:.1f} min, {k} frames -> {out / 'video.mp4'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
