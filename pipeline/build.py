"""The edit: paintings + diagrams + narration + music -> the 1920x1080 episode, or with "format": "short" the
1080x1920 Short (fast cuts, bold 2-3 word captions, no cards, no end screen: Shorts loop).

    python pipeline/build.py episodes/<slug>/script.json out/<slug>/   ->  video.mp4, chapters.txt

Timeline per the brand: cold open, 2 s logo sting, then each section behind a chapter card (ink, yellow
Anton title, red underline). Every shot lasts its narration + a breath, with a slow push-in, pull-out or
pan over the painting (supersampled 2x, no shimmer), 0.4 s dissolves. Music by section mood (licensed library tracks
in brand/music), kept low under the voice. Ends on a 6 s end screen (logo) for YouTube's end elements.
"""
import json
import os
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
import words  # noqa: E402

W, H, FPS, SR = 1920, 1080, 30, 48000
XF = 0.4  # dissolve seconds
BREATH = float(K.B.get("breath", 0.45))  # air after each line; brand.json "breath" (The Hidden Why 0.25, 13 Nov 2026 voice review)


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


DEV = re.compile(r"[\u0900-\u097F]")


def _bf(text, size, weight="SemiBold"):
    """The body font, or the Hindi one when the text has Devanagari (Inter has none: Hindi showed as boxes)."""
    return _HI(size) if DEV.search(text) else K.body(size, weight)


def _dim(frame, k):
    return (frame.astype(np.float32) * (1 - k)).astype(np.uint8)


def date_callout(frame, text, t, dur=2.0):
    """A big bold date over a softened, dimmed frame ("OCTOBER 1947"), underlined; a typewriter hit lands on it."""
    a = min(1.0, t / 0.2, (dur - t) / 0.3)
    frame = cv2.addWeighted(frame, 1 - 0.6 * a, cv2.GaussianBlur(frame, (0, 0), 9), 0.6 * a, 0)
    frame = _dim(frame, 0.25 * a)
    f = _bf(text, 130, "Bold")
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
        f = _bf(text, 110, "Bold")
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
    ImageDraw.Draw(img).text((W / 2, 100), ("  ".join(text.upper()) if not DEV.search(text) else text), font=_bf(text, 66), anchor="mm", fill=(250, 246, 238, 255))
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


# ---- story beats (Logos & Legends, owner 8 Nov 2026: "the video for very long goes for changing images and continuous
# captions... add sound effects on those visual details and hook the attention"): a headline, a place, a then-vs-now and
# a quote, each with its own sound, about every 15 s (studio/ll_beats.py writes them into the script)

def _font(rel, size):
    lay = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    p = K.HERE / rel
    return ImageFont.truetype(str(p if p.exists() else K.HERE / K.B["fonts"]["display"]), size, layout_engine=lay)


_HI = lambda size: _font("logos-and-legends/fonts/Mukta-Bold.ttf", size)
_SERIF = lambda size: _font("devlok-katha/fonts/TiroDevanagariHindi-Regular.ttf", size)
_BEATS = {}
ease = lambda x: 1 - (1 - min(1.0, max(0.0, x))) ** 3


def punch(frame, t, k=0.06):
    """The camera jumps in a little as a beat lands, easing back over 0.3 s."""
    if t >= 0.3:
        return frame
    z = 1 + k * (1 - ease(t / 0.3))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def _wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        if cur and d.textlength(cur + " " + w, font=f) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def news_sheet(spec):
    """An aged front page: a made-up masthead (never a real paper's), the date line, the headline in heavy type,
    columns of grey text bars and a blank photo box."""
    key = ("news", json.dumps(spec, ensure_ascii=False))
    if key in _BEATS:
        return _BEATS[key]
    pw, ph = 1180, 760
    rng = np.random.default_rng(11)
    base = np.full((ph, pw, 3), (232, 222, 196), np.int16) - rng.integers(0, 22, (ph, pw))[..., None]
    yy, xx = np.mgrid[0:ph, 0:pw]
    vig = (((xx - pw / 2) / pw) ** 2 + ((yy - ph / 2) / ph) ** 2) * 90  # browned edges
    img = Image.fromarray(np.clip(base - vig[..., None], 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    ink = (34, 28, 22, 255)
    d.text((pw / 2, 70), spec.get("paper") or "भारत दर्पण", font=_SERIF(84), anchor="mm", fill=ink)
    d.line((40, 128, pw - 40, 128), fill=ink, width=4)
    d.line((40, 136, pw - 40, 136), fill=ink, width=1)
    d.text((44, 156), spec.get("date", ""), font=_HI(26), anchor="lm", fill=ink)
    d.text((pw - 44, 156), "मूल्य 25 पैसे", font=_HI(26), anchor="rm", fill=ink)
    d.line((40, 176, pw - 40, 176), fill=ink, width=1)
    f = _HI(92)
    lines = _wrap(d, spec["headline"], f, pw - 100)
    while len(lines) > 2 and f.size > 56:
        f = _HI(f.size - 6)
        lines = _wrap(d, spec["headline"], f, pw - 100)
    y = 190
    for ln in lines[:3]:
        d.text((pw / 2, y), ln, font=f, anchor="ma", fill=ink)
        y += int(f.size * 1.18)
    if spec.get("sub"):
        d.text((pw / 2, y + 8), spec["sub"], font=_HI(36), anchor="ma", fill=(70, 60, 50, 255))
        y += 60
    y += 24
    d.rectangle((44, y, pw * 0.42, ph - 40), outline=ink, width=2, fill=(205, 195, 170, 255))  # the photo box
    for col in range(2):
        x0 = pw * 0.46 + col * (pw * 0.27)
        for r in range(int((ph - 40 - y) / 22)):
            ln = pw * 0.24 * (0.6 + 0.4 * rng.random()) if r % 7 != 6 else pw * 0.1
            d.rectangle((x0, y + r * 22, x0 + ln, y + r * 22 + 8), fill=(120, 110, 96, 255))
    _BEATS[key] = img
    return img


def newspaper(frame, spec, t, dur=3.8):
    """The front page spins in and lands (the old newsreel move), held over the darkened scene."""
    a = min(1.0, (dur - t) / 0.35)
    frame = _dim(cv2.GaussianBlur(frame, (0, 0), 6 * min(1.0, t / 0.3)), 0.5 * min(1.0, t / 0.3) * a)
    u = ease(t / 0.45)
    sheet = news_sheet(spec)
    sc = (0.25 + 0.75 * u) * (1 + 0.03 * t)  # then a slow push
    im = sheet.rotate((1 - u) * 540 - 3, expand=True, resample=Image.BICUBIC)
    im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.BILINEAR)
    return _paste(frame, im, W / 2 - im.width / 2, H / 2 - im.height / 2, a)


_MAP = {}


def geocode(name):
    """(lat, lon) of an Indian place from OpenStreetMap's Nominatim, once per build."""
    import urllib.parse
    import urllib.request
    try:
        q = urllib.parse.urlencode({"q": f"{name}, India", "format": "json", "limit": 1})
        req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}", headers={"User-Agent": "LogosAndLegends/1.0"})
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return (float(r[0]["lat"]), float(r[0]["lon"])) if r else None
    except Exception as e:  # noqa: BLE001 - then the map without a pin
        print(f"geocode {name}: {e}")
        return None


def place_pin(frame, spec, t, dur=3.4):
    """The place on India's map (owner, 8 Nov 2026: "whenever there's a place, mark it on map correctly"): the
    whole country, drawn from the Survey of India boundary with state lines (DataMeet, CC BY 4.0), zooms in on the
    city, a pin drops on its real position with ripples, the name in heavy type beside it. spec: {"name": "PUNE",
    "sub": "AKURDI · 1960", "lat": .., "lon": ..} (lat/lon looked up when missing)."""
    if "geo" not in _MAP:
        _MAP["geo"] = json.loads(K.asset("india-map.json").read_text())
    key = spec["name"]
    if key not in _MAP:
        _MAP[key] = (spec["lat"], spec["lon"]) if spec.get("lat") else geocode(spec.get("query") or spec["name"])
    ll = _MAP[key]
    a = min(1.0, t / 0.3, (dur - t) / 0.35)
    tgt = ll or (22.0, 82.8)
    abroad = not (6 <= tgt[0] <= 37.5 and 68 <= tgt[1] <= 97.5)  # outside India: the world map around it
    c0, span0, span1 = ((tgt[0], tgt[1]), 70.0, 16.0) if abroad else ((22.0, 82.8), 31.5 / 0.86, 31.5 / 0.86 / 2.6)
    e = ease((t - 0.15) / 0.9)  # zoom in on the place
    span = span0 * (1 - e) + span1 * e
    cy, cx = c0[0] + (tgt[0] - c0[0]) * e, c0[1] + (tgt[1] - c0[1]) * e
    cosl = np.cos(np.radians(cy))
    k = H / span
    sx, sy = W * 0.40, H * 0.5  # the place ends left of centre, its name to the right
    proj = lambda r: np.int32([[(sx + (lo - cx) * cosl * k) * 4, (sy - (la - cy) * k) * 4] for lo, la in r])
    m = np.zeros((H, W, 3), np.uint8)
    m[:] = (40, 24, 14)  # deep navy sea (BGR)
    if not _WORLD:
        MapShot({}, 1)  # loads brand/maps/world.json
    half = (W / (2 * k * cosl) + 2, H / (2 * k) + 2)
    near = [c for c in _WORLD["countries"] if c["iso"] != "IND" and not (c["bbox"][2] < cx - half[0] or c["bbox"][0] > cx + half[0]
                                                                       or c["bbox"][3] < cy - half[1] or c["bbox"][1] > cy + half[1])]
    others = [proj(r) for c in near for r in c["rings"]]
    cv2.fillPoly(m, others, (54, 38, 26), cv2.LINE_AA, shift=2)  # the neighbours, a little darker than India
    cv2.polylines(m, others, True, (84, 66, 50), 1, cv2.LINE_AA, shift=2)
    land = [proj(r) for r in _MAP["geo"]["india"]]  # India: the Survey of India boundary
    cv2.fillPoly(m, land, (62, 44, 30), cv2.LINE_AA, shift=2)
    cv2.polylines(m, [proj(r) for r in _MAP["geo"]["states"]], True, (96, 76, 58), 1, cv2.LINE_AA, shift=2)
    cv2.polylines(m, land, True, tuple(int(c) for c in K.YEL[::-1]), 3, cv2.LINE_AA, shift=2)
    frame = cv2.addWeighted(m, a, frame, 1 - a, 0)
    if ll is None:
        return frame
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    px, py = sx, sy
    pt = t - 1.0
    if pt > 0:
        drop = ease(pt / 0.3)
        y = py - (1 - drop) * 260
        for j in range(3):  # ripples
            r_t = pt - 0.3 - j * 0.3
            if 0 < r_t < 1.2:
                r = 14 + 90 * r_t
                d.ellipse((px - r, py - r * 0.5, px + r, py + r * 0.5), outline=(255, 255, 255, int(220 * (1 - r_t / 1.2))), width=4)
        red = (232, 36, 40, 255)
        d.polygon([(px - 26, y - 52), (px + 26, y - 52), (px, y)], fill=red)
        d.ellipse((px - 36, y - 100, px + 36, y - 28), fill=red)
        d.ellipse((px - 13, y - 77, px + 13, y - 51), fill=(255, 255, 255, 255))
        b = ease((pt - 0.3) / 0.35)
        if b > 0:
            x = px + 70 + 30 * (1 - b)
            d.text((x, py - 30), spec["name"].upper(), font=_HI(110), anchor="ls", fill=(255, 255, 255, int(255 * b)))
            if spec.get("sub"):
                d.text((x + 4, py - 6), spec["sub"].upper(), font=_HI(44), anchor="la", fill=K.YEL + (int(255 * b),))
    return _paste(frame, img, 0, 0, a)


def corner_label(frame, text, t, dur=4.0):
    """The scene's place and date in heavy white type, bottom left, the way the niche's best documentaries mark a
    jump in time ("BOMBAY · 1945", "16 दिसंबर 1903"); it slides up and holds for the scene."""
    a = min(1.0, t / 0.25, (dur - t) / 0.4)
    key = ("corner", text)
    if key not in _BEATS:
        f = _HI(76)
        d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = int(d0.textlength(text.upper(), font=f)) + 40
        im = Image.new("RGBA", (tw, 130), (0, 0, 0, 0))
        sh = Image.new("RGBA", (tw, 130), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((10, 70), text.upper(), font=f, anchor="lm", fill=(0, 0, 0, 200))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(6)))
        ImageDraw.Draw(im).text((10, 66), text.upper(), font=f, anchor="lm", fill=(255, 255, 255, 255))
        _BEATS[key] = im
    im = _BEATS[key]
    return _paste(frame, im, 60, H * 0.70 - 30 * (1 - ease(t / 0.3)), a)


def split_compare(frame, spec, out, t, dur=4.0):
    """Then vs now: two of the episode's paintings side by side, sliding in from either edge, a gold divider
    and a label on each ("1960 · VESPA" | "1972 · CHETAK")."""
    key = ("split", spec.get("left"), spec.get("right"))
    if key not in _BEATS:
        half = []
        for sid in (spec.get("left"), spec.get("right")):
            f = out / f"shot-{int(str(sid).split('-')[0]):02d}{'-' + str(sid).split('-')[1] if '-' in str(sid) else ''}.png"
            im = paint_read(f) if sid else None
            im = frame.copy() if im is None else im
            h, w = im.shape[:2]
            s = max((W // 2) / w, H / h)
            im = cv2.resize(im, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA)
            x0, y0 = (im.shape[1] - W // 2) // 2, (im.shape[0] - H) // 2
            half.append(im[y0:y0 + H, x0:x0 + W // 2])
        _BEATS[key] = half
    left, right = _BEATS[key]
    a = min(1.0, (dur - t) / 0.35)
    u = ease(t / 0.4)
    canvas = frame.copy()
    off = int((1 - u) * W / 2)
    canvas[:, 0:W // 2 - off] = left[:, off:]
    canvas[:, W // 2 + off:] = right[:, :W // 2 - off]
    out_fr = cv2.addWeighted(canvas, a, frame, 1 - a, 0)
    g = ease((t - 0.3) / 0.3)
    if g > 0:
        cv2.rectangle(out_fr, (W // 2 - 3, int(H / 2 - H / 2 * g)), (W // 2 + 3, int(H / 2 + H / 2 * g)), K.YEL[::-1], -1)
    b = ease((t - 0.45) / 0.3) * a
    if b > 0:
        img = Image.new("RGBA", (W, 140), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for x, txt in ((W / 4, spec.get("l", "")), (W * 3 / 4, spec.get("r", ""))):
            if txt:
                f = _HI(58)
                tw = d.textlength(txt.upper(), font=f)
                d.rounded_rectangle((x - tw / 2 - 26, 30, x + tw / 2 + 26, 112), 14, fill=(10, 10, 14, 200))
                d.text((x, 71), txt.upper(), font=f, anchor="mm", fill=(255, 255, 255, 255))
        out_fr = _paste(out_fr, img, 0, H * 0.08, b)
    return out_fr


def quote_card(frame, spec, t, dur=4.0):
    """A line that matters, on black: gold quote marks, the words appearing one by one in a serif, who said it
    under it. The music drops out under it (the shot's "drop")."""
    a = min(1.0, t / 0.3, (dur - t) / 0.4)
    frame = _dim(frame, 0.88 * a)
    text = spec["text"] if isinstance(spec, dict) else str(spec)
    words = text.split()
    step = min(0.16, 2.0 / max(1, len(words)))  # every word in by ~2.2 s
    shown = " ".join(words[:max(1, int((t - 0.2) / step) + 1)]) if t > 0.2 else ""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = _SERIF(84)
    lines = _wrap(d, text, f, W * 0.72)
    shown_lines = _wrap(d, shown, f, W * 0.72) if shown else []
    y = H / 2 - len(lines) * f.size * 0.65
    d.text((W * 0.12, y - 40), "“", font=_SERIF(220), anchor="lm", fill=K.YEL + (255,))
    for ln in shown_lines:
        d.text((W / 2, y), ln, font=f, anchor="ma", fill=(248, 242, 230, 255))
        y += int(f.size * 1.3)
    who = spec.get("who") if isinstance(spec, dict) else None
    if who and t > 0.2 + step * len(words):
        d.text((W / 2, y + 30), "— " + who, font=_HI(40), anchor="ma", fill=K.YEL + (255,))
    return _paste(frame, img, 0, 0, a)


def end_screen(last_shot):
    """The 20 s end screen (owner, 8 Nov 2026): the last painting dark and soft, the emblem, and two framed spaces
    where YouTube's end-screen elements sit (the next video on the right, subscribe on the left)."""
    bg = Image.open(last_shot).convert("RGB").resize((W, H)) if last_shot and Path(last_shot).exists() \
        else Image.new("RGB", (W, H), K.INK)
    bg = Image.blend(bg.filter(ImageFilter.GaussianBlur(14)), Image.new("RGB", (W, H), (8, 8, 10)), 0.72)
    d = ImageDraw.Draw(bg)
    lg = Image.open(K.asset("logo.png")).convert("RGBA").resize((230, 230), Image.LANCZOS)
    bg.paste(lg, (W // 2 - 115, 70), lg)
    d.text((W / 2, 350), "अगली कहानी भी उतनी ही दिलचस्प है", font=_HI(54), anchor="mm", fill=(248, 242, 230))
    gold = K.YEL
    vx, vy, vw, vh = 1010, 440, 760, 428  # YouTube's video element (16:9), right
    d.rounded_rectangle((vx - 8, vy - 8, vx + vw + 8, vy + vh + 8), 18, outline=gold, width=4)
    d.text((vx + vw / 2 + 20, vy + vh + 50), "अगली कहानी", font=_HI(40), anchor="mm", fill=gold)
    tx = vx + vw / 2 - d.textlength("अगली कहानी", font=_HI(40)) / 2 - 18
    d.polygon([(tx - 22, vy + vh + 36), (tx - 22, vy + vh + 64), (tx + 2, vy + vh + 50)], fill=gold)
    cx, cy, r = 520, 654, 150  # the subscribe element (round), left
    d.ellipse((cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8), outline=gold, width=4)
    d.text((cx, cy + r + 50), "Subscribe करें", font=_HI(40), anchor="mm", fill=gold)
    return bgr(K.grain(bg))


def tick(sec=0.035, seed=21):
    """A counter's click, made here."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    return (rng.standard_normal(n) * np.exp(-np.arange(n) / (0.004 * SR))).astype(np.float32)


def pop(sec=0.25):
    """A pin landing: a quick falling sine blip."""
    t = np.arange(int(sec * SR)) / SR
    return (np.sin(2 * np.pi * (900 * t - 1400 * t ** 2)) * np.exp(-t / 0.05)).astype(np.float32)


def paper(sec=0.6, seed=17):
    """A page flapping into place: crackly band-passed noise, a few sharp grains."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    x = rng.standard_normal(n) * (rng.random(n) < 0.08) * 3 + rng.standard_normal(n) * 0.4
    spec = np.fft.rfft(x)
    fq = np.fft.rfftfreq(n, 1 / SR)
    x = np.fft.irfft(spec * np.exp(-((fq - 3500) / 2500) ** 2), n)
    return (x * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7).astype(np.float32)


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


_WORLD = {}


MAP_TERRAIN = os.environ.get("MAPSHOT_TERRAIN", "0") == "1"  # 1 = the satellite base under MapShot; off: the drawn maps the owner preferred (13 Nov 2026)
_MAP_TER = []


def _map_grade(im):
    """Natural Earth's relief is pale: more colour and contrast on land, a deeper blue sea, so labels and fills read."""
    for y in range(0, im.shape[0], 540):
        f = im[y:y + 540].astype(np.float32) / 255
        r, b = f[..., 0], f[..., 2]  # RGB
        sea = np.clip((b - r - 0.05) / 0.10, 0, 1)[..., None]
        gray = f.mean(2, keepdims=True)
        land = np.clip(((gray + (f - gray) * 1.55) - 0.5) * 1.25 + 0.5 - 0.08, 0, 1) ** 1.05 * np.array([0.90, 0.89, 0.78], np.float32)
        deep = np.clip((gray + (f - gray) * 1.3) * np.array([0.38, 0.58, 0.78], np.float32), 0, 1)
        im[y:y + 540] = (np.clip(land * (1 - sea) + deep * sea, 0, 1) * 255 + 0.5).astype(np.uint8)


def _map_terrain_levels():
    if not _MAP_TER:
        im = cv2.cvtColor(cv2.imread(str(K.HERE / "maps" / "terrain_10800.jpg")), cv2.COLOR_BGR2RGB)
        _map_grade(im)
        _MAP_TER.append(im)
        for _ in range(3):
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            _MAP_TER.append(im)
    return _MAP_TER


_MAP_GR = []


def _map_grain():
    if not _MAP_GR:
        _MAP_GR.append(np.random.RandomState(7).randint(0, 7, (H, W, 1), np.uint8).repeat(3, 2))
    return _MAP_GR[0]


def map_terrain(lat0, lon0, span, cosl=None, cy=None):
    """The raster under the view (W x H, RGB): a pure affine sample, lon/lat -> pixel is linear in this projection."""
    lv = _map_terrain_levels()
    cosl = np.cos(np.radians(lat0)) if cosl is None else cosl
    k = H / span
    L = 0
    while L < len(lv) - 1 and 30.0 / 2 ** (L + 1) >= k * cosl * 0.9:
        L += 1
    ppd = 30.0 / 2 ** L
    a, d = ppd / (cosl * k), ppd / k
    M = np.float32([[a, 0, (lon0 + 180 - W / (2 * cosl * k)) * ppd - 0.5], [0, d, (90 - lat0 - (H / 2 if cy is None else cy) / k) * ppd - 0.5]])
    up = a < 1 or d < 1
    img = cv2.warpAffine(lv[L], M, (W, H), flags=(cv2.INTER_CUBIC if up else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if a < 0.5:  # several screen pixels per raster pixel: smooth the blockiness, add a fine grain so it is not mush
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        img = cv2.subtract(cv2.add(img, _map_grain()), 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class MapShot:
    """A real map instead of a painting (owner, 8 Nov 2026: Gazab Kyun's painted maps had "random red marks, lines";
    FLUX can't draw geography). Drawn from map data (brand/maps/world.json: Natural Earth, India's own boundary) at
    the frame's size, animated over the line: the camera zooms in, highlighted countries and areas fill in turn, lines
    draw themselves, pins drop, bold labels pop. spec (a shot's "map"):
      {"center": [lat, lon], "span": 8, "from_span": 60,               # degrees of height shown, end and start
       "countries": [{"iso": "EGY", "label": "मिस्र", "color": "red"}],  # ISO A3, colours: red yellow teal white
       "areas": [{"name": "Bir Tawil", "label": "बीर तवील", "color": "yellow"}],   # disputed areas by name
       "lines": [{"lat": 22, "label": "22° N"}, {"from": [lat, lon], "to": [lat, lon], "label": "..."}],
       "points": [{"lat": .., "lon": .., "label": "Point Nemo"}],
       "circles": [{"lat": .., "lon": .., "km": 2688, "label": "2,688 km"}]}
    Events (pins, fills) sit in the first 70% of the line, so the picture is complete while the line ends."""
    COLORS = {"red": (60, 70, 255), "yellow": (26, 194, 255), "teal": (170, 160, 40), "white": (240, 240, 240)}

    def __init__(self, spec, dur):
        if not _WORLD:
            _WORLD.update(json.loads((K.HERE / "maps" / "world.json").read_text()))
            for c in _WORLD["countries"] + _WORLD["disputed"]:
                a = np.concatenate([np.array(r) for r in c["rings"]])
                c["bbox"] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
        self.s, self.dur = spec, dur
        self.events = []  # (seconds into the shot, kind) for the sound
        n = 0
        for k in ("countries", "areas", "lines", "circles", "points"):
            for _ in spec.get(k) or []:
                self.events.append((min(0.7 * dur, 0.5 + 0.55 * n), k))
                n += 1

    def _proj(self, u):
        lat, lon = self.s["center"]
        e = ease(u / 0.45)
        span = self.s.get("from_span", self.s.get("span", 20) * 4) * (1 - e) + self.s.get("span", 20) * e
        k = H / span
        cosl = np.cos(np.radians(lat))
        return lambda lo, la: (W / 2 + (lo - lon) * cosl * k, H * 0.46 - (la - lat) * k), k, cosl

    def frame(self, u):
        P, k, cosl = self._proj(u)
        t = u * self.dur
        lat, lon = self.s["center"]
        half_lon, half_lat = W / (2 * k * cosl) + 2, H / k + 2
        view = (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        if MAP_TERRAIN:  # the real earth under the overlays (Natural Earth shaded relief, brand/maps/LICENSES.md)
            fr = map_terrain(lat, lon, H / k, cosl, H * 0.46)
        else:
            fr = np.zeros((H, W, 3), np.uint8)
            fr[:] = (52, 30, 14)  # deep sea
        vis = lambda b: not (b[2] < view[0] or b[0] > view[2] or b[3] < view[1] or b[1] > view[3])
        pts = lambda r: np.int32([[x * 4, y * 4] for x, y in (P(lo, la) for lo, la in r)])
        countries = [c for c in _WORLD["countries"] if vis(c["bbox"])]
        land = [pts(r) for c in countries for r in c["rings"]]
        if not MAP_TERRAIN:
            cv2.fillPoly(fr, land, (78, 58, 40), cv2.LINE_AA, shift=2)
        fill = fr.copy()
        labels = []
        n = 0
        for key, pool, by in (("countries", countries, "iso"), ("areas", _WORLD["disputed"], "name")):
            for it in self.s.get(key) or []:
                at = 0.5 + 0.55 * n
                n += 1
                a = min(1.0, max(0.0, (t - at) / 0.35))
                if a <= 0:
                    continue
                want = str(it.get("iso" if key == "countries" else "name", "")).lower()
                shape = next((c for c in pool if str(c.get(by, "")).lower() == want), None)
                if shape is None:
                    continue
                col = self.COLORS.get(it.get("color", "yellow"), self.COLORS["yellow"])
                layer = fr.copy()
                cv2.fillPoly(layer, [pts(r) for r in shape["rings"]], col, cv2.LINE_AA, shift=2)
                op = (0.92 if key == "areas" else 0.42) * a  # countries tinted, the area in question solid
                fr = cv2.addWeighted(layer, op, fr, 1 - op, 0)
                if key == "areas":  # a disputed area gets a bright outline, it is often tiny
                    cv2.polylines(fr, [pts(r) for r in shape["rings"]], True, (255, 255, 255), 3, cv2.LINE_AA, shift=2)
                if it.get("label"):  # at the middle of the part on screen
                    inside = [P(lo, la) for r in shape["rings"] for lo, la in r]
                    inside = [q for q in inside if 60 < q[0] < W - 60 and 80 < q[1] < H - 300] or inside
                    if it.get("at"):
                        pos = P(it["at"][1], it["at"][0])
                    elif key == "areas":
                        b = shape["bbox"]
                        pos = P((b[0] + b[2]) / 2, b[1])  # under the area: lines and coasts run above
                        pos = (pos[0], pos[1] + 62)
                    else:
                        pos = (float(np.median([q[0] for q in inside])), float(np.median([q[1] for q in inside])))
                    labels.append((pos, it["label"], a, 64 if key == "areas" else 58))
        if MAP_TERRAIN:  # thin white borders at ~45% over the fills
            lay = fr.copy()
            cv2.polylines(lay, land, True, (255, 255, 255), 1, cv2.LINE_AA, shift=2)
            fr = cv2.addWeighted(lay, 0.45, fr, 0.55, 0)
        else:
            cv2.polylines(fr, land, True, (140, 118, 96), 2, cv2.LINE_AA, shift=2)  # borders over the fills
        for it in self.s.get("lines") or []:
            at = 0.5 + 0.55 * n
            n += 1
            p = min(1.0, max(0.0, (t - at) / 0.8))
            if p <= 0:
                continue
            if "lat" in it:
                a0, a1 = P(view[0], it["lat"]), P(view[2], it["lat"])
            else:
                a0, a1 = P(it["from"][1], it["from"][0]), P(it["to"][1], it["to"][0])
            end = (a0[0] + (a1[0] - a0[0]) * p, a0[1] + (a1[1] - a0[1]) * p)
            cv2.line(fr, (int(a0[0]), int(a0[1])), (int(end[0]), int(end[1])), self.COLORS[it.get("color", "yellow")], 7, cv2.LINE_AA)
            if it.get("label") and p > 0.6:
                lx = W * 0.08 if "lat" in it else (a0[0] + a1[0]) / 2
                ly = a0[1] + 150 if "lat" in it else (a0[1] + a1[1]) / 2 - 50
                labels.append(((lx + (W * 0.12 if "lat" in it else 0), ly), it["label"], min(1.0, (p - 0.6) / 0.3), 52))
        for it in self.s.get("circles") or []:
            at = 0.5 + 0.55 * n
            n += 1
            p = min(1.0, max(0.0, (t - at) / 0.8))
            if p <= 0:
                continue
            c = P(it["lon"], it["lat"])
            r = it["km"] / 111.0 * k * p
            cv2.ellipse(fr, (int(c[0]), int(c[1])), (int(r), int(r)), 0, 0, 360 * p, (255, 255, 255), 4, cv2.LINE_AA)
            if it.get("label") and p > 0.8:
                labels.append(((c[0], c[1] - r - 40), it["label"], min(1.0, (p - 0.8) / 0.2), 56))
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        d = ImageDraw.Draw(img)
        for it in self.s.get("points") or []:
            at = 0.5 + 0.55 * n
            n += 1
            pt = t - at
            if pt <= 0:
                continue
            px, py = P(it["lon"], it["lat"])
            y = py - (1 - ease(pt / 0.3)) * 240
            for j in range(3):
                r_t = pt - 0.3 - j * 0.3
                if 0 < r_t < 1.2:
                    rr = 16 + 100 * r_t
                    d.ellipse((px - rr, py - rr * 0.5, px + rr, py + rr * 0.5), outline=(255, 255, 255, int(220 * (1 - r_t / 1.2))), width=4)
            red = (238, 40, 44, 255)
            d.polygon([(px - 28, y - 56), (px + 28, y - 56), (px, y)], fill=red)
            d.ellipse((px - 38, y - 106, px + 38, y - 30), fill=red)
            d.ellipse((px - 14, y - 82, px + 14, y - 54), fill=(255, 255, 255, 255))
            if it.get("label") and pt > 0.3:
                labels.append(((px, py - 150), it["label"], min(1.0, (pt - 0.3) / 0.3), 64))
        for (x, y), text, a, size in labels:
            f = _bf(text, size, "Bold") if not DEV.search(text) else _font(K.B["fonts"]["display"], size)
            d.text((x, y), text, font=f, anchor="mm", fill=(255, 255, 255, int(255 * a)), stroke_width=6, stroke_fill=(10, 14, 24, int(230 * a)))
        return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


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
        d.rectangle((W / 2 - tw / 2 - 24, y - 66, W / 2 + tw / 2 + 24, y + 62), fill=_Y() + (255,))
        d.text((W / 2, y), ln, font=f, anchor="mm", fill=K.INK + (255,))
        y += 132
    a = np.asarray(img)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0


def punch_word(frame, text, t, dur=1.3):
    """TEASER Shorts: the question's shocking word or number slams onto the screen (big, yellow on a red shadow,
    from 140% to 100% in 0.12 s), holds, and fades."""
    if t > dur:
        return frame
    a = min(1.0, (dur - t) / 0.25)
    sc = 1.4 - 0.4 * min(1.0, t / 0.12)
    f = K.display(int(150 * sc))
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    while d.textlength(text, font=f) > W * 0.9 and f.size > 60:
        f = K.display(f.size - 6)
    img = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((W / 2 + 8, 158), text, font=f, anchor="mm", fill=_R() + (235,))
    d.text((W / 2, 150), text, font=f, anchor="mm", fill=_Y() + (255,), stroke_width=8, stroke_fill=K.INK + (255,))
    return _paste(frame, img, 0, H * 0.27 - 150, a)


def episode_card(path):
    """TEASER Shorts' end: the full episode's own thumbnail (the owner's ChatGPT one) as a framed card with a
    "FULL EPISODE" ribbon, pre-rendered once (RGBA)."""
    th = Image.open(path).convert("RGB")
    cw = int(W * 0.86)
    ch = round(cw * 9 / 16)
    th = th.resize((cw, ch), Image.LANCZOS)
    pad, rib = 10, 70
    card = Image.new("RGBA", (cw + 2 * pad, ch + 2 * pad + rib), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, rib - 20, cw + 2 * pad, ch + 2 * pad + rib), 22, fill=(255, 255, 255, 255))
    card.paste(th, (pad, pad + rib))
    f = K.display(46)
    label = "पूरा EPISODE"  # the play triangle is drawn: the fonts have no ▶
    tw = int(d.textlength(label, font=f)) + 110
    d.rounded_rectangle((pad, 0, pad + tw, rib + 4), 16, fill=_R() + (255,))
    d.polygon([(pad + 28, rib / 2 - 16), (pad + 28, rib / 2 + 18), (pad + 58, rib / 2 + 1)], fill=(255, 255, 255, 255))
    d.text((pad + 76, rib / 2 + 2), label, font=f, anchor="lm", fill=(255, 255, 255, 255))
    return card


def end_card(frame, card, t):
    """The episode card drops in (scale 0.85 -> 1, 0.35 s) over the dimmed last picture; a curved arrow draws itself
    down to the Shorts player's related-video link (bottom left, above the title) and keeps nudging toward it."""
    a = min(1.0, t / 0.35)
    frame = _dim(frame, 0.55 * a)
    sc = 0.85 + 0.15 * (1 - (1 - a) ** 3)
    c = card.resize((int(card.width * sc), int(card.height * sc)), Image.LANCZOS)
    cy = int(H * 0.30)
    frame = _paste(frame, c, (W - c.width) / 2, cy - c.height / 2, a)
    # the arrow: from under the card's right side, curving down to the link at about (0.16 W, 0.80 H)
    k = min(1.0, max(0.0, (t - 0.35) / 0.45))
    if k > 0:
        bob = 14 * abs(np.sin(t * 5))
        p0 = np.array([W * 0.70, cy + card.height * 0.55])
        p1 = np.array([W * 0.72, H * 0.70])
        p2 = np.array([W * 0.20 - bob * 0.5, H * 0.79 + bob])
        pts = [(1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2 for u in np.linspace(0, k, 40)]
        pts = np.array(pts, np.int32)
        cv2.polylines(frame, [pts], False, (20, 20, 20), 46, cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, _Y()[::-1], 30, cv2.LINE_AA)
        if k >= 1.0:  # the head, pointing along the curve's end
            tip, back = pts[-1].astype(np.float32), pts[-6].astype(np.float32)
            dv = (tip - back) / (np.linalg.norm(tip - back) + 1e-6)
            nv = np.array([-dv[1], dv[0]])
            head = np.array([tip + dv * 40, tip - dv * 54 + nv * 60, tip - dv * 54 - nv * 60], np.int32)
            cv2.fillPoly(frame, [head], (20, 20, 20), cv2.LINE_AA)
            cv2.fillPoly(frame, [(head * 0.9 + tip * 0.1).astype(np.int32)], _Y()[::-1], cv2.LINE_AA)
            f = K.display(84)
            img = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((W * 0.40, 60), "यहाँ tap करें", font=f, anchor="lm", fill=_Y() + (255,),
                                     stroke_width=8, stroke_fill=(20, 20, 20, 255))
            frame = _paste(frame, img, 0, H * 0.66 - 60, min(1.0, (t - 0.8) / 0.25))
    return frame


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
    d.rounded_rectangle((x0 - 30, 14, x0 + tw + 140, bh - 14), 26, fill=_Y() + (255,))
    d.polygon([(x0 + 10, bh / 2 - 30), (x0 + 10, bh / 2 + 30), (x0 + 62, bh / 2)], fill=_R() + (255,))
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


class Footage:
    """A stock clip as the shot's picture (14 Nov 2026: the winners of THW, Gazab Kyun and Logos & Legends cut through real
    footage): read in order as the shot plays, cover-cropped to the frame, a slow 4% push, looped if short."""

    def __init__(self, path, dur):
        self.path, self.dur = str(path), dur
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.n = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self.start = max(0, int((self.n - dur * self.fps) / 3))  # a little in: clips often open on a fade
        self.idx, self.last = -1, None

    def frame(self, u):
        target = (self.start + int(u * self.dur * self.fps)) % self.n
        if target < self.idx:  # looped: from the top again
            self.cap.release()
            self.cap, self.idx = cv2.VideoCapture(self.path), -1
        while self.idx < target:
            ok, f = self.cap.read()
            self.idx += 1
            if not ok:
                break
            self.last = f
        f = self.last if self.last is not None else np.zeros((H, W, 3), np.uint8)
        h, w = f.shape[:2]
        z = 1.0 + 0.04 * u
        sc = max(W / w, H / h) * z
        r = cv2.resize(f, (int(w * sc) + 1, int(h * sc) + 1), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
        return r[y0:y0 + H, x0:x0 + W]


def clips_in(script, out):
    """The stock clips studio/real_pictures.py picked (Pixabay ids, by eye), downloaded as clips/NN.mp4."""
    import urllib.parse
    import urllib.request
    key = os.environ.get("PIXABAY_KEY", "")
    n = 0
    for sec in script.get("sections") or []:
        for s in sec.get("shots") or []:
            n += 1
            cid = (s.get("clip") or {}).get("id")
            dest = Path(out) / "clips" / f"{n:02d}.mp4"
            if not cid or dest.exists() or not key:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                u = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode({"key": key, "id": cid})
                h = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())["hits"][0]
                v = h["videos"].get("large") if (h["videos"].get("large") or {}).get("url") else h["videos"].get("medium")
                dest.write_bytes(urllib.request.urlopen(urllib.request.Request(v["url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read())
            except Exception as e:  # noqa: BLE001 - the shot falls back to a card
                print(f"  clip {n}: {e}", flush=True)


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
        # owner, 7 Nov 2026: the aesthetic long-form style; 8 Nov: DK Shorts small and white too ("short_captions")
        self.soft = K.B.get("short_captions") if short else K.B.get("long_captions")
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
        if not pill and self.soft.get("shade", True):  # LL (owner, 8 Nov 2026): "no background - just white, small text"
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
            if st.get("shadow", True):  # DK Shorts (owner, 8 Nov 2026): "without black stroke"
                ImageDraw.Draw(sh).text((W / 2, bh / 2 + 3), text, font=f, anchor="mm", fill=(0, 0, 0, 255),
                                        stroke_width=3, stroke_fill=(0, 0, 0, 255))
            img = sh.filter(ImageFilter.GaussianBlur(4))
            ImageDraw.Draw(img).text((W / 2, bh / 2), text, font=f, anchor="mm", fill=tuple(st.get("color", (250, 243, 230))) + (255,))
            a = np.asarray(img)
            self.cache[text] = (cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[..., 3:4] / 255.0)
        rgb, al = self.cache[text]
        # fade only at the edges of a run of groups: back-to-back groups swap without a dip
        after = any(abs(h[1] - g[0]) < 0.05 for h in self.groups)
        follows = any(abs(h[0] - g[1]) < 0.05 for h in self.groups)
        al = al * min(1.0, 1.0 if after else (t - g[0]) / 0.12, 1.0 if follows else (g[1] - t) / 0.12)
        y = int(H * self.soft.get("y", 0.885)) - rgb.shape[0] // 2  # Shorts sit higher, clear of YouTube's title
        roi = frame[y:y + rgb.shape[0]].astype(np.float32)
        frame[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
        return frame


_LT = {}
VIVID = False  # LL teaser Shorts (owner, 8 Nov 2026: "use vibrant colors... currently faded colors"): pure yellow/red
_Y = lambda: (255, 214, 0) if VIVID else K.YEL
_R = lambda: (255, 40, 40) if VIVID else K.RED
MOVES_CYCLE = ("pan_right", "pull", "pan_left", "push")
# a colour grade per episode (script "grade"; owner, 7 Nov 2026: Maruti 800 in Cross process), the same settings as the
# dashboard's filter previews (clipper/filters.json), applied while encoding
GRADES = {"cross-process": "eq=contrast=1.1:saturation=1.08,hue=h=-12",
          "cinematic": "eq=contrast=1.08:saturation=1.0,colorbalance=rs=0.05:bs=-0.05:rh=0.04:bh=-0.04",
          "warm": "colorbalance=rs=0.06:gs=0.02:bs=-0.06"}
WORDS = {}  # painting key ("12", "12-2") -> its words (shot "words" in the script, or the "paint_from" episode's)


def photos_in(script, out):
    """Real photographs (studio/real_pictures.py, 13 Nov 2026) become the shot's picture file before the edit: downloaded,
    composed to 16:9 (a photo far from it sits whole on a blurred, dimmed copy of itself) and saved as shot-NN.png, so
    every move, framing and subject crop works on them as on a painting. A failed download leaves the shot as it was."""
    import io
    import urllib.request
    n = 0
    for sec in script.get("sections") or []:
        for s in sec.get("shots") or []:
            n += 1
            ph = s.get("photo") or {}
            dest = Path(out) / f"shot-{n:02d}.png"
            if not ph.get("src") or dest.exists():
                continue
            try:
                data = urllib.request.urlopen(urllib.request.Request(
                    ph["src"], headers={"User-Agent": "EvolveBetterStudio/1.0 (decypherlabs.io@gmail.com)"}), timeout=60).read()
                im = Image.open(io.BytesIO(data)).convert("RGB")
                tw, th = 2400, 1350
                ar = im.width / im.height
                if 1.45 <= ar <= 2.1:
                    sc = max(tw / im.width, th / im.height)
                    im = im.resize((int(im.width * sc) + 1, int(im.height * sc) + 1), Image.LANCZOS)
                    x0, y0 = (im.width - tw) // 2, (im.height - th) // 2
                    canvas = im.crop((x0, y0, x0 + tw, y0 + th))
                else:
                    from PIL import ImageFilter, ImageEnhance
                    bg = im.resize((tw // 8, th // 8)).filter(ImageFilter.GaussianBlur(3)).resize((tw, th), Image.LANCZOS)
                    canvas = ImageEnhance.Brightness(bg).enhance(0.45)
                    sc = min(tw * 0.92 / im.width, th * 0.92 / im.height)
                    fg = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
                    canvas.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
                canvas.save(dest)
            except Exception as e:  # noqa: BLE001
                print(f"  photo {n}: {e}", flush=True)


def paint_read(f):
    """A painting from disk, toned to the brand (brand.json "tone": {"saturation": 0.85}; owner, 9 Nov 2026: Logos &
    Legends' teasers and long-forms "regularly have high saturation images... adapt as per the theme"). Only the
    paintings: captions and graphics keep their own colours."""
    im = cv2.imread(str(f))
    key = Path(str(f)).stem[len("shot-"):]
    if WORDS.get(key):  # the painting's words, set by the edit (pipeline/words.py, owner 11 Nov 2026)
        im = words.apply(im, WORDS[key])
    tone = K.B.get("tone") or {}
    sat, knee, con = float(tone.get("saturation") or 1), float(tone.get("knee") or 255), float(tone.get("contrast") or 1)
    if im is None or (abs(sat - 1) < 0.01 and knee >= 255 and abs(con - 1) < 0.01):
        return im
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] * sat
    # neon colours are pulled in harder than natural ones (owner, 9 Nov 2026: "not the high-contrast and saturation
    # images"): above the knee, saturation rises at half the rate
    hsv[..., 1] = np.clip(np.where(s > knee, knee + (s - knee) * 0.5, s), 0, 255)
    v = hsv[..., 2]
    hsv[..., 2] = np.clip(v.mean() + (v - v.mean()) * con, 0, 255)  # softer contrast around the picture's own level
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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
    f = _bf(shown, 110, "Bold")
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
        f1, f2 = _bf(main, 34, "Bold"), (_HI(32) if DEV.search(sub) else K.body(22))  # Hindi needs the size to read
        spaced = lambda t: t.strip() if DEV.search(t) else " ".join(t.strip().upper())  # letter-spacing (Latin only)
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
    src = Path(script_path).parent.parent / str(script.get("paint_from") or "-") / "script.json"
    WORDS.update(words.table(json.loads(src.read_text())) if src.is_file() else {})  # a Short on its episode's paintings
    WORDS.update(words.table(script))
    short = script.get("format") == "short"
    FX = not short and bool(K.B.get("motion_fx"))
    PANEL = short and K.B.get("short_layout") == "panel"  # Logos & Legends Shorts: the landscape painting whole (8 Nov 2026)
    TEASER = short and K.B.get("short_layout") == "teaser"  # LL Shorts: fast, punchy, question after question (8 Nov 2026)
    hits = []  # TEASER: (time, punch word) at every new question
    global VIVID
    VIVID = TEASER
    if short:
        W, H, XF = 1080, 1920, 0.15
    caps = []
    out = Path(out_dir)
    photos_in(script, out)  # real photographs into their shots' picture files (13 Nov 2026)
    clips_in(script, out)  # stock clips picked by eye (14 Nov 2026)
    vo = json.loads((out / "vo.json").read_text())
    items, n = [], 0  # (start, dur, visual, vo_file, section_id)
    cuts = []  # Shorts: every picture change (new shot or new crop), for the whoosh
    explainers = []  # time ranges of the science diagrams: the music sits a little higher under them
    t = 0.0
    chapters = []
    labels = []  # (t0, t1, text): lower-thirds
    titles = []  # (t0, t1, text): big phrases over the picture
    counters, years_seen, cards_at, last_counter = [], set(), [], -99.0  # FX: counting amounts, auto date labels
    gfx = []  # (t0, kind, payload): date callouts, stamps, keywords, typed cards, logos, merge graphics, story beats
    drops = []  # FX: music drop-outs before a twist
    mapfx = []  # (t, kind): the sounds of a map shot (zoom, fills, lines, pins)
    intro_at = None
    # "late_intro" (script; 12 Nov 2026 engine review: no benchmark shows a logo or a card in its first minute, ours spent
    # 0:44-0:52 on them): the cold open runs straight into chapter 1 with no card, and the title sequence plays before
    # chapter 2 in place of that chapter's card. The chapter list for YouTube is unchanged. Scripts without it build as before.
    late = bool(script.get("late_intro")) and not short
    carded = 0  # chapters with a card so far
    for si, sec in enumerate(script["sections"]):
        intro_now = False
        if (FX and K.B.get("intro") == "title" and intro_at is None and si > 0 and sec.get("card")
                and (not late or carded >= 1)):
            # the show's title sequence between the cold open and the first chapter
            name = script.get("intro_title") or script.get("title", "").split("|")[0].strip()
            name = re.sub(r"[^\w\s&'.:-]", "", name).strip()
            intro_at = t
            items.append((t, 4.0, Intro(name), None, "intro"))
            t += 4.0
            intro_now = True
        if sec.get("sting"):
            items.append((t, 2.2, Seq(sting_frames(int(2.2 * FPS))), None, sec["id"]))
            t += 2.2
            continue
        if sec.get("card"):
            carded += 1
            chapters.append((t, sec["card"].title()))
            if late and (carded == 1 or intro_now):
                pass  # chapter 1 follows the cold open with no card; the title sequence stands in for chapter 2's
            elif K.B.get("chapter_style") == "numbered" and not short:  # the reference's navy numbered card
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
            if s.get("clip") and not short:  # a stock clip picked for this line; a failed download shows the line as a card
                cp = out / "clips" / f"{n:02d}.mp4"
                vis = Footage(cp, d) if cp.exists() else Still(card(s.get("title") or " ".join(s["vo"].split()[:5])))
            elif s.get("diagram") and s["diagram"] not in DIAGRAMS:  # a diagram this builder doesn't have (old scripts,
                # 14 Nov 2026: ep 13's "gut_two_doors" failed the build): its line shown as a card instead
                vis = Still(card(s.get("title") or " ".join(s["vo"].split()[:5])))
            elif s.get("diagram"):
                vis = Shot(bgr(DIAGRAMS[s["diagram"]]()), "push")
                explainers.append((t, t + d))
            elif s.get("map"):  # a real map, never a painted one (Gazab Kyun, owner 8 Nov 2026)
                mp = MapShot(s["map"], d)
                mapfx.append((t, "zoom"))
                mapfx += [(t + e, k) for e, k in mp.events]
                vis = [mp] if short and TEASER else mp
            else:
                src = s.get("from")  # an episode's own painting (Logos & Legends Shorts from a long-form, 8 Nov 2026)
                img = paint_read(str(out / (f"shot-{int(src.split('-')[0]):02d}{'-' + src.split('-')[1] if '-' in src else ''}.png"
                                            if src else f"shot-{n:02d}.png")))
                if s.get("overlay"):  # science drawn onto the painting (a crack, a hole) where SDXL can't
                    img = bgr(OVERLAYS[s["overlay"]](Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                parts = 3 if d > 5.4 else 2 if d > 2.6 else 1
                if short and TEASER:  # the line's painting + 1-2 more, full frame, cut every ~1.5 s
                    # its own vertical FLUX paintings (owner, 8 Nov 2026: "paint the shorts' shots on FLUX"): the line's
                    # picture + its "more" pictures; or, reusing an episode's, the "from" + "also" ones
                    ids = [src] + list(s.get("also") or []) if src else [f"{n:02d}"] + [
                        x.stem[len("shot-"):] for x in sorted(out.glob(f"shot-{n:02d}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))]
                    if s.get("montage"):  # the last line: a flash recap of the Short's pictures under the call
                        ids = s["montage"]
                    pics = []
                    for q in ids:
                        f = out / f"shot-{int(q.split('-')[0]):02d}{'-' + q.split('-')[1] if '-' in q else ''}.png"
                        im = paint_read(f)
                        if im is None:
                            continue
                        im = cv2.addWeighted(im, 1.6, cv2.GaussianBlur(im, (0, 0), 1.6), -0.6, 0)  # crisper upscale
                        fx, fy, faces = subject(im)
                        pics.append(Shot(im, ("push", "pull")[len(pics) % 2], faces or (fx, fy)))
                    vis, parts = (pics or [Shot(img, "push")]), 1
                elif short and PANEL:  # a landscape painting kept whole in a vertical frame
                    vis, parts = Panel(img, MOVES_CYCLE[n % 4]), 1
                else:
                    # "late_intro" scripts (12 Nov 2026): the cold open's pictures change twice as often, each painting then
                    # a tighter crop on its subject, like the Shorts (the niche's winners show 16-26 pictures in the first
                    # minute, ours 8-12); a line that has its own extra paintings keeps those instead
                    fast = late and si == 0 and not any(out.glob(f"shot-{n:02d}-*.png"))
                    vis = Split(img, parts) if (short or fast) and parts > 1 else Shot(img, s.get("move", "push"))
                if short:
                    cuts += [t + d * i / parts for i in range(1, parts)]
            extras = [] if short or s.get("diagram") or s.get("map") else sorted(
                out.glob(f"shot-{n:02d}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
            if isinstance(vis, list):  # TEASER: the line's pictures in turn, the voice on the first
                hits.append((t, s.get("key") if hits else None))  # line 1 has the hook headline
                part = d / len(vis)
                for i, v in enumerate(vis):
                    items.append((t + i * part, part, v, out / f"vo-{n:02d}.wav" if i == 0 else None, sec["id"]))
                    if i:
                        cuts.append(t + i * part)
            elif extras:  # a new picture every ~4 s (owner, 7 Nov 2026, after The Quiet Story): the line's pictures in turn
                pics = [vis] + [Shot(paint_read(x), MOVES_CYCLE[(n + i) % 4]) for i, x in enumerate(extras)]
                part = d / len(pics)
                for i, v in enumerate(pics):
                    items.append((t + i * part, part, v, out / f"vo-{n:02d}.wav" if i == 0 else None, sec["id"]))
            else:
                items.append((t, d, vis, out / f"vo-{n:02d}.wav", sec["id"]))
            if s.get("title") and not short:  # a big phrase over the picture
                titles.append((t + 0.2, t + min(d, 3.2), s["title"]))
            if s.get("label") and not short:  # a lower-third (Logos & Legends, 7 Nov 2026: names, dates, places)
                lab_txt = s["label"]
                if isinstance(lab_txt, dict):  # {"name": ..., "role": ...} -> "NAME · role"
                    lab_txt = " · ".join(str(v) for v in (lab_txt.get("name"), lab_txt.get("role")) if v)
                labels.append((t + 0.3, t + min(d, 4.3), lab_txt))
                years_seen.update(YEAR.findall(lab_txt))
            line = s.get("caption") or s["vo"]
            if FX and s.get("date"):
                gfx.append((t + 0.2, "date", s["date"]))
                years_seen.update(YEAR.findall(s["date"]))
            elif FX and not s.get("label"):  # every first mention of a year: a big date callout (or a lower-third)
                y = next((y for y in YEAR.findall(line) if y not in years_seen), None)
                if y:
                    years_seen.add(y)
                    if K.B.get("date_style") in ("callout", "corner"):
                        gfx.append((t + 0.2, "date", y))
                    else:
                        labels.append((t + 0.3, t + min(d, 3.6), y))
            if FX:
                for kind in ("stamp", "keyword", "typecard", "logo", "merge", "news", "place", "split", "quote"):
                    if s.get(kind):
                        gfx.append((t + (0.6 if kind == "stamp" else 0.25), kind, s[kind]))
                if s.get("drop"):  # the music cuts out just before this line (a twist), a hit as it lands
                    drops.append(t)
            if FX and t - last_counter > 12:  # an amount or a percentage counts up as it's said
                m = AMOUNT.search(line)
                if m and not YEAR.fullmatch(m.group().strip()):
                    counters.append((t + 0.2, m.group().strip()))
                    last_counter = t
            if short:  # the very first words show at 0.0 s, not after a pause
                caps += Captions.for_shot(s.get("caption") or s["vo"], t + (0.0 if t == 0 else 0.15), vo[f"{n:02d}"],
                                          **({"size": 6} if K.B.get("short_captions") else {}))  # small subtitles: longer phrases
            elif not s.get("diagram") and not (late and t < 60):  # diagrams carry their own labels; no running captions
                # in a "late_intro" episode's first minute (owner, 13 Nov 2026: big titles there, as the winners do)
                caps += Captions.for_shot(s.get("caption") or s["vo"], t + 0.15, vo[f"{n:02d}"], size=7)
            t += d
    if short:  # no end screen: Shorts loop back to the start
        total = t
    elif K.B.get("end_screen_s"):  # a full end screen with spaces for YouTube's elements (LL, 8 Nov 2026)
        es = float(script.get("end_s") or K.B["end_screen_s"])
        items.append((t, es, Shot(end_screen(out / f"shot-{n:02d}.png"), "push"), None, "outro"))
        total = t + es
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
    if script.get("music"):  # every section has music (11 Nov 2026: a rewrite renamed "ch1" to "chapter_1" and
        # Lijjat's and Ambassador's chapters played silent): an unplanned id takes its number's mood, else reflective
        num = lambda x: "".join(c for c in str(x) if c.isdigit())
        by_num = {num(k): v for k, v in mood_of.items() if num(k)}
        for sid in dict.fromkeys(x[4] for x in items):
            if sid and sid != "intro" and sid not in mood_of:  # the title sequence has its own riser and impact
                mood_of[sid] = by_num.get(num(sid)) or "reflective"
                print(f"  music: section {sid} wasn't in the plan, plays {mood_of[sid]}", flush=True)
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
    ml = {} if short else (K.B.get("music_long") or {})  # per channel, from its niche's winners (analysis/voice-review.md, 13 Nov 2026)
    talk_db, pause_db, read_db = ml.get("talk", talk_db), ml.get("pause", pause_db), ml.get("read", read_db)
    under = np.full(len(gain), 10 ** (talk_db / 20))
    for a, b in explainers:  # under the explainers: a clear, curious bed while the viewer reads
        under[int(a * SR):int(b * SR)] = 10 ** (read_db / 20)
    under = np.convolve(under, np.ones(SR // 2) / (SR // 2), mode="same")  # ease in/out over 0.5 s
    level = voice_rms * (under * gain + 10 ** (pause_db / 20) * (1 - gain))
    fade = np.minimum(1, np.arange(len(music)) / ((0.05 if short else 1.5) * SR)) * \
        np.minimum(1, (len(music) - np.arange(len(music))) / (3 * SR))
    for d0 in drops:  # silence ~1 s before the twist, the music back over 1.5 s after it
        a, b = int(max(0, d0 - 1.2) * SR), min(len(fade), int(d0 * SR))
        fade[a:b] *= np.clip(1 - np.arange(b - a) / (0.3 * SR), 0, 1)
        c = min(len(fade), b + int(1.5 * SR))
        fade[b:c] *= np.linspace(0, 1, c - b)
    if ml.get("mode") == "sparse":  # no bed under the narration (The Hidden Why's winners): only the opening and a
        # few seconds into each chapter, faded
        keep = np.zeros(len(fade))
        for a0, b0 in [(0.0, float(ml.get("opening", 20)))] + [(c, c + float(ml.get("chapter", 5))) for c, _ in chapters if c > 0]:
            a1, b1 = int(a0 * SR), min(len(keep), int(b0 * SR))
            keep[a1:b1] = 1
        keep = np.convolve(keep, np.ones(SR) / SR, mode="same")  # 1 s fades
        fade = fade * keep
    mix = voice + music * level * fade
    if short:
        sfx = np.zeros_like(mix)

        def put(x, at, db):
            i = int(at * SR)
            if 0 <= i < len(sfx):
                sfx[i:i + len(x)] += x[:len(sfx) - i] * voice_rms * 10 ** (db / 20) / rms(x)

        style = K.B.get("sfx") or {}
        last = items[-1][0] if items else 0
        page = TEASER and K.asset("sfx/page.wav").exists()
        # the brand's chapter-change page sound once per question, not on every picture (owner, 8 Nov 2026:
        # "reduce the number of page turn sfx"); other Shorts keep a whoosh on every cut
        at = [h[0] for h in hits[1:]] if page else cuts + [it[0] for it in items[1:]]
        gap = float(style.get("min_gap") or 0)
        if gap:  # brand.json sfx.min_gap (owner, 9 Nov 2026, Gazab Kyun: "the sfx on every shot change feels a bit weird...
            # 5-6 times in 30-40 seconds"): one whoosh + hit on a new line, none closer than `gap` seconds
            keep = []
            for t in sorted(h[0] for h in hits[1:]):
                if t < last - 0.5 and (not keep or t - keep[-1] >= gap):
                    keep.append(t)
            at = keep
        for c in sorted(set(round(x, 3) for x in at)):
            if abs(c - last) > 0.05:
                if page:
                    put(load_sfx("page"), max(0, c - 0.05), -13)
                else:
                    put(whoosh(), max(0, c - 0.18), -15)
        for m0, kind in mapfx:  # a whoosh as the map zooms, a tick per fill or line, a pop per pin
            if kind == "zoom":
                put(whoosh(0.5, seed=37), m0, -17)
            elif kind == "points":
                put(pop(), m0 + 0.3, -13)
            else:
                put(tick(0.05, seed=43), m0, -18)
        if TEASER:
            for h in hits[1:-1]:
                if not gap or round(h[0], 3) in {round(x, 3) for x in at}:  # thinned with the whooshes
                    put(load_sfx("impact"), h[0], -14)
            if len(hits) > 1:
                put(riser(1.2), max(0, hits[-1][0] - 1.2), -16)
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
            elif kind == "news":  # the page spins in, flaps down, a low hit
                put(whoosh(0.4, seed=19), g0, -20)
                put(paper(), g0 + 0.3, -16)
                put(load_sfx("impact"), g0 + 0.42, -15)
            elif kind == "place":
                put(whoosh(0.3, seed=23), g0, -24)
                put(pop(), g0 + 0.33, -14)
            elif kind == "split":
                put(whoosh(0.4, seed=29), g0, -18)
                put(load_sfx("typehit") if hit else tick(), g0 + 0.45, -18)
            elif kind == "quote" and K.asset("sfx/typing.wav").exists():
                put(load_sfx("typing")[:int(SR * min(3.0, 0.16 * len(str(pay).split()) + 0.3))], g0 + 0.2, -26)
        for d0 in drops:
            put(load_sfx("impact"), d0, -11)
        for t0, _, _ in labels:  # a soft tick as a name card slides in
            put(tick(0.05, seed=31), t0, -24)
        if intro_at is not None:
            put(riser(1.8), max(0, intro_at - 1.6), -17)
            put(load_sfx("impact"), intro_at + 0.15, -12)
        for t0, _, _ in titles:
            put(load_sfx("impact"), t0, -18)
        for t0, _ in counters:
            put(whoosh(0.25, seed=11), t0, -26)
            for i in range(14):  # the digits tick as they count up, slowing down, then a hit
                put(tick(seed=40 + i), t0 + 1.1 * (1 - (1 - i / 14) ** 0.5), -27)
            put(load_sfx("impact"), t0 + 1.15, -20)
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
    ep_thumb = Path(script_path).parent.parent / (script.get("full_episode") or "-") / "thumb-custom.jpg"
    ecard = episode_card(ep_thumb) if TEASER and ep_thumb.exists() else None
    end_at = (hits[-1][0] + 0.6) if TEASER and hits else total
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
                if gt < 0 or gt > {"date": 4.0 if K.B.get("date_style") == "corner" else 2.0, "stamp": 2.4, "keyword": 2.6, "typecard": 3.4, "logo": 2.8, "merge": 4.2,
                                   "news": 3.8, "place": 3.0, "split": 4.0, "quote": 4.0}[kind]:
                    continue
                corner = kind == "date" and K.B.get("date_style") == "corner"
                if kind in ("date", "stamp", "news", "place", "split") and not corner:
                    fr = punch(fr, gt)
                if corner:
                    fr = corner_label(fr.copy(), pay, gt)
                    continue
                if kind == "news":
                    fr = newspaper(fr.copy(), pay, gt)
                elif kind == "place":
                    fr = place_pin(fr.copy(), pay, gt)
                elif kind == "split":
                    fr = split_compare(fr.copy(), pay, out, gt)
                elif kind == "quote":
                    fr = quote_card(fr.copy(), pay, gt)
                elif kind == "date":
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
            if captions and not (ecard is not None and k / FPS >= end_at):  # the end card has the stage
                fr = captions.draw(fr.copy(), k / FPS)
            if hook is not None and (PANEL or k / FPS < HOOK_S):  # the question headline, fading out over its last 0.3 s
                rgb, al = hook                                     # (the panel layout keeps it above the picture)
                al = al if PANEL else al * min(1.0, (HOOK_S - k / FPS) / 0.3)
                y = int(H * Panel.TOP) - 540 if PANEL else int(H * 0.12)
                roi = fr[y:y + 520].astype(np.float32)
                fr[y:y + 520] = (rgb * al + roi * (1 - al)).astype(np.uint8)
            if TEASER:
                tt = k / FPS
                h0 = next((h for h in hits if 0 <= tt - h[0] < 1.3), None)
                if h0 is not None:
                    a = tt - h0[0]
                    if a < 0.25:  # a camera jolt as the question lands
                        amp = 18 * (1 - a / 0.25)
                        M = np.float32([[1, 0, amp * np.sin(a * 90)], [0, 1, amp * np.cos(a * 70)]])
                        fr = cv2.warpAffine(fr, M, (W, H), borderMode=cv2.BORDER_REFLECT)
                    if a < 0.1:  # a white flash
                        fr = cv2.addWeighted(fr, 1 - 0.55 * (1 - a / 0.1), np.full_like(fr, 255), 0.55 * (1 - a / 0.1), 0)
                    if h0[1]:
                        fr = punch_word(fr, h0[1], a)
                cv2.rectangle(fr, (0, 0), (int(W * min(1.0, tt / total)), 12), _Y()[::-1], -1)  # watch-to-the-end bar
            if ecard is not None and k / FPS >= end_at:  # the full episode's thumbnail + the arrow to its link
                fr = end_card(fr, ecard, k / FPS - end_at)
            elif cta is not None and k / FPS > total - CTA_S:  # "the answers: the full episode" (Logos & Legends Shorts)
                rgb, al = cta
                al = al * min(1.0, (k / FPS - (total - CTA_S)) / 0.3)
                y = int(H * 0.86) - rgb.shape[0] // 2
                roi = fr[y:y + rgb.shape[0]].astype(np.float32)
                fr[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
                if TEASER:  # a bouncing arrow down to the related-video link
                    by = int(H * 0.86) + 95 + int(18 * abs(np.sin(k / FPS * 6)))
                    pts = np.array([[W // 2 - 46, by], [W // 2 + 46, by], [W // 2, by + 56]], np.int32)
                    cv2.fillPoly(fr, [pts], _Y()[::-1])
            if mark is not None:  # the banner's name + tagline at the top, above the hook (it starts at 12%)
                rgb, al = mark
                y = 40
                roi = fr[y:y + rgb.shape[0]].astype(np.float32)
                fr[y:y + rgb.shape[0]] = (rgb * al + roi * (1 - al)).astype(np.uint8)
            ff.stdin.write(fr.tobytes())
            k += 1
    ff.stdin.close()
    ff.wait()
    if any(kind == "place" for _, kind, _ in gfx):  # the map's licence asks for credit (build.yml adds it to the description)
        (out / "credits.txt").write_text("Map: India boundaries by DataMeet India community (CC BY 4.0), github.com/datameet/maps")
    (out / "chapters.txt").write_text("" if short else"\n".join(f"{int(s // 60):02d}:{int(s % 60):02d} {name}" for s, name in chapters))
    print(f"video {total / 60:.1f} min, {k} frames -> {out / 'video.mp4'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
