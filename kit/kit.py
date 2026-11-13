"""The edit-bible element kit (studio repo docs/channels/edit-bible.md, sections 3-4) for Point of Ruin and The State
Maps: plain numpy/PIL/OpenCV + ffmpeg, 1920x1080 at 30 fps. Copied from the studio's experiments/sample-edit (owner,
12 Nov 2026 sample openings) and used by kit/build.py on GitHub Actions (kit.yml)."""
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
W, H, FPS, SR = 1920, 1080, 30, 48000
FONTS = HERE / "fonts"
SFX = REPO / "brand" / "logos-and-legends" / "sfx"  # registered in clipper/music/licenses.json
MUSIC = REPO / "brand" / "music"


# ---------- basics ----------
def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def back_out(t):
    t = min(1.0, max(0.0, t))
    return 1 + 2.7 * (t - 1) ** 3 + 1.7 * (t - 1) ** 2


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


_font_cache = {}


def font(name, size, var=None):
    """name: file stem in fonts/; var: a variation name ('Black', 'Bold', 'Italic'...) for variable fonts."""
    key = (name, size, var)
    if key not in _font_cache:
        f = ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)
        if var:
            try:
                f.set_variation_by_name(var)
            except (OSError, ValueError):
                pass
        _font_cache[key] = f
    return _font_cache[key]


def spaced(d, xy, text, f, fill, track=0.08, anchor_right=False):
    """Letter-spaced text; returns its width."""
    width = sum(d.textlength(c, font=f) + f.size * track for c in text) - f.size * track
    x, y = xy
    if anchor_right:
        x -= width
    for c in text:
        d.text((x, y), c, font=f, fill=fill)
        x += d.textlength(c, font=f) + f.size * track
    return width


# ---------- footage: "the pass" (3.1) ----------
class Clip:
    """Frames of a stock clip, cover-scaled with headroom for the push, streamed one at a time from ffmpeg (a whole
    clip at this size would be several GB: the Mac has 8)."""
    PAD = 1.06

    def __init__(self, path, start=0.0, sec=10.0):
        self.path, self.start, self.sec = path, start, sec
        self.w, self.h = round(W * self.PAD) // 2 * 2, round(H * self.PAD) // 2 * 2
        self.p, self.k, self.cur = None, -1, None

    def _read(self, i):
        if self.p is None or i < self.k:
            if self.p:
                self.p.kill()
            cmd = ["ffmpeg", "-v", "error", "-ss", str(self.start), "-i", str(self.path), "-t", str(self.sec + 0.5),
                   "-an", "-vf", f"fps={FPS},scale={self.w}:{self.h}:force_original_aspect_ratio=increase,crop={self.w}:{self.h}",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            self.p, self.k = subprocess.Popen(cmd, stdout=subprocess.PIPE), -1
        size = self.w * self.h * 3
        while self.k < i:
            raw = self.p.stdout.read(size)
            if len(raw) < size:  # the clip ran out: hold its last frame
                break
            self.cur = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            self.k += 1
        return self.cur

    def frame(self, i, n, push=0.04, drift=1):
        """Frame i of a shot n frames long: a slow push (and a lateral drift, alternating per shot)."""
        src = self._read(i)
        u = i / max(1, n - 1)
        z = 1 + push * u
        cw, ch = W / z, H / z
        cx = self.w / 2 + drift * (self.w - W) * 0.35 * (u - 0.5)
        cy = self.h / 2
        x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
        crop = src[y0:y0 + int(ch), x0:x0 + int(cw)]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


def grade(rgb, sat=0.8, tint="#5B6770", dim=1.0):
    """Lift blacks to 8, highlights to 235, saturation, a 10% shadow tint toward the channel base, gentle S."""
    f = rgb.astype(np.float32)
    f = 8 + f * (227 / 255)
    g = (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114)[..., None]
    f = g + (f - g) * sat
    w = ((1 - g / 255) ** 2) * 0.10
    f = f * (1 - w) + np.array(hexrgb(tint), np.float32) * w
    f = f + 0.06 * (f - 128) * (1 - np.abs(f - 128) / 128)
    return np.clip(f * dim, 0, 255).astype(np.uint8)


_rng = np.random.default_rng(7)
GRAIN = [_rng.normal(0, 6, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.10 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2)[..., None].astype(np.float32)
del yy, xx


def finish(rgb, i, strength=1.0):
    """Per-frame grain (a new plate every frame) and the 10% vignette: one texture for maps, footage and cards."""
    g = GRAIN[i % 6]
    g = np.roll(np.roll(g, (i * 37) % (H // 2), 0), (i * 91) % (W // 2), 1)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * strength
    return np.clip(rgb.astype(np.float32) * VIG + g, 0, 255).astype(np.uint8)


def over(rgb, layer):
    """Alpha-composite an RGBA PIL layer over an RGB array."""
    a = np.asarray(layer, np.float32)
    al = a[..., 3:4] / 255
    return (rgb.astype(np.float32) * (1 - al) + a[..., :3] * al).astype(np.uint8)


def blank():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


# ---------- overlays ----------
def source_tag(d, text, alpha=1.0, bone=(216, 210, 196)):
    """3.1: bottom-left, mono 16 px caps, 60% bone."""
    d.text((28, H - 46), text.upper(), font=font("IBMPlexMono-Regular", 16), fill=(*bone, int(150 * alpha)))


def watermark(d, text, alpha=1.0, bone=(216, 210, 196)):
    """3.12: the wordmark bottom-right at 35%."""
    spaced(d, (W - 28, H - 48), text.upper(), font("Inter", 18), (*bone, int(90 * alpha)), track=0.2, anchor_right=True)


def label_plate(layer, xy, text, t, plate=(0, 0, 0), plate_a=0.85, size=28, caps=True, accent=None, target=None):
    """3.2: plate scales 0.9->1 and fades in over 8 f, text from f 4; optional leader line + dot to a target.
    Returns the plate's resting box (x0, y0, x1, y1), so an underline or arrow can sit exactly on it."""
    if t <= 0:
        return None
    d = ImageDraw.Draw(layer)
    f = font("Inter", size)
    text = text.upper() if caps else text
    tw = sum(d.textlength(c, font=f) + size * 0.08 for c in text)
    pw, ph = tw + 20, size + 16
    a = ease_out(t / (8 / FPS))
    s = 0.9 + 0.1 * a
    x, y = xy
    if target is not None:
        p = ease_out((t - 0.1) / (10 / FPS))
        if p > 0:
            ex, ey = (x + pw / 2, y + ph) if target[1] > y else (x + pw / 2, y)
            tx, ty = target
            d.line([(tx, ty), (tx + (ex - tx) * p, ty + (ey - ty) * p)], fill=(216, 210, 196, int(255 * a)), width=2)
            d.ellipse((tx - 6, ty - 6, tx + 6, ty + 6), fill=(216, 210, 196, int(255 * a)))
    cx, cy = x + pw / 2, y + ph / 2
    box = (cx - pw * s / 2, cy - ph * s / 2, cx + pw * s / 2, cy + ph * s / 2)
    d.rounded_rectangle(box, radius=2, fill=(*plate, int(255 * plate_a * a)),
                        outline=(*accent, int(90 * a)) if accent else None)
    if t > 4 / FPS:
        ta = ease_out((t - 4 / FPS) / (6 / FPS))
        spaced(d, (x + 10, y + 6), text, f, (216, 210, 196, int(255 * ta)))
    return (x, y, x + pw, y + ph)


def underline(layer, box, t, color=(226, 58, 46), width=5, gap=8):
    """3.3: a red underline the plate's width, 8 f left to right, with a slight hand jitter (+/-1 px)."""
    if t <= 0 or not box:
        return
    d = ImageDraw.Draw(layer)
    x0, _, x1, y1 = box
    p = ease_out(t / (8 / FPS))
    xe = x0 + (x1 - x0) * p
    jit = np.random.default_rng(int(x0 + y1)).uniform(-1, 1, 24)
    pts = [(x0 + (xe - x0) * k / 23, y1 + gap + jit[k]) for k in range(24)]
    d.line(pts, fill=(*color, 255), width=width, joint="curve")


def ring(layer, c, r, t, color=(226, 58, 46), width=4):
    """3.3: arc from 12 o'clock over 14 f, then one pulse (+6% radius, alpha 0.6) out and back."""
    if t <= 0:
        return
    d = ImageDraw.Draw(layer)
    p = ease_out(t / (14 / FPS))
    pulse = max(0.0, t - 14 / FPS) / (12 / FPS)
    k = 0.06 * (math.sin(min(pulse, 2.0) * math.pi / 2) if pulse < 2 else 0)
    rr = r * (1 + k)
    alpha = int(255 * (1 - 0.4 * (k / 0.06 if k else 0)))
    d.arc((c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr), -90, -90 + 360 * p, fill=(*color, alpha), width=width)


def arrow(layer, a, b, t, color=(226, 58, 46), width=6, head=22):
    """3.3: the shaft draws over 10 f, the head over the last 3 f, with a dark halo."""
    if t <= 0:
        return
    d = ImageDraw.Draw(layer)
    p = ease_out(t / (10 / FPS))
    ex, ey = a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p
    d.line([a, (ex, ey)], fill=(0, 0, 0, 128), width=width + 4)
    d.line([a, (ex, ey)], fill=(*color, 255), width=width)
    if p > 0.7:
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        hh = head * min(1.0, (p - 0.7) / 0.3)
        pts = [(ex, ey), (ex - hh * math.cos(ang - 0.45), ey - hh * math.sin(ang - 0.45)),
               (ex - hh * math.cos(ang + 0.45), ey - hh * math.sin(ang + 0.45))]
        d.polygon(pts, fill=(*color, 255))


def sketch_arrow(layer, a, ctrl, b, t, color=(226, 58, 46), width=7, seed=5, dur=14):
    """A hand-drawn marker arrow (owner, 12 Nov 2026: "feel free to have the arrow images as well"): a curved stroke
    (quadratic Bezier) with marker-like width wobble and a little hand jitter, drawn along its path over `dur`
    frames; the two head strokes flick in over the last 4 frames. A soft dark halo keeps it readable on footage."""
    if t <= 0:
        return
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(layer)
    n = 60
    u = np.linspace(0, 1, n)
    pts = [((1 - s) ** 2 * a[0] + 2 * (1 - s) * s * ctrl[0] + s ** 2 * b[0],
            (1 - s) ** 2 * a[1] + 2 * (1 - s) * s * ctrl[1] + s ** 2 * b[1]) for s in u]
    jit = np.cumsum(rng.normal(0, 0.35, (n, 2)), 0)
    jit -= np.linspace(jit[0], jit[-1], n)  # wobble in the middle, ends on target
    pts = [(x + jx, y + jy) for (x, y), (jx, jy) in zip(pts, jit)]
    wv = width * (0.85 + 0.3 * np.abs(np.sin(u * 7 + rng.uniform(0, 3))))
    p = ease_out(t / (dur / FPS))
    m = max(2, int(n * p))
    for halo, col in ((4, (0, 0, 0, 110)), (0, (*color, 255))):
        for k in range(m - 1):
            w = wv[k] + halo
            d.line([pts[k], pts[k + 1]], fill=col, width=int(w))
            d.ellipse((pts[k][0] - w / 2, pts[k][1] - w / 2, pts[k][0] + w / 2, pts[k][1] + w / 2), fill=col)
    hp = ease_out((t - (dur - 4) / FPS) / (4 / FPS))
    if hp > 0:
        ex, ey = pts[-1]
        ang = math.atan2(ey - pts[-6][1], ex - pts[-6][0])
        L = 34 * hp
        for side in (-0.5, 0.45):  # slightly uneven, like a hand
            hx, hy = ex - L * math.cos(ang + side), ey - L * math.sin(ang + side)
            for halo, col in ((4, (0, 0, 0, 110)), (0, (*color, 255))):
                d.line([(ex, ey), (hx, hy)], fill=col, width=width + halo)
                d.ellipse((hx - (width + halo) / 2, hy - (width + halo) / 2, hx + (width + halo) / 2, hy + (width + halo) / 2), fill=col)


_arrow_cache = {}


def image_arrow(layer, path, center, height, t, angle=0, color=(226, 58, 46), wipe="down", dur=12, flip=False):
    """An arrow from real artwork (owner, 12 Nov 2026: "creating it by self makes it weird"): a licensed PNG
    (assets/arrows, Pixabay, logged), recoloured to the brand red with its own shading kept, revealed by a wipe in
    its direction over `dur` frames with a 4% settle."""
    if t <= 0:
        return
    key = (str(path), height, angle, color, flip)
    if key not in _arrow_cache:
        im = Image.open(path).convert("RGBA")
        bb = im.getbbox()
        im = im.crop(bb)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(im).astype(np.float32)
        lum = a[..., :3].mean(axis=2, keepdims=True)
        shade = 0.72 + 0.28 * lum / max(1.0, float(lum[a[..., 3] > 0].max()))
        a[..., :3] = np.array(color, np.float32) * shade
        im = Image.fromarray(a.astype(np.uint8))
        im = im.resize((max(1, int(im.width * height / im.height)), height), Image.LANCZOS)
        if angle:
            im = im.rotate(angle, expand=True, resample=Image.BICUBIC)
        _arrow_cache[key] = im
    im = _arrow_cache[key]
    p = ease_out(t / (dur / FPS))
    m = Image.new("L", im.size, 0)
    md = ImageDraw.Draw(m)
    w, h = im.size
    if wipe == "down":
        md.rectangle((0, 0, w, h * p), fill=255)
    elif wipe == "right":
        md.rectangle((0, 0, w * p, h), fill=255)
    elif wipe == "left":
        md.rectangle((w * (1 - p), 0, w, h), fill=255)
    else:
        md.rectangle((0, h * (1 - p), w, h), fill=255)
    a = im.copy()
    a.putalpha(Image.fromarray(np.minimum(np.asarray(im)[..., 3], np.asarray(m))))
    s = 1.04 - 0.04 * ease_out((t - dur / FPS) / (6 / FPS)) if t > dur / FPS else 1.04
    if s != 1:
        a = a.resize((int(w * s), int(h * s)), Image.BICUBIC)
    layer.alpha_composite(a, (int(center[0] - a.width / 2), int(center[1] - a.height / 2)))


def measure(layer, a, b, t, label, color=(216, 210, 196)):
    """3.3: a dashed bone line A->B over 14 f with 12-px end ticks; the distance plate at the midpoint at f 14."""
    if t <= 0:
        return
    d = ImageDraw.Draw(layer)
    p = ease_out(t / (14 / FPS))
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    s = 0.0
    while s < L * p:
        e = min(s + 10, L * p)
        d.line([(a[0] + ux * s, a[1] + uy * s), (a[0] + ux * e, a[1] + uy * e)], fill=(*color, 255), width=3)
        s += 16
    nx, ny = -uy, ux
    for q, show in ((a, True), (b, p >= 1)):
        if show:
            d.line([(q[0] - nx * 12, q[1] - ny * 12), (q[0] + nx * 12, q[1] + ny * 12)], fill=(*color, 255), width=3)
    if t > 14 / FPS:
        ta = ease_out((t - 14 / FPS) / (8 / FPS))
        f = font("BarlowCondensed-Bold", 40)
        mx, my = (a[0] + b[0]) / 2 + 26, (a[1] + b[1]) / 2 - 26
        tw = d.textlength(label, font=f)
        d.rectangle((mx, my, mx + tw + 22, my + 50), fill=(0, 0, 0, int(220 * ta)))
        d.text((mx + 11, my + 2), label, font=f, fill=(240, 138, 36, int(255 * ta)))


def stat_panel(layer, t, value, unit, fmt="{:,.0f}", prefix="", suffix="", box=(1290, 640, 1850, 880),
               numfont=("BarlowCondensed-Bold", 120, None), numcolor=(240, 138, 36), base=(10, 12, 14),
               accent=(240, 138, 36), source=None, second=None):
    """3.4 option B: corner panel, 3-px accent rule on top, the number counts up over 24 f (tabular), a 2-px rule
    under it draws over the same 24 f. second=(value, t_offset): a comparison numeral lands 12 f later."""
    if t <= 0:
        return
    d = ImageDraw.Draw(layer)
    a = ease_out(t / (8 / FPS))
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(*base, int(235 * a)))
    d.line([(x0, y0), (x1, y0)], fill=(*accent, int(255 * a)), width=3)
    c = ease_out(t / (24 / FPS))
    nf = font(*numfont)
    txt = prefix + fmt.format(value * c) + suffix
    full = prefix + fmt.format(value) + suffix
    d.text((x0 + 28, y0 + 14), txt, font=nf, fill=(*numcolor, int(255 * a)))
    # laid out from the number's measured outline (owner, 12 Nov 2026: the rule cut through Playfair's
    # descending figures and the arrow floated): the rule 10 px under the lowest ink, the unit under the rule
    bx0, by0, bx1, by1 = d.textbbox((x0 + 28, y0 + 14), full, font=nf)
    rule_y = by1 + 10
    d.line([(bx0, rule_y), (bx0 + (bx1 - bx0) * c, rule_y)], fill=(*accent, int(255 * a)), width=2)
    if second:
        v2, off, label2 = second
        t2 = t - off
        if t2 > 0:
            a2 = ease_out(t2 / (12 / FPS))
            af = font("SourceSerif4", int(nf.size * 0.5))
            d.text((bx1 + 22, (by0 + by1) / 2), "→", font=af, anchor="lm", fill=(216, 210, 196, int(255 * a2)))
            gx = bx1 + 22 + d.textlength("→", font=af) + 18
            d.text((gx, y0 + 14), label2, font=nf, fill=(226, 58, 46, int(255 * a2)))
    spaced(d, (x0 + 30, rule_y + 12), unit.upper(), font("Inter", 30), (216, 210, 196, int(255 * a)), track=0.1)
    if source:
        d.text((x0 + 30, y1 - 32), source.upper(), font=font("IBMPlexMono-Regular", 15), fill=(216, 210, 196, int(140 * a)))


def captions(layer, words, t, dur, accent=(240, 138, 36), keys=(), y=H - 150):
    """3.11: Inter SemiBold 40, bone with a 1-px charcoal halo, groups of 2-4 words, key words in the accent,
    each group pops 0.96 -> 1.0 over 4 f."""
    if not words or t < 0 or t > dur:
        return
    groups = [words[i:i + 3] for i in range(0, len(words), 3)]
    per = dur / len(groups)
    k = min(len(groups) - 1, int(t / per))
    g = groups[k]
    d = ImageDraw.Draw(layer)
    f = font("Inter", 44, "SemiBold")
    sp = d.textlength(" ", font=f)
    widths = [d.textlength(w, font=f) for w in g]
    total = sum(widths) + sp * (len(g) - 1)
    x = W / 2 - total / 2
    for w, ww in zip(g, widths):
        key = any(ch.isdigit() for ch in w) or w.strip(".,?").lower() in keys
        col = accent if key else (216, 210, 196)
        for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            d.text((x + ox, y + oy), w, font=f, fill=(25, 27, 30, 230))
        d.text((x, y), w, font=f, fill=(*col, 255))
        x += ww + sp


# ---------- the retro headline clipping (4.1) ----------
class Clipping:
    """A dated clipping of our own design: paper, typed dateline, headline, greeked columns, optional stamp; the key
    phrase gets the highlighter swipe. Never a real masthead or article text."""

    def __init__(self, dateline, headline, key, paper="#E9E2D0", accent="#C8862B", style="por", seed=1, stamp=None,
                 size=(1180, 660)):
        self.rng = np.random.default_rng(seed)
        self.dateline, self.style, self.stamp = dateline, style, stamp
        sw = size[0] * (1 + self.rng.uniform(-0.1, 0.1))
        self.w, self.h = int(sw), int(size[1] * (1 + self.rng.uniform(-0.08, 0.08)))
        self.accent = hexrgb(accent)
        pr = np.array(hexrgb(paper), np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        warm = ((xx / self.w + yy / self.h) / 2)[..., None] * np.array([6, 3, -8], np.float32)  # yellowing to a corner
        fib = self.rng.normal(0, 1, (self.h, self.w)).astype(np.float32)
        fib = cv2.GaussianBlur(fib, (0, 0), 1.5)[..., None] * 10
        base = np.clip(pr + warm + fib, 0, 255)
        fy = self.h // 2 + int(self.rng.uniform(-30, 30))  # the fold
        base[fy] *= 0.94
        base[fy + 1:fy + 3] = np.minimum(base[fy + 1:fy + 3] * 1.04, 255)
        self.paper = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.paper)
        ink = (43, 46, 51)
        self.dl_xy = (48, 40)
        d.line([(48, 76), (self.w - 48, 76)], fill=(*ink, 200), width=1)
        d.line([(48, 80), (self.w - 48, 80)], fill=(*ink, 200), width=1)
        # headline: fitted to two lines max
        hf_name, var = (("PlayfairDisplay", "Black") if style == "por" else ("BarlowCondensed-Bold", None))
        size_h = 92
        while True:
            hf = font(hf_name, size_h, var)
            lines = self._wrap(d, headline, hf, self.w - 96)
            if len(lines) <= 2 or size_h <= 50:
                break
            size_h -= 4
        self.hf, self.lines = hf, lines
        y = 104
        self.key_boxes = []
        self.head_layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        hd = ImageDraw.Draw(self.head_layer)
        for ln in lines:
            hd.text((48, y), ln, font=hf, fill=(*ink, 255))
            if key and key in ln:
                x0 = 48 + d.textlength(ln[:ln.index(key)], font=hf)
                self.key_boxes.append((x0 - 8, y + hf.size * 0.22, x0 + d.textlength(key, font=hf) + 8, y + hf.size * 1.14))
            y += int(hf.size * 1.12)
        self.body_top = y + 26
        # greeked columns (3.6): rows of 25% grey bars, random lengths, every 5th short; no readable text
        cols = 3 if self.w > 900 else 2
        cw = (self.w - 96 - 30 * (cols - 1)) / cols
        for ci in range(cols):
            cx = 48 + ci * (cw + 30)
            ry = self.body_top
            row = 0
            while ry < self.h - 40:
                L = cw * (self.rng.uniform(0.6, 1.0) if (row % 5) != 4 else self.rng.uniform(0.25, 0.5))
                d.rectangle((cx, ry, cx + L, ry + 7), fill=(150, 146, 138, 255))
                ry += 17
                row += 1
        if stamp:
            st = Image.new("RGBA", (360, 120), (0, 0, 0, 0))
            sd = ImageDraw.Draw(st)
            sd.rectangle((6, 6, 354, 114), outline=(178, 34, 34, 200), width=6)
            spaced(sd, (30, 22), stamp, font("IBMPlexMono-Medium", 64), (178, 34, 34, 210), track=0.12)
            noise = (self.rng.random((120, 360)) > 0.25).astype(np.uint8) * 255  # ink texture
            st.putalpha(Image.fromarray(np.minimum(np.asarray(st)[..., 3], noise)))
            st = st.rotate(8, expand=True, resample=Image.BICUBIC)
            self.paper.alpha_composite(st, (self.w - st.width - 40, self.h - st.height - 30))
        # torn edge mask (per-edge random walk) and its shadow
        m = Image.new("L", (self.w, self.h), 0)
        md = ImageDraw.Draw(m)
        def walk(n):
            return np.cumsum(self.rng.normal(0, 1.2, n)).clip(-4, 4)
        top, bot = walk(self.w), walk(self.w)
        left, right = walk(self.h), walk(self.h)
        poly = [(x, 6 + top[x]) for x in range(0, self.w, 4)] + [(self.w - 6 + right[y], y) for y in range(0, self.h, 4)] \
            + [(x, self.h - 6 + bot[x]) for x in range(self.w - 1, 0, -4)] + [(6 + left[y], y) for y in range(self.h - 1, 0, -4)]
        md.polygon(poly, fill=255)
        self.mask = m

    @staticmethod
    def _wrap(d, text, f, maxw):
        words, lines, cur = text.split(), [], ""
        for w in words:
            nxt = (cur + " " + w).strip()
            if d.textlength(nxt, font=f) <= maxw:
                cur = nxt
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def render(self, t, dim=1.0):
        """The clipping t seconds after it lands: dateline typed one character a frame from f 10, highlighter
        swipe f 30-41. Returns an RGBA PIL image (with its shadow margin)."""
        img = self.paper.copy()
        sw = ease_out((t - 30 / FPS) / (11 / FPS))
        if sw > 0 and self.key_boxes:  # 3.3 highlighter: a multiply swipe at 38%, so the ink stays dark on top
            arr = np.asarray(img).astype(np.float32)
            tint = 1 - 0.38 + 0.38 * np.array(self.accent, np.float32) / 255
            for (x0, y0, x1, y1) in self.key_boxes:
                xe = int(x0 + (x1 - x0) * sw)
                arr[int(y0):int(y1), int(x0):xe, :3] *= tint
            img = Image.fromarray(arr.astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        img.alpha_composite(self.head_layer)
        n = max(0, int((t - 10 / FPS) * FPS))
        txt = self.dateline[:n]
        spaced(d, self.dl_xy, txt.upper(), font("IBMPlexMono-Medium", 22), (43, 46, 51, 205), track=0.12)
        img.putalpha(Image.fromarray(np.minimum(np.asarray(img)[..., 3], np.asarray(self.mask))))
        if dim < 1:
            arr = np.asarray(img).astype(np.float32)
            arr[..., :3] *= dim
            img = Image.fromarray(arr.astype(np.uint8))
        pad = 40
        out = Image.new("RGBA", (self.w + 2 * pad, self.h + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sh.paste((0, 0, 0, 102), (pad, pad + 10), self.mask)
        sh = sh.filter(ImageFilter.GaussianBlur(12))
        out.alpha_composite(sh)
        out.alpha_composite(img, (pad, pad))
        return out


def place(layer, img, center, angle, scale):
    """Rotate/scale an RGBA image and composite it centred at `center` on a full-frame layer."""
    if scale != 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    layer.alpha_composite(img, (int(center[0] - img.width / 2), int(center[1] - img.height / 2))) if (
        0 <= int(center[0] - img.width / 2) and 0 <= int(center[1] - img.height / 2)
        and int(center[0] + img.width / 2) <= W and int(center[1] + img.height / 2) <= H) else _clip_paste(layer, img, center)


def _clip_paste(layer, img, center):
    x, y = int(center[0] - img.width / 2), int(center[1] - img.height / 2)
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp.paste(img, (x, y), img)
    layer.alpha_composite(tmp)


# ---------- audio ----------
def load_audio(path, sec=None):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), *(["-t", str(sec)] if sec else []), "-ac", "2", "-ar", str(SR),
           "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()


def synth(kind, sec=None):
    """Short SFX made here (no third-party audio): marker swipe, low thud, riser, whoosh, soft tick."""
    rng = np.random.default_rng(3)
    if kind == "thud":
        n = int(SR * 0.45)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * (52 - 10 * t) * t) * np.exp(-t * 9) * 0.9
    elif kind == "tick":
        n = int(SR * 0.06)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 90) * 0.35
    elif kind in ("swipe", "whoosh", "riser"):
        dur = {"swipe": 0.26, "whoosh": 0.65, "riser": sec or 1.5}[kind]
        n = int(SR * dur)
        x = rng.normal(0, 1, n)
        t = np.arange(n) / SR
        lo = {"swipe": 2500, "whoosh": 300, "riser": 200}[kind]
        hi = {"swipe": 6000, "whoosh": 1400, "riser": 5000}[kind]
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(n, 1 / SR)
        X[(fr < lo) | (fr > hi)] = 0
        x = np.fft.irfft(X, n)
        x /= np.abs(x).max() + 1e-9
        if kind == "riser":
            env = (t / dur) ** 2.2
        else:
            env = np.sin(np.pi * t / dur) ** 2
        x = x * env * {"swipe": 0.25, "whoosh": 0.35, "riser": 0.4}[kind]
    else:
        raise ValueError(kind)
    return np.stack([x, x], 1).astype(np.float32)


def sfx(name):
    return load_audio(SFX / f"{name}.wav")


class Mix:
    def __init__(self, sec):
        self.buf = np.zeros((int(sec * SR) + SR, 2), np.float32)
        self.vo_spans = []

    def add(self, audio, at, gain=1.0):
        i = int(at * SR)
        if i < 0 or i >= len(self.buf):
            return
        a = audio[: len(self.buf) - i] * gain
        self.buf[i:i + len(a)] += a

    def vo(self, audio, at):
        self.add(audio, at, 1.0)
        self.vo_spans.append((at, at + len(audio) / SR))

    def music(self, audio, start, end, gain=0.20, duck=0.42, fade=1.0, silence=()):
        """The bed from start to end, ducked under the voice, faded at both ends, silent in the given spans."""
        n0, n1 = int(start * SR), int(end * SR)
        a = np.resize(audio, (n1 - n0, 2)).copy()
        t = start + np.arange(len(a)) / SR
        env = np.ones(len(a), np.float32) * gain
        for s, e in self.vo_spans:
            k = np.clip(1 - np.maximum(np.maximum(s - 0.25 - t, t - e - 0.25), 0) / 0.25, 0, 1)
            env *= 1 - (1 - duck) * k
        env *= np.clip((t - start) / fade, 0, 1) * np.clip((end - t) / fade, 0, 1)
        for s, e in silence:
            env *= np.clip(np.maximum(s - t, t - e) / 0.15, 0, 1)
        self.buf[n0:n1] += a * env[:, None]

    def write(self, path):
        peak = np.abs(self.buf).max()
        out = self.buf / max(1.0, peak / 0.95)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-c:a", "pcm_s16le", str(path)], input=out.astype(np.float32).tobytes(), check=True)


# ---------- output ----------
class Writer:
    def __init__(self, path):
        self.path = path
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                                   "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def mux(video, audio, dest):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", str(dest)], check=True)


def lines():
    return json.loads((HERE / "vo" / "lines.json").read_text())
