"""Brand diagrams (ink background, bone lines, yellow labels, red for the danger), drawn in code so the
facts are exact. Each returns a 1920x1080 RGB image; build.py reveals it with a slow push-in.

    corner_stress    a square window and a round one, stress lines flowing around them; the square
                     corner glows red where the stress piles up
    round_vs_square  1954 (square, cracked) vs today (rounded, safe)
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
INK, BONE, YEL, RED, TEAL = (10, 11, 15), (242, 238, 227), (255, 214, 10), (255, 59, 48), (15, 59, 68)
W, H = 1920, 1080


def font(size, bold=True):
    return ImageFont.truetype(str(ROOT / "brand/fonts" / ("Anton-Regular.ttf" if bold else "Inter.ttf")), size)


def base():
    img = Image.new("RGB", (W, H), INK)
    g = Image.new("L", (W, H), 0)
    ImageDraw.Draw(g).ellipse((W * 0.1, -H * 0.2, W * 0.9, H * 1.2), fill=90)
    img.paste(TEAL, mask=g.filter(ImageFilter.GaussianBlur(220)))
    return img


def flow_lines(d, box, round_r, cx_shift=0):
    """Horizontal stress lines bending around an opening (box), bunching near its corners."""
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    for k in range(-5, 6):
        y_far = cy + k * 40
        pts = []
        for x in range(int(x0 - 230), int(x1 + 231), 6):
            u = (x - (x0 + x1) / 2) / ((x1 - x0) / 2 + 160)
            bend = math.exp(-u * u * 2.2)
            half = (y1 - y0) / 2 + 26
            target = cy + math.copysign(half + abs(k) * 14, k if k else 1) if abs(y_far - cy) < half + 40 else y_far
            pts.append((x + cx_shift, y_far + (target - y_far) * bend))
        d.line(pts, fill=BONE, width=3)


def corner_stress(_=None):
    img = base()
    d = ImageDraw.Draw(img)
    d.text((W / 2, 90), "WHERE THE STRESS GOES", font=font(78), fill=YEL, anchor="mm")
    sq = (300, 360, 700, 760)
    rd = (1220, 360, 1620, 760)
    flow_lines(d, sq, 0)
    flow_lines(d, rd, 120)
    d.rectangle(sq, fill=INK, outline=BONE, width=6)
    d.rounded_rectangle(rd, radius=150, fill=INK, outline=BONE, width=6)
    hot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hot)
    for cx, cy in ((sq[0], sq[1]), (sq[2], sq[1]), (sq[0], sq[3]), (sq[2], sq[3])):
        hd.ellipse((cx - 60, cy - 60, cx + 60, cy + 60), fill=RED + (200,))
    img.paste(hot.filter(ImageFilter.GaussianBlur(22)), (0, 0), hot.filter(ImageFilter.GaussianBlur(22)))
    d = ImageDraw.Draw(img)
    d.text((500, 930), "SHARP CORNER", font=font(64), fill=BONE, anchor="mm")
    d.text((500, 1005), "stress piles up in one point", font=font(40, False), fill=RED, anchor="mm")
    d.text((1420, 930), "ROUNDED CORNER", font=font(64), fill=BONE, anchor="mm")
    d.text((1420, 1005), "stress flows around evenly", font=font(40, False), fill=YEL, anchor="mm")
    return img


def round_vs_square(_=None):
    img = base()
    d = ImageDraw.Draw(img)
    d.text((W / 2, 90), "THE LESSON OF 1954", font=font(78), fill=YEL, anchor="mm")
    for cx, label, sub, rounded in ((560, "1954", "square corners: fatigue cracks", False),
                                    (1360, "TODAY", "rounded corners: stress spreads", True)):
        box = (cx - 170, 280, cx + 170, 760)
        d.rounded_rectangle((box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40), radius=60 if rounded else 18,
                            fill=(214, 209, 196))
        if rounded:
            d.rounded_rectangle(box, radius=170, fill=(86, 140, 200))
        else:
            d.rounded_rectangle(box, radius=14, fill=(86, 140, 200))
            d.line([(box[2] - 4, box[1] + 4), (box[2] + 30, box[1] - 28), (box[2] + 22, box[1] - 60),
                    (box[2] + 48, box[1] - 96)], fill=RED, width=8)
        d.text((cx, 870), label, font=font(72), fill=BONE, anchor="mm")
        d.text((cx, 950), sub, font=font(38, False), fill=RED if not rounded else YEL, anchor="mm")
    return img


def title(d, text):
    d.text((W / 2, 90), text, font=font(78), fill=YEL, anchor="mm")


def altitude(_=None):
    """Propeller airliners cruised in the weather (~20,000 ft); the Comet about twice as high."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "ABOVE THE WEATHER")
    ground, top = 960, 200
    ft = lambda f: ground - (ground - top) * f / 45000
    for f in (10000, 20000, 30000, 40000):
        d.line((260, ft(f), 1700, ft(f)), fill=(60, 70, 78), width=2)
        d.text((240, ft(f)), f"{f // 1000},000 ft", font=font(32, False), fill=BONE, anchor="rm")
    d.rectangle((260, ft(18000), 1700, ground), fill=(30, 52, 60))  # weather band
    d.text((1690, ft(9000)), "WEATHER", font=font(46), fill=(120, 150, 160), anchor="rm")
    for x, f, name, col in ((620, 20000, "PROPELLER AIRLINERS", BONE), (1300, 40000, "DE HAVILLAND COMET", YEL)):
        d.rectangle((x - 120, ft(f), x + 120, ground), fill=col)
        d.text((x, ft(f) - 95), name, font=font(44), fill=col, anchor="mm")
        d.text((x, ft(f) - 45), f"about {f // 1000},000 ft", font=font(34, False), fill=BONE, anchor="mm")
    return img


def pressurisation(_=None):
    """Cabin air pushes out on every part of the skin: the fuselage swells on every flight."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "EVERY FLIGHT, THE CABIN SWELLS")
    cx, cy, r = W / 2, 580, 300
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 40, 46), outline=BONE, width=10)
    d.text((cx, cy - 30), "CABIN AIR", font=font(70), fill=BONE, anchor="mm")
    d.text((cx, cy + 40), "close to ground-level pressure", font=font(36, False), fill=BONE, anchor="mm")
    for k in range(16):
        a = 2 * math.pi * k / 16
        x0, y0 = cx + (r - 70) * math.cos(a), cy + (r - 70) * math.sin(a)
        x1, y1 = cx + (r + 70) * math.cos(a), cy + (r + 70) * math.sin(a)
        d.line((x0, y0, x1, y1), fill=YEL, width=8)
        hx, hy = math.cos(a), math.sin(a)
        d.polygon([(x1 + 22 * hx, y1 + 22 * hy), (x1 - 14 * hy, y1 + 14 * hx), (x1 + 14 * hy, y1 - 14 * hx)], fill=YEL)
    d.text((cx + r + 250, cy - 180), "THIN AIR", font=font(56), fill=(120, 150, 160), anchor="mm")
    d.text((cx + r + 250, cy - 120), "outside, at cruising height", font=font(32, False), fill=(120, 150, 160), anchor="mm")
    d.text((cx, 1000), "the metal skin is stretched outward, then relaxes after landing", font=font(38, False), fill=YEL, anchor="mm")
    return img


def crash_map(_=None):
    """Schematic (not to scale): both flights left Rome; one broke up near Elba, one near Naples."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "TWO FLIGHTS. SAME FATE.")
    pts = {"LONDON": (520, 230), "ROME": (980, 600), "CAIRO": (1500, 930)}
    lerp = lambda a, b, u: (pts[a][0] + (pts[b][0] - pts[a][0]) * u, pts[a][1] + (pts[b][1] - pts[a][1]) * u)
    pts["ELBA"], pts["NAPLES"] = lerp("ROME", "LONDON", 0.22), lerp("ROME", "CAIRO", 0.22)  # just after leaving Rome
    def route(a, b, col):
        (x0, y0), (x1, y1) = pts[a], pts[b]
        for t in range(0, 100, 4):
            u0, u1 = t / 100, (t + 2) / 100
            d.line((x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0, x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1), fill=col, width=6)
    route("ROME", "LONDON", BONE)
    route("ROME", "CAIRO", BONE)
    for name in ("LONDON", "ROME", "CAIRO"):
        x, y = pts[name]
        d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=BONE)
        d.text((x + 28, y), name, font=font(44), fill=BONE, anchor="lm")
    for name, label, sub in (("ELBA", "NEAR ELBA", "10 January 1954 · 35 lost"), ("NAPLES", "NEAR NAPLES", "8 April 1954 · 21 lost")):
        x, y = pts[name]
        d.line((x - 26, y - 26, x + 26, y + 26), fill=RED, width=12)
        d.line((x - 26, y + 26, x + 26, y - 26), fill=RED, width=12)
        tx = x - 50 if name == "ELBA" else x + 50
        d.text((tx, y - 70), label, font=font(48), fill=RED, anchor="rm" if name == "ELBA" else "lm")
        d.text((tx, y - 20), sub, font=font(32, False), fill=BONE, anchor="rm" if name == "ELBA" else "lm")
    d.text((W - 40, H - 30), "schematic, not to scale", font=font(26, False), fill=(110, 120, 128), anchor="rd")
    return img


def pressure_cycles(_=None):
    """The water tank: pressurise, release, repeat. Every cycle = one flight."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "ONE CYCLE = ONE FLIGHT")
    x0, x1, yb, yt = 220, 1720, 860, 360
    d.line((x0, yb, x1, yb), fill=BONE, width=4)
    d.line((x0, yb, x0, yt - 40), fill=BONE, width=4)
    d.text((x0 - 20, yt), "cabin\npressure", font=font(34, False), fill=BONE, anchor="rm")
    pts, n = [], 7
    for k in range(n):
        a = x0 + 40 + k * (x1 - x0 - 300) / n
        w = (x1 - x0 - 300) / n
        pts += [(a, yb), (a + w * 0.15, yt), (a + w * 0.7, yt), (a + w * 0.85, yb)]
    d.line(pts, fill=YEL, width=8)
    d.text((x1 - 110, (yb + yt) / 2), "x 3,000", font=font(110), fill=RED, anchor="mm")
    d.text(((x0 + x1) / 2, yb + 70), "inflate · release · inflate · release ...  day and night, in a tank of water",
           font=font(36, False), fill=BONE, anchor="mm")
    return img


def fatigue_crack(_=None):
    """Fatigue: the crack at a sharp corner grows a little with every flight."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "METAL FATIGUE")
    for i, (label, length) in enumerate((("FLIGHT 1", 0), ("FLIGHT 1,000", 45), ("FLIGHT 3,000", 140))):
        cx = 380 + i * 580
        box = (cx - 110, 370, cx + 110, 590)
        d.rectangle((box[0] - 150, box[1] - 150, box[2] + 150, box[3] + 150), fill=(150, 150, 146))
        d.rectangle(box, fill=INK)
        if length:
            x, y = box[2], box[1]
            pts = [(x, y)]
            for k in range(1, 8):
                pts.append((x + length * k / 7 + (6 if k % 2 else -6), y - length * k / 7))
            d.line(pts, fill=RED, width=7)
        d.text((cx, 820), label, font=font(60), fill=BONE, anchor="mm")
    d.text((W / 2, 930), "too small to see, until the skin lets go", font=font(40, False), fill=YEL, anchor="mm")
    return img


DIAGRAMS = {"corner_stress": corner_stress, "round_vs_square": round_vs_square, "altitude": altitude,
            "pressurisation": pressurisation, "crash_map": crash_map, "pressure_cycles": pressure_cycles,
            "fatigue_crack": fatigue_crack}
