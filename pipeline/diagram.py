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


def decompression(_=None):
    """The skin fails at one point and the pressurised cabin bursts outward."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "IT DOESN'T LEAK. IT BURSTS.")
    cx, cy, r = W / 2 - 120, 590, 290
    gap = (math.radians(-60), math.radians(-30))  # the failed section, upper right
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 40, 46))
    steps = 200
    for k in range(steps):
        a0, a1 = 2 * math.pi * k / steps, 2 * math.pi * (k + 1) / steps
        am = (a0 + a1) / 2
        if gap[0] <= (am if am <= math.pi else am - 2 * math.pi) <= gap[1]:
            continue
        d.line((cx + r * math.cos(a0), cy + r * math.sin(a0), cx + r * math.cos(a1), cy + r * math.sin(a1)), fill=BONE, width=10)
    am = sum(gap) / 2
    for k in range(-3, 4):  # air blasting out through the gap
        a = am + k * 0.07
        x0, y0 = cx + (r - 120) * math.cos(a), cy + (r - 120) * math.sin(a)
        x1, y1 = cx + (r + 260 + abs(k) * -20) * math.cos(a), cy + (r + 260 + abs(k) * -20) * math.sin(a)
        d.line((x0, y0, x1, y1), fill=RED, width=9)
        d.polygon([(x1 + 26 * math.cos(a), y1 + 26 * math.sin(a)), (x1 - 16 * math.sin(a), y1 + 16 * math.cos(a)),
                   (x1 + 16 * math.sin(a), y1 - 16 * math.cos(a))], fill=RED)
    for k in range(5):  # torn metal flaps
        a = gap[0] + (gap[1] - gap[0]) * k / 4
        d.line((cx + r * math.cos(a), cy + r * math.sin(a), cx + (r + 50) * math.cos(a + 0.08), cy + (r + 50) * math.sin(a + 0.08)),
               fill=BONE, width=6)
    d.text((cx, cy - 20), "CABIN AIR", font=font(64), fill=BONE, anchor="mm")
    d.text((cx, cy + 45), "escapes in a fraction of a second", font=font(34, False), fill=BONE, anchor="mm")
    d.text((W / 2, 1010), "one crack, and the whole fuselage tears open", font=font(40, False), fill=YEL, anchor="mm")
    return img


def _ring(d, x, y, r):
    d.ellipse((x - r, y - r, x + r, y + r), outline=RED, width=7)


def _crack(d, x, y, dx, dy, length, seed, width=4):
    """A thin jagged dark crack from (x, y) along (dx, dy)."""
    import random
    rnd = random.Random(seed)
    pts, n = [(x, y)], 9
    for k in range(1, n + 1):
        u = k / n
        jit = (rnd.random() - 0.5) * 14
        pts.append((x + dx * length * u - dy * jit, y + dy * length * u + dx * jit))
    d.line(pts, fill=(18, 16, 14), width=width)
    d.line([(px + 1, py + 1) for px, py in pts], fill=(200, 200, 205), width=1)  # glint on one edge


def corner_crack(img):
    """Shot 26: a square window cut-out in the riveted panel, a crack from its sharp top corner."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    x0, y0, x1, y1 = int(w * 0.36), int(h * 0.30), int(w * 0.58), int(h * 0.78)
    d.rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), fill=(120, 124, 130))
    d.rectangle((x0, y0, x1, y1), fill=(14, 16, 20))
    _crack(d, x1, y0, 0.72, -0.69, int(h * 0.26), 3)
    _ring(d, x1, y0, int(h * 0.12))
    return img


def rivet_crack(img):
    """Shot 30: a hairline fatigue crack growing from a rivet hole."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    x, y = int(w * 0.50), int(h * 0.48)
    d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(30, 30, 34))
    _crack(d, x + 6, y - 2, 0.95, -0.3, int(w * 0.20), 5, width=3)
    _crack(d, x - 6, y + 2, -0.9, 0.35, int(w * 0.12), 9, width=3)
    _ring(d, x, y, int(h * 0.16))
    return img


def square_hole(img):
    """Shot 32: a small square opening in the recovered roof panel, one corner torn."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy, a = int(w * 0.48), int(h * 0.58), int(h * 0.09)
    d.rectangle((cx - a - 5, cy - a - 5, cx + a + 5, cy + a + 5), fill=(150, 154, 160))
    d.polygon([(cx - a, cy - a), (cx + a, cy - a), (cx + a + 30, cy - a - 26), (cx + a + 8, cy - a + 18),
               (cx + a, cy + a), (cx - a, cy + a)], fill=(12, 12, 14))
    _crack(d, cx + a + 20, cy - a - 18, 0.8, -0.6, int(h * 0.10), 11)
    _ring(d, cx + a, cy - a, int(h * 0.11))
    return img


def _button(d, cx, cy, r, label, lit=False):
    d.ellipse((cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6), fill=(120, 104, 70))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(250, 214, 120) if lit else (40, 38, 34), outline=(196, 160, 90), width=4)
    if label:
        d.text((cx, cy), label, font=font(int(r * 1.0)), fill=INK if lit else BONE, anchor="mm")


def elevator_gap(_=None):
    """A brass elevator panel: 10, 11, 12, (gap), 14, 15 - the missing 13 ringed."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "SOMETHING IS MISSING")
    d.rounded_rectangle((W / 2 - 260, 190, W / 2 + 260, 1030), radius=30, fill=(70, 62, 48), outline=(170, 140, 80), width=8)
    floors = [16, 15, 14, 13, 12, 11, 10]
    for i, f in enumerate(floors):
        cy = 270 + i * 112
        if f == 13:
            d.text((W / 2, cy), "?", font=font(90), fill=RED, anchor="mm")
            d.ellipse((W / 2 - 80, cy - 80, W / 2 + 80, cy + 80), outline=RED, width=10)
        else:
            _button(d, W / 2, cy, 44, str(f), lit=(f == 14))
    d.text((W / 2 + 420, 270 + 3 * 112), "no button 13", font=font(54), fill=RED, anchor="mm")
    return img


def otis_85(_=None):
    """Otis estimate (2002): about 85% of its buildings with 13+ floors had no floor labelled 13."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "NO FLOOR 13")
    x0, y0, w, h = 300, 360, 1320, 150
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=20, fill=(40, 52, 58))
    d.rounded_rectangle((x0, y0, x0 + int(w * 0.85), y0 + h), radius=20, fill=YEL)
    d.text((x0 + int(w * 0.85) / 2, y0 + h / 2), "ABOUT 85%", font=font(96), fill=INK, anchor="mm")
    d.text((x0 + w * 0.925, y0 + h / 2), "15%", font=font(54), fill=BONE, anchor="mm")
    d.text((W / 2, 640), "of tall buildings with Otis elevators had no button marked 13", font=font(46, False), fill=BONE, anchor="mm")
    d.text((W / 2, 720), "estimate by an Otis executive, 2002", font=font(34, False), fill=(130, 140, 148), anchor="mm")
    return img


def real_floors(_=None):
    """Count the real floors: the floor labelled 14 is the 13th floor."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "COUNT THE FLOORS")
    bx0, bx1, top, bottom, n = 760, 1160, 190, 1010, 8
    fh = (bottom - top) / n
    for k in range(n):  # real floors 9..16 from the bottom
        real = 9 + k
        y1 = bottom - k * fh
        y0 = y1 - fh
        hot = real == 13
        d.rectangle((bx0, y0 + 4, bx1, y1 - 4), fill=(80, 40, 36) if hot else (34, 52, 60), outline=RED if hot else BONE, width=6 if hot else 3)
        label = real if real < 13 else real + 1
        d.text((bx0 - 40, (y0 + y1) / 2), f"button {label}", font=font(44), fill=RED if hot else BONE, anchor="rm")
        d.text((bx1 + 40, (y0 + y1) / 2), f"real floor {real}", font=font(44), fill=YEL if hot else (150, 160, 166), anchor="lm")
    return img


def tetraphobia(_=None):
    """Some towers skip every floor with a 4 (and 13): the top button overstates the real height."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "SKIPPED: 4, 13, 14, 24...")
    nums = list(range(1, 31))
    skipped = {4, 13, 14, 24}
    for i, f in enumerate(nums):
        r, c = divmod(i, 10)
        cx, cy = 300 + c * 146, 330 + r * 210
        if f in skipped:
            d.ellipse((cx - 50, cy - 50, cx + 50, cy + 50), outline=RED, width=6)
            d.text((cx, cy), str(f), font=font(52), fill=RED, anchor="mm")
            d.line((cx - 40, cy + 40, cx + 40, cy - 40), fill=RED, width=6)
        else:
            _button(d, cx, cy, 44, str(f))
    d.text((W / 2, 1000), "in Chinese, \"four\" sounds almost like \"death\"", font=font(42, False), fill=YEL, anchor="mm")
    return img


def underground_fire(_=None):
    """Cross-section: town on the surface, coal seam and old mine tunnels below, the fire front moving
    along the seam, air drawn in through cracks and old openings, smoke and heat venting up."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "WHY IT CAN'T BE PUT OUT")
    ground = 330
    d.rectangle((0, ground, W, H), fill=(44, 34, 28))
    for x in (260, 420, 600, 1500, 1660):  # houses
        d.rectangle((x, ground - 70, x + 110, ground), fill=(150, 150, 146))
        d.polygon([(x - 10, ground - 70), (x + 55, ground - 120), (x + 120, ground - 70)], fill=(110, 110, 106))
    d.polygon([(0, 640), (W, 560), (W, 700), (0, 780)], fill=(18, 18, 20))  # coal seam
    d.text((80, 720), "COAL SEAM", font=font(44), fill=(150, 150, 150), anchor="lm")
    for x in range(120, W, 260):  # old mine tunnels in the seam
        y = 640 + (560 - 640) * x / W
        d.rectangle((x, y + 28, x + 150, y + 82), fill=(70, 60, 52))
    fx = 1050
    fire = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fire)
    fd.polygon([(fx - 260, 640 - 50), (fx + 120, 600 - 40), (fx + 120, 700 + 40), (fx - 260, 760 + 20)], fill=(255, 90, 20, 210))
    img.paste(fire.filter(ImageFilter.GaussianBlur(26)), (0, 0), fire.filter(ImageFilter.GaussianBlur(26)))
    d = ImageDraw.Draw(img)
    d.text((fx - 70, 860), "FIRE FRONT  →  moving through the coal", font=font(46), fill=YEL, anchor="mm")
    for x in (760, 980, 1180):  # smoke and heat venting through cracks
        d.line((x, 600, x - 20, 470, x + 10, ground), fill=(190, 190, 190), width=5)
        d.text((x, ground - 40), "~", font=font(70), fill=(190, 190, 190), anchor="mm")
    for x in (1380, 1840):  # air drawn in
        d.line((x, ground - 60, x, 540), fill=(120, 190, 255), width=8)
        d.polygon([(x, 560), (x - 18, 530), (x + 18, 530)], fill=(120, 190, 255))
    d.text((1610, ground + 120), "AIR IN", font=font(44), fill=(120, 190, 255), anchor="mm")
    d.text((W / 2, 1010), "old tunnels and cracks keep feeding it air", font=font(40, False), fill=BONE, anchor="mm")
    return img


def cost_compare(_=None):
    """1983 estimate to put the fire out vs the money Congress gave to move the town."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "PUT IT OUT, OR MOVE THE TOWN?")
    base_y, top_y = 900, 260
    for x, val, label, sub, col in ((650, 663, "$663 MILLION", "to try to put it out (1983 estimate, no guarantee)", RED),
                                    (1270, 42, "$42 MILLION", "to move the people (1984)", YEL)):
        h = (base_y - top_y) * val / 663
        d.rectangle((x - 140, base_y - h, x + 140, base_y), fill=col)
        d.text((x, base_y - h - 50), label, font=font(64), fill=col, anchor="mm")
        d.text((x, base_y + 50), sub, font=font(32, False), fill=BONE, anchor="mm")
    return img


def population(_=None):
    """Centralia: about 1,100 people in 1962, a handful today; ZIP code revoked in 2002."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "A TOWN DISAPPEARS")
    pts = [(1962, 1100), (1981, 1000), (2005, 12), (2010, 7)]
    x0, x1, yb, yt = 300, 1620, 880, 260
    X = lambda yr: x0 + (x1 - x0) * (yr - 1960) / 52
    Y = lambda p: yb - (yb - yt) * p / 1100
    d.line((x0, yb, x1, yb), fill=BONE, width=4)
    d.line([(X(y), Y(p)) for y, p in pts], fill=YEL, width=10)
    for y, p in pts:
        d.ellipse((X(y) - 14, Y(p) - 14, X(y) + 14, Y(p) + 14), fill=YEL)
        d.text((X(y), Y(p) - 50 if p > 50 else Y(p) - 50), f"{p:,}", font=font(52), fill=YEL, anchor="mm")
        d.text((X(y), yb + 45), str(y), font=font(40), fill=BONE, anchor="mm")
    d.line((X(2002), yb - 20, X(2002), yb - 330), fill=RED, width=4)
    d.text((X(2002) - 20, yb - 360), "2002: ZIP code 17927 revoked", font=font(38), fill=RED, anchor="rm")
    return img


def two_barrels(_=None):
    """1890s: liquor was taxed, soda wasn't; the soda barrels were painted red so agents could tell them apart."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "SAME BARREL. DIFFERENT TAX.")
    for cx, col, hoop, label, sub, subcol in ((640, (120, 82, 50), (60, 40, 25), "LIQUOR", "TAXED", RED),
                                             (1280, (196, 22, 28), (90, 10, 12), "SODA", "NOT TAXED", YEL)):
        top, bot, rw = 260, 820, 190
        d.rounded_rectangle((cx - rw, top, cx + rw, bot), radius=90, fill=col)
        d.ellipse((cx - rw - 18, 470, cx + rw + 18, 610), outline=hoop, width=0)
        for y in (330, 540, 750):
            d.line((cx - rw - 6, y, cx + rw + 6, y), fill=hoop, width=22)
        d.text((cx, bot + 60), label, font=font(64), fill=BONE, anchor="mm")
        d.text((cx, bot + 130), sub, font=font(52), fill=subcol, anchor="mm")
    return img


def red_timeline(_=None):
    """The red came first: red lettering (1886), red barrels (1890s), a red Santa in a White Rock ad (1915),
    Coca-Cola's Santa only in 1931."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "WHICH CAME FIRST?")
    ev = [(1886, "red letters on white", "the first signs", RED), (1895, "red barrels", "so tax agents could tell", RED),
          (1915, "a RED Santa", "White Rock water ad", YEL), (1931, "Coca-Cola's Santa", "16 years later", BONE)]
    x0, x1, y = 230, 1690, 560
    X = lambda yr: x0 + (x1 - x0) * (yr - 1880) / 55
    d.line((x0, y, x1, y), fill=BONE, width=6)
    for i, (yr, a, b, col) in enumerate(ev):
        x = X(yr)
        d.ellipse((x - 20, y - 20, x + 20, y + 20), fill=col)
        up = i % 2 == 0
        d.line((x, y - 20 if up else y + 20, x, y - 120 if up else y + 120), fill=col, width=4)
        ty = y - 250 if up else y + 170
        d.text((x, ty), "1890s" if yr == 1895 else str(yr), font=font(60), fill=col, anchor="mm")
        d.text((x, ty + 62), a, font=font(40), fill=BONE, anchor="mm")
        d.text((x, ty + 106), b, font=font(30, False), fill=BONE, anchor="mm")
    return img


def seasonal_hours(_=None):
    """Egypt: daylight split into 12 hours, so a summer hour was longer than a winter hour."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "AN HOUR WASN'T AN HOUR")
    for y, label, frac, col in ((380, "SUMMER DAY", 0.62, YEL), (700, "WINTER DAY", 0.42, BONE)):
        x0, w = 260, 1400 * frac
        d.text((x0, y - 90), label, font=font(52), fill=col, anchor="lm")
        for k in range(12):
            a = x0 + w * k / 12
            d.rectangle((a + 3, y - 40, a + w / 12 - 3, y + 40), fill=col)
        d.text((x0 + w + 30, y), "12 hours", font=font(44, False), fill=col, anchor="lm")
    d.text((W / 2, 960), "same 12 hours of daylight, different lengths", font=font(40, False), fill=BONE, anchor="mm")
    return img


def finger_count(_=None):
    """One theory: the thumb counts the 12 finger segments of one hand; the other hand counts the dozens."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "ONE THEORY: 12 x 5 = 60")
    cx0 = 560
    for f in range(4):  # four fingers, three segments each
        x = cx0 - 210 + f * 140
        for k in range(3):
            y = 300 + k * 120
            d.rounded_rectangle((x - 50, y, x + 50, y + 105), radius=30, outline=BONE, width=5)
            n = f * 3 + k + 1
            d.text((x, y + 52), str(n), font=font(48), fill=YEL, anchor="mm")
    d.text((cx0, 720), "thumb counts 12 segments", font=font(44, False), fill=BONE, anchor="mm")
    d.text((W / 2 + 120, 500), "x", font=font(120), fill=RED, anchor="mm")
    for f in range(5):
        x = 1250 + f * 95
        d.rounded_rectangle((x - 35, 330, x + 35, 640), radius=35, fill=YEL if f < 5 else BONE)
    d.text((1440, 720), "5 fingers keep score", font=font(44, False), fill=BONE, anchor="mm")
    d.text((W / 2, 900), "= 60", font=font(110), fill=YEL, anchor="mm")
    return img


def divisors(_=None):
    """60 splits evenly 12 ways; 100 only 9 ways, and a third of 100 never ends."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "WHY 60 BEATS 100")
    rows = ((330, "60", [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60], YEL), (610, "100", [1, 2, 4, 5, 10, 20, 25, 50, 100], BONE))
    for y, n, ds, col in rows:
        d.text((230, y), n, font=font(110), fill=col, anchor="mm")
        for i, v in enumerate(ds):
            x = 380 + i * 104
            d.ellipse((x - 44, y - 44, x + 44, y + 44), outline=col, width=5)
            d.text((x, y), str(v), font=font(38), fill=col, anchor="mm")
        d.text((380 + len(ds) * 104 - 20, y), f"{len(ds)} ways", font=font(48), fill=col, anchor="lm")
    d.text((W / 2, 820), "a third of an hour: 20 minutes", font=font(56), fill=YEL, anchor="mm")
    d.text((W / 2, 900), "a third of 100: 33.333...", font=font(56), fill=RED, anchor="mm")
    return img


def decimal_clock(_=None):
    """Revolutionary France, 1793-1795: 10 hours a day, 100 minutes an hour; 1 decimal hour = 2 h 24 min."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "FRANCE, 1793: A 10-HOUR DAY")
    cx, cy, r = 620, 590, 330
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(242, 238, 227), outline=YEL, width=10)
    for k in range(10):
        a = math.radians(-90 + 36 * k)
        d.text((cx + (r - 70) * math.cos(a), cy + (r - 70) * math.sin(a)), str(10 if k == 0 else k), font=font(64), fill=INK, anchor="mm")
    d.line((cx, cy, cx + 150 * math.cos(math.radians(-90 + 36 * 5)), cy + 150 * math.sin(math.radians(-90 + 36 * 5))), fill=INK, width=14)
    d.line((cx, cy, cx, cy - 230), fill=RED, width=6)
    for i, (t, c) in enumerate((("10 hours a day", YEL), ("100 minutes an hour", YEL), ("100 seconds a minute", YEL),
                                ("1 hour = 2 h 24 min of ours", RED), ("gone by 1795", BONE))):
        d.text((1080, 350 + i * 105), t, font=font(58), fill=c, anchor="lm")
    return img


def time_timeline(_=None):
    """How the hour got its 60 minutes: Babylon's base 60, Egypt's 12+12, Ptolemy's 'small parts', clocks."""
    img = base()
    d = ImageDraw.Draw(img)
    title(d, "4,000 YEARS TO MAKE A MINUTE")
    ev = [("~2000 BC", "Babylon counts in 60s", YEL), ("Egypt", "12 hours of day, 12 of night", YEL),
          ("~150 AD", "Ptolemy splits into 60ths", YEL), ("1175", "'minute' = small part", BONE),
          ("1656", "pendulum clock", BONE), ("~1690", "minute hands appear", RED)]
    x0, x1, y = 160, 1760, 560
    d.line((x0, y, x1, y), fill=BONE, width=6)
    for i, (a, b, col) in enumerate(ev):
        x = x0 + (x1 - x0) * (i + 0.5) / len(ev)
        d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=col)
        up = i % 2 == 0
        ty = y - 200 if up else y + 130
        d.line((x, y - 18 if up else y + 18, x, ty + (110 if up else -20)), fill=col, width=4)
        d.text((x, ty), a, font=font(54), fill=col, anchor="mm")
        d.text((x, ty + 60), b, font=font(32, False), fill=BONE, anchor="mm")
    return img


OVERLAYS = {"corner_crack": corner_crack, "rivet_crack": rivet_crack, "square_hole": square_hole}


DIAGRAMS = {"seasonal_hours": seasonal_hours, "finger_count": finger_count, "divisors": divisors, "decimal_clock": decimal_clock, "time_timeline": time_timeline, "two_barrels": two_barrels, "red_timeline": red_timeline, "underground_fire": underground_fire, "cost_compare": cost_compare, "population": population,
            "elevator_gap": elevator_gap, "otis_85": otis_85, "real_floors": real_floors, "tetraphobia": tetraphobia,
            "decompression": decompression, "corner_stress": corner_stress, "round_vs_square": round_vs_square, "altitude": altitude,
            "pressurisation": pressurisation, "crash_map": crash_map, "pressure_cycles": pressure_cycles,
            "fatigue_crack": fatigue_crack}
