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


DIAGRAMS = {"corner_stress": corner_stress, "round_vs_square": round_vs_square}
