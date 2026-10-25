"""Video 01 thumbnail art: an unmistakable AIRPLANE window (wing + engine outside, half-pulled shade,
curved cabin wall), drawn at 2x and downsampled.

    python3 videos/01-round-windows/make_art.py   ->  art.png (1280x720)
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S = 2
W, H = 1280 * S, 720 * S


def grad(w, h, top, bottom):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line((0, y, w, y), fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return img


def main():
    # cabin wall: warm dark gray, lit from the window, with panel seams
    img = grad(W, H, (58, 56, 52), (30, 29, 27))
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((W * 0.40, -H * 0.1, W * 1.1, H * 1.1), fill=170)
    img.paste(Image.new("RGB", (W, H), (120, 116, 108)), mask=glow.filter(ImageFilter.GaussianBlur(160 * S)))
    d = ImageDraw.Draw(img)
    for x in (int(W * 0.47), int(W * 0.99)):
        d.line((x, 0, x, H), fill=(24, 23, 22), width=3 * S)

    # window opening (outer bezel, inner frame, glass)
    ox0, oy0, ox1, oy1 = int(W * 0.53), int(H * 0.06), int(W * 0.88), int(H * 0.94)
    ix0, iy0, ix1, iy1 = ox0 + 34 * S, oy0 + 36 * S, ox1 - 34 * S, oy1 - 36 * S
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle((ox0 + 14 * S, oy0 + 20 * S, ox1 + 14 * S, oy1 + 20 * S), radius=200 * S, fill=200)
    img.paste((10, 10, 10), mask=shadow.filter(ImageFilter.GaussianBlur(30 * S)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((ox0, oy0, ox1, oy1), radius=200 * S, fill=(214, 209, 196))
    d.rounded_rectangle((ox0 + 18 * S, oy0 + 18 * S, ox1 - 18 * S, oy1 - 18 * S), radius=180 * S, fill=(128, 124, 116))

    gw, gh = ix1 - ix0, iy1 - iy0
    glass = grad(gw, gh, (36, 104, 196), (126, 182, 236))
    g = ImageDraw.Draw(glass)
    # cloud deck far below
    g.rectangle((0, int(gh * 0.70), gw, gh), fill=(236, 240, 246))
    for cx, cy, r in ((0.15, 0.70, 0.18), (0.45, 0.68, 0.22), (0.80, 0.71, 0.2), (1.0, 0.69, 0.16)):
        g.ellipse((gw * (cx - r), gh * (cy - r * 0.35), gw * (cx + r), gh * (cy + r * 0.35)), fill=(240, 243, 248))
    # the wing: from lower left, sweeping back to the right, with an engine underneath and a winglet
    wing = [(-0.10, 0.80), (1.05, 0.50), (1.05, 0.585), (-0.10, 0.98)]
    g.polygon([(gw * x, gh * y) for x, y in wing], fill=(188, 192, 198))
    g.polygon([(gw * x, gh * y) for x, y in [(-0.10, 0.80), (1.05, 0.50), (1.05, 0.52), (-0.10, 0.83)]], fill=(232, 234, 238))
    g.polygon([(gw * x, gh * y) for x, y in [(-0.10, 0.93), (1.05, 0.57), (1.05, 0.585), (-0.10, 0.98)]], fill=(140, 144, 152))
    g.polygon([(gw * x, gh * y) for x, y in [(0.86, 0.555), (0.93, 0.36), (0.97, 0.36), (0.95, 0.55)]], fill=(176, 180, 188))  # winglet
    # engine nacelle hanging under the wing
    ex, ey, er = gw * 0.30, gh * 0.83, gw * 0.13
    g.rounded_rectangle((ex - er * 1.6, ey - er * 0.62, ex + er * 0.9, ey + er * 0.62), radius=int(er * 0.6), fill=(150, 154, 162))
    g.ellipse((ex - er * 1.9, ey - er * 0.62, ex - er * 1.0, ey + er * 0.62), fill=(70, 74, 82))
    g.ellipse((ex - er * 1.78, ey - er * 0.48, ex - er * 1.12, ey + er * 0.48), fill=(38, 40, 46))
    g.polygon([(ex - er * 0.2, ey - er * 0.6), (ex + er * 0.4, ey - er * 0.6), (ex + er * 0.3, ey - er * 1.1), (ex - er * 0.1, ey - er * 1.0)],
              fill=(160, 164, 172))  # pylon
    # sun glare
    glare = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(glare).polygon([(gw * 0.55, 0), (gw * 0.72, 0), (gw * 0.38, gh), (gw * 0.25, gh)], fill=60)
    glass.paste((255, 255, 255), mask=glare.filter(ImageFilter.GaussianBlur(26 * S)))
    # half-pulled window shade with a finger tab
    g = ImageDraw.Draw(glass)
    sh = int(gh * 0.26)
    g.rectangle((0, 0, gw, sh), fill=(226, 221, 208))
    for k in range(1, 6):
        g.line((0, sh * k / 6, gw, sh * k / 6), fill=(210, 204, 190), width=2 * S)
    g.rounded_rectangle((gw * 0.40, sh - 8 * S, gw * 0.60, sh + 14 * S), radius=10 * S, fill=(196, 190, 176))
    mask = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, gw, gh), radius=160 * S, fill=255)
    img.paste(glass, (ix0, iy0), mask)
    img = img.resize((1280, 720), Image.LANCZOS)
    img.save(HERE / "art.png")
    print(HERE / "art.png")


if __name__ == "__main__":
    main()
