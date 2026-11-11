"""Words in the pictures, set by the edit (owner, 11 Nov 2026: "FLUX actually garbles the text", then after the trial
in experiments/text-render: "perfect job - we'll do this during edit now... as much as we can, so that pictures add
more sense"). A script with "words_in_edit": true keeps writing its words the old way, quoted in the prompt
('a shop sign reading "BIG BAZAAR"'), and:

  gpu/paint.py         lift(prompt)[0]: the painter gets the prompt with every quote taken out and the surface blank
  studio/place_words   finds each blank surface in the painting (Claude's rough corners, refined on the pixels) and
                       stores a grid of points per text in the shot: "words": {"12-2": {"src": md5, "items": [...]}}
  pipeline/build.py    apply(img, items): the exact words set in a real font, bent onto that grid, multiplied
                       into the surface with ink as strong and soft as the surface is sharp

Kept import-light (re only at the top): the painter imports lift() on Kaggle.
"""
import re

QUOTE = re.compile(r"""\s*(?:,\s*)?(?:that\s+|which\s+)?(?:reading|reads|read|titled|saying|says|labell?ed|marked|printed with|"""
                   r"""with the words?|with the text|headline)?\s*[:\-]?\s*["“]([^"”]{1,60})["”]""", re.I)
# what the older prompts asked FLUX to print; with the words in the edit every surface is painted blank instead
ASKED = (", only the words the prompt quotes printed crisp and correctly spelled",
         "only the words the prompt quotes printed crisp and correctly spelled, ",
         ", only the quoted words printed, large, crisp and spelled exactly letter by letter; every other sign, board, "
         "label, can, newspaper and banner plain and blank",
         ", every sign, board, label, can, newspaper and banner plain and blank, no lettering or text anywhere")
BLANK = (", every sign, board, label, page, document, screen and package completely plain and blank, a clean empty "
         "surface with no letters, words or numbers anywhere")


def lift(prompt):
    """(the painter's prompt, [(the words, the surface they sit on)]): 'a red shop sign reading "BIG BAZAAR" over
    the door' -> 'a red shop sign left blank over the door, ...blank' and [("BIG BAZAAR", "a red shop sign")]."""
    found = []
    for m in QUOTE.finditer(prompt):
        before = re.split(r"[,;.]", prompt[:m.start()])[-1].split()
        found.append((m.group(1).strip(), " ".join(before[-8:])))
    p = QUOTE.sub(" left blank", prompt) if found else prompt
    for a in ASKED:
        p = p.replace(a, "")
    return p.rstrip(". ") + BLANK, found


DEVANAGARI = re.compile(r"[ऀ-ॿ]")
FONTS = {  # style -> (Latin, Devanagari or anything beyond Latin-1, e.g. ₹); all OFL, in brand/. Serif by default
    # (owner, 11 Nov 2026: "serif would be better for the painting text"); sans only for a modern sign
    "print": ("fonts/OldStandard-Bold.ttf", "fonts/Martel-Bold.ttf"),  # newspapers, documents, labels
    "sign": ("fonts/Martel-ExtraBold.ttf", "fonts/Martel-ExtraBold.ttf"),  # painted boards, shop fronts, gates
    "modern": ("logos-and-legends/fonts/Poppins-ExtraBold.ttf", "logos-and-legends/fonts/Mukta-ExtraBold.ttf"),
    "hand": ("fonts/Kalam-Bold.ttf", "fonts/Kalam-Bold.ttf"),  # chalk boards, notes, a register
    "type": ("logos-and-legends/fonts/SpecialElite-Regular.ttf", "devlok-katha/fonts/TiroDevanagariHindi-Regular.ttf"),
}


def font_for(text, style):
    from pathlib import Path
    latin, wide = FONTS.get(style) or FONTS["sign"]
    rel = wide if any(ord(c) > 0xFF for c in text) else latin
    return str(Path(__file__).resolve().parent.parent / "brand" / rel)


def _set_flat(text, font, w, h, color, pad=0.08):
    """The words at the largest size that fits w x h, broken into the 1-3 lines that fit biggest, centred."""
    from PIL import Image, ImageDraw, ImageFont, features
    lay = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC  # raqm shapes Devanagari
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words = text.split()
    best = None
    for n in range(1, min(3, len(words)) + 1):
        lines = [" ".join(words[round(i * len(words) / n):round((i + 1) * len(words) / n)]) for i in range(n)]
        size = int(h * (1 - 2 * pad) / n)
        while size > 8:
            f = ImageFont.truetype(font, size, layout_engine=lay)
            boxes = [d.textbbox((0, 0), ln, font=f) for ln in lines]
            bh = sum(b[3] - b[1] for b in boxes) + size * 0.25 * (n - 1)
            if max(b[2] - b[0] for b in boxes) <= w * (1 - 2 * pad) and bh <= h * (1 - 2 * pad):
                break
            size -= 2
        if not best or size > best[0]:
            best = (size, lines, f, boxes, bh)
    size, lines, f, boxes, bh = best
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = (h - bh) / 2
    for ln, b in zip(lines, boxes):
        d.text((w / 2, y - b[1]), ln, font=f, anchor="ma", fill=color)
        y += b[3] - b[1] + size * 0.25
    return img


def _outline(grid):
    import numpy as np
    return np.int32([*grid[0], *grid[1:, -1], *grid[-1, ::-1][1:], *grid[::-1, 0][1:]])


def sharpness(rgb, grid):
    """0 (soft, far, in haze) .. 1 (crisp, close): the Laplacian's spread on the surface's own pixels."""
    import cv2
    import numpy as np
    g = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    area = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(area, [_outline(grid)], 255)
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.5), cv2.CV_32F)
    return float(np.clip(np.sqrt(lap[area > 0].var()) / 12, 0, 1)) if (area > 0).any() else 0.5


def _warp_grid(flat, grid, size):
    """The flat words onto the surface cell by cell, each cell its own perspective (a curling page, a spine), the
    words shared out by each row's and column's real size."""
    import cv2
    import numpy as np
    R, C = grid.shape[:2]
    fh, fw = flat.shape[:2]
    ys = np.concatenate([[0], np.cumsum(np.linalg.norm(grid[1:] - grid[:-1], axis=2).mean(1))])
    xs = np.concatenate([[0], np.cumsum(np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2).mean(0))])
    ys, xs = ys / ys[-1] * fh, xs / xs[-1] * fw
    out = np.zeros((size[1], size[0], 4), np.float32)
    for r in range(R - 1):
        for c in range(C - 1):
            src = np.float32([[xs[c], ys[r]], [xs[c + 1], ys[r]], [xs[c + 1], ys[r + 1]], [xs[c], ys[r + 1]]])
            dst = np.float32([grid[r, c], grid[r, c + 1], grid[r + 1, c + 1], grid[r + 1, c]])
            cell = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), size, flags=cv2.INTER_AREA)
            m = np.zeros((size[1], size[0]), np.uint8)
            cv2.fillConvexPoly(m, np.int32(np.round(dst)), 1)
            m = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)  # no hairline seams between cells
            out[m] = np.maximum(out[m], cell[m])
    return out


def clean(rgb, grid):
    """Wipe the stray squiggles a model still paints on a 'blank' surface: pixels darker than the surface around
    them, inside the grid, are filled in from their neighbours."""
    import cv2
    import numpy as np
    u8 = (rgb * 255).astype(np.uint8)
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    area = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(area, [_outline(grid)], 255)
    dark = ((g < cv2.medianBlur(u8, 21)[..., 1].astype(np.float32) - 18) & (area > 0)).astype(np.uint8) * 255
    if not dark.any():
        return rgb
    dark = cv2.dilate(dark, np.ones((5, 5), np.uint8))
    return cv2.inpaint(u8, dark, 6, cv2.INPAINT_TELEA).astype(np.float32) / 255


def hexrgb(c):
    c = str(c or "#191614").lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) == 6 else (25, 22, 20)


def place(rgb, item):
    """One text onto the painting (rgb float 0..1). item: {"text", "grid": rows x cols of [x, y], "style", "color",
    optional "opacity", "blur"}."""
    import cv2
    import numpy as np
    grid = np.float32(item["grid"])
    R, C = grid.shape[:2]
    w = int(max(np.linalg.norm(grid[0, c + 1] - grid[0, c]) for c in range(C - 1)) * (C - 1))
    h = int(max(np.linalg.norm(grid[r + 1, 0] - grid[r, 0]) for r in range(R - 1)) * (R - 1))
    if w < 12 or h < 6:
        return rgb
    rgb = clean(rgb, grid)
    s = sharpness(rgb, grid)
    opacity = float(item.get("opacity") or round(0.62 + 0.25 * s, 2))  # ink is never solid: fainter on a soft surface
    blur = float(item.get("blur") or round(0.5 + 1.1 * (1 - s), 2))  # and as soft-edged as the surface it sits on
    flat = np.array(_set_flat(item["text"], font_for(item["text"], item.get("style")), w * 2, h * 2,
                              hexrgb(item.get("color")) + (255,))).astype(np.float32) / 255
    rng = np.random.default_rng(5)
    wear = cv2.resize(rng.random((max(2, h // 12), max(2, w // 12))).astype(np.float32), (w * 2, h * 2))
    flat[..., 3] *= 0.8 + 0.2 * wear  # uneven ink: a press and a brush both leave some letters lighter
    H, W = rgb.shape[:2]
    warped = cv2.GaussianBlur(_warp_grid(flat, grid, (W, H)), (0, 0), blur)
    a = warped[..., 3:4] * opacity
    return rgb * (1 - a) + rgb * warped[..., :3] * a  # multiply: the surface's light and shadow stay on the letters


def apply(bgr, items):
    """A painting as OpenCV reads it (BGR uint8) with its words set."""
    import cv2
    import numpy as np
    if bgr is None or not items:
        return bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    for it in items:
        try:
            rgb = place(rgb, it)
        except Exception as e:  # noqa: BLE001 - a bad grid costs the words, never the video
            print(f"  words '{it.get('text')}' not set: {e}", flush=True)
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def table(script):
    """{painting key ("12", "12-2"): items} from a script's shots."""
    out, n = {}, 0
    for sec in script.get("sections") or []:
        for s in sec.get("shots") or []:
            n += 1
            for k, v in (s.get("words") or {}).items():
                if isinstance(v, dict) and v.get("items"):
                    out[k] = v["items"]
    return out
