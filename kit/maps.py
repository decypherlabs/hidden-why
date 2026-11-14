"""The State Maps' "Slate Atlas" map shots, drawn from a script's map spec (kit/build.py): dark slate land, steel-blue
neighbours, the story's places in muted orange, every marker in red (owner, 11 Nov 2026). Generalised from the
studio's experiments/style-test/slate_map.py (the Hormuz test the owner liked).

    frame(spec, t, dur, i) -> RGB array, t seconds into a shot of dur seconds
spec: {"action": locate|measure|route|fill|border_change|pin|flow, "center": [lat, lon], "span": <degrees tall>,
       "places": [{"name", "lat", "lon"}], "label": "..", "years": [..], "disputed": bool}
"""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
OCEAN, LAND, BONE, ORANGE, STEEL = (14, 19, 23), (38, 45, 52), (216, 210, 196), (240, 138, 36), (79, 127, 166)
RED = (226, 58, 46)
WORLD = json.loads((REPO / "brand" / "maps" / "world.json").read_text())
_PL = REPO / "brand" / "maps" / "places.json"  # Natural Earth cities and seas (public domain): the map's other names
PLACES = json.loads(_PL.read_text()) if _PL.exists() else {"cities": [], "seas": []}
for c in WORLD["countries"]:
    a = np.concatenate([np.array(r) for r in c["rings"]])
    c["bbox"] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
_F = {}

# Satellite/terrain base (12 Nov 2026): Natural Earth NE1_50M_SR_W (public domain, 10800x5400, brand/maps/LICENSES.md),
# graded for legibility and sampled with the same projection as the outlines. TERRAIN = False (or MAPS_TERRAIN=0)
# brings back the flat slate drawing.
TERRAIN = os.environ.get("MAPS_TERRAIN", "1") != "0"  # the modern look: plates, motion, pins (any style below)
# the base under it (owner, 13 Nov 2026: "the first image is better... make it realistic - real color - blue for water"):
# "terrain" = the shaded-relief raster; "atlas" / "natural" / "light" = the crisp drawn map in real colours
STYLE = os.environ.get("MAPS_STYLE", "atlas")
PALETTES = {  # RGB: ocean, land, coast line, border line, focus, neighbour
    "atlas": ((70, 136, 190), (234, 224, 196), (52, 98, 140), (255, 255, 255), (232, 96, 44), (120, 160, 200)),
    "natural": ((46, 102, 160), (190, 210, 160), (34, 74, 118), (250, 250, 240), (226, 88, 52), (232, 200, 120)),
    "light": ((158, 198, 226), (246, 243, 236), (110, 150, 180), (170, 170, 170), (214, 48, 40), (120, 150, 190)),
}
_TER = []


def _grade(im):
    """Natural Earth's relief is pale: more colour and contrast on land, a deeper blue sea, so labels and fills read."""
    for y in range(0, im.shape[0], 540):
        f = im[y:y + 540].astype(np.float32) / 255
        r, b = f[..., 0], f[..., 2]  # RGB
        sea = np.clip((b - r - 0.05) / 0.10, 0, 1)[..., None]
        gray = f.mean(2, keepdims=True)
        land = np.clip(((gray + (f - gray) * 1.55) - 0.5) * 1.25 + 0.5 - 0.08, 0, 1) ** 1.05 * np.array([0.90, 0.89, 0.78], np.float32)
        deep = np.clip((gray + (f - gray) * 1.3) * np.array([0.38, 0.58, 0.78], np.float32), 0, 1)
        im[y:y + 540] = (np.clip(land * (1 - sea) + deep * sea, 0, 1) * 255 + 0.5).astype(np.uint8)


def _terrain_levels():
    if not _TER:
        im = cv2.cvtColor(cv2.imread(str(REPO / "brand" / "maps" / "terrain_10800.jpg")), cv2.COLOR_BGR2RGB)
        _grade(im)
        _TER.append(im)
        for _ in range(3):
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            _TER.append(im)
    return _TER


_GR = []


def _grain():
    if not _GR:
        _GR.append(np.random.RandomState(7).randint(0, 7, (H, W, 1), np.uint8).repeat(3, 2))
    return _GR[0]


def terrain(lat0, lon0, span, cosl=None, cy=None):
    """The raster under the view (W x H, RGB): a pure affine sample, lon/lat -> pixel is linear in this projection."""
    lv = _terrain_levels()
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
        img = cv2.subtract(cv2.add(img, _grain()), 3)
    return img


def inter(size):
    if size not in _F:
        _F[size] = ImageFont.truetype(str(REPO / "kit" / "fonts" / "Inter.ttf"), size)
    return _F[size]


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _inside(c, lon, lat):
    b = c["bbox"]
    if not (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
        return False
    return any(cv2.pointPolygonTest(np.float32(r), (float(lon), float(lat)), False) >= 0 for r in c["rings"])


def _countries_of(places):
    out = []
    for p in places or []:
        for c in WORLD["countries"]:
            if _inside(c, p["lon"], p["lat"]):
                if c["iso"] not in out:
                    out.append(c["iso"])
                break
    return out


_CACHE = {}
_LAYOUT, _REC = {}, None  # the labels of a shot, placed once at its final view (frame -> _record)


def _record(spec, dur, i):
    global _REC
    _REC = []
    try:
        frame(spec, dur * 0.97, dur, i)
        _LAYOUT[json.dumps(spec, sort_keys=True) + f"|{dur:.2f}"] = _REC
    finally:
        _REC = None


def _focus(spec):
    key = json.dumps(spec, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = _countries_of(spec.get("places"))
    return _CACHE[key]


def base(lat0, lon0, span, focus=(), fill=1.0):
    k, cosl = H / span, np.cos(np.radians(lat0))
    P = lambda lo, la: (W / 2 + (lo - lon0) * cosl * k, H / 2 - (la - lat0) * k)
    pts = lambda r: np.int32([[x * 4, y * 4] for x, y in (P(lo, la) for lo, la in r)])
    hw, hh = W / (2 * k * cosl) + 2, H / (2 * k) + 2
    vis = lambda b: not (b[2] < lon0 - hw or b[0] > lon0 + hw or b[3] < lat0 - hh or b[1] > lat0 + hh)
    shown = [c for c in WORLD["countries"] if vis(c["bbox"])]
    if TERRAIN and STYLE in PALETTES:  # crisp drawn map in real colours, sharp at any zoom
        ocean, land, coast, border, foc, nb = PALETTES[STYLE]
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = ocean
        for c in shown:  # each country a touch lighter or darker, like an atlas
            j = (sum(map(ord, c["iso"])) % 7 - 3) * 3
            cv2.fillPoly(img, [pts(r) for r in c["rings"]], tuple(int(np.clip(v + j, 0, 255)) for v in land), cv2.LINE_AA, shift=2)
        if focus and fill > 0:
            layer = img.copy()
            for c in shown:
                if c["iso"] in focus:
                    cv2.fillPoly(layer, [pts(r) for r in c["rings"]], foc if c["iso"] == focus[0] else nb, cv2.LINE_AA, shift=2)
            img = cv2.addWeighted(layer, 0.62 * fill, img, 1 - 0.62 * fill, 0)
        for c in shown:  # the coast a darker blue, the borders a fine light line
            cv2.polylines(img, [pts(r) for r in c["rings"]], True, coast, 2, cv2.LINE_AA, shift=2)
        layer = img.copy()
        for c in shown:
            cv2.polylines(layer, [pts(r) for r in c["rings"]], True, border, 1, cv2.LINE_AA, shift=2)
        return cv2.addWeighted(layer, 0.7, img, 0.3, 0), P
    if TERRAIN:
        img = terrain(lat0, lon0, span, cosl)
        if focus and fill > 0:
            layer = img.copy()
            for c in shown:
                if c["iso"] in focus:
                    cv2.fillPoly(layer, [pts(r) for r in c["rings"]], ORANGE if c["iso"] == focus[0] else STEEL, cv2.LINE_AA, shift=2)
            img = cv2.addWeighted(layer, 0.45 * fill, img, 1 - 0.45 * fill, 0)
        layer = img.copy()
        for c in shown:
            cv2.polylines(layer, [pts(r) for r in c["rings"]], True, (255, 255, 255), 1, cv2.LINE_AA, shift=2)
        return cv2.addWeighted(layer, 0.42, img, 0.58, 0), P
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = OCEAN
    step = 2 if span < 30 else 10 if span < 90 else 20
    for g in range(-180, 181, step):
        if abs(g) <= 85:
            cv2.line(img, tuple(map(int, P(-180, g))), tuple(map(int, P(180, g))), (34, 39, 43), 1, cv2.LINE_AA)
        cv2.line(img, tuple(map(int, P(g, -85))), tuple(map(int, P(g, 85))), (34, 39, 43), 1, cv2.LINE_AA)
    for c in shown:
        cv2.fillPoly(img, [pts(r) for r in c["rings"]], LAND, cv2.LINE_AA, shift=2)
    if focus and fill > 0:
        layer = img.copy()
        for c in shown:
            if c["iso"] in focus:
                cv2.fillPoly(layer, [pts(r) for r in c["rings"]], ORANGE if c["iso"] == focus[0] else STEEL, cv2.LINE_AA, shift=2)
        img = cv2.addWeighted(layer, 0.42 * fill, img, 1 - 0.42 * fill, 0)
    layer = img.copy()
    for c in shown:
        cv2.polylines(layer, [pts(r) for r in c["rings"]], True, BONE, 1, cv2.LINE_AA, shift=2)
    img = cv2.addWeighted(layer, 0.5, img, 0.5, 0)
    return img, P


def _spaced(d, xy, text, f, fill, a=1.0, track=0.08, halo=True):
    x, y = xy
    if halo:
        for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            xx = x
            for ch in text:
                d.text((xx + ox, y + oy), ch, font=f, fill=(10, 12, 14, int(200 * a)))
                xx += d.textlength(ch, font=f) + f.size * track
    for ch in text:
        d.text((x, y), ch, font=f, fill=(*fill, int(255 * a)))
        x += d.textlength(ch, font=f) + f.size * track


def _dashed(img, a, b, p, color=RED, width=4):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    m = max(1, int(L * p // 22))
    for i in range(0, m, 2):
        p0 = (int(a[0] + (b[0] - a[0]) * p * i / m), int(a[1] + (b[1] - a[1]) * p * i / m))
        p1 = (int(a[0] + (b[0] - a[0]) * p * (i + 1) / m), int(a[1] + (b[1] - a[1]) * p * (i + 1) / m))
        cv2.line(img, p0, p1, color, width, cv2.LINE_AA)


def _tw(d, text, f, track=0.1):
    return sum(d.textlength(ch, font=f) + f.size * track for ch in text) - f.size * track


def _plate(d, x, y, text, f, a=1.0, track=0.1, fg=(255, 255, 255)):
    """A name on a small solid black plate, white caps, letter-spaced (RealLifeLore / Wendover look)."""
    w, h = _tw(d, text, f, track), f.size
    px, py = 11, 7
    d.rectangle((x, y, x + w + 2 * px, y + h + 2 * py), fill=(6, 8, 10, int(238 * a)))
    xx = x + px
    for ch in text:
        d.text((xx, y + py - 2), ch, font=f, fill=(*fg, int(255 * a)))
        xx += d.textlength(ch, font=f) + f.size * track


def _hit(r, taken):
    return any(r[0] < b[2] and r[2] > b[0] and r[1] < b[3] and r[3] > b[1] for b in taken)


def _country_spots(shown, P, focus, limit=12):
    """Where each country's name goes: the deepest point of the part that is on screen (a low-res mask + distance
    transform), so it sits inside the shape, not on the coast."""
    sc, mw, mh = 8, W // 8, H // 8
    out = []
    for c in shown:
        m = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(m, [np.int32([[x / sc * 4, y / sc * 4] for x, y in (P(lo, la) for lo, la in r)]) for r in c["rings"]], 255, shift=2)
        m[:12], m[-14:], m[:, :6], m[:, -6:] = 0, 0, 0, 0
        if not m.any():
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        _, mx, _, loc = cv2.minMaxLoc(dt)
        if mx * sc >= 26:
            out.append((c["iso"] in focus, mx, c, (loc[0] * sc + sc / 2, loc[1] * sc + sc / 2)))
    out.sort(key=lambda o: (not o[0], -o[1]))
    return out[:limit]


def frame(spec, t, dur, i=0):
    places = [p for p in spec.get("places") or [] if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))]
    if TERRAIN:  # a whole country is shown by its fill and its name, not a pin in its middle (13 Nov 2026)
        _cn = {c["name"].upper() for c in WORLD["countries"]}
        places = [p for p in places if str(p.get("name", "")).upper() not in _cn] or places[:1]
    lat0, lon0 = (spec.get("center") or [places[0]["lat"], places[0]["lon"]] if places else [20, 0])[:2]
    span = float(spec.get("span") or 20)
    action = spec.get("action") or "locate"
    u = t / max(dur, 0.1)
    te = min(3.4, dur * 0.6)
    if True:  # every map: the whole region first, eased in log space toward the detail (owner, 13 Nov 2026)
        e = ease(t / te)
        s0 = min(150.0, span * 6)
        span_now = float(np.exp(np.log(s0) * (1 - e) + np.log(span) * e))
        if TERRAIN:  # never still: a slow push once there
            span_now *= 1 - 0.07 * ease((t - te) / max(dur - te, 0.5))
    else:
        span_now = span * ((1.05 - 0.07 * ease(u)) if TERRAIN else (1.04 - 0.04 * ease(u)))  # a slow push
    if TERRAIN:  # and a slow drift sideways
        lon0 = lon0 + 0.05 * span_now / max(np.cos(np.radians(lat0)), 0.2) * (u - 0.5)
    focus = _focus(spec)
    img, P = base(lat0, lon0, span_now, focus, fill=ease((t - te * 0.6) / 1.0))
    pts = [P(p["lon"], p["lat"]) for p in places]
    t0 = te * 0.8  # routes, pins, rings once the camera is there
    reveal = ease((t - t0) / 1.2)
    na_ = ease((t - 1.4) / 0.8)
    taken = []
    if action in ("route", "flow") and len(pts) > 1 and reveal > 0:
        seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
        todo = sum(seg) * reveal
        for (a, b), L in zip(zip(pts, pts[1:]), seg):
            if todo <= 0:
                break
            f = min(1.0, todo / max(L, 1))
            todo -= L
            if TERRAIN:
                _dashed(img, a, b, f, (8, 10, 12), 8)
            _dashed(img, a, b, f)
            if action == "flow" and f >= 1:
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                hd = [(int(b[0]), int(b[1])), (int(b[0] - 26 * math.cos(ang - 0.45)), int(b[1] - 26 * math.sin(ang - 0.45))),
                      (int(b[0] - 26 * math.cos(ang + 0.45)), int(b[1] - 26 * math.sin(ang + 0.45)))]
                cv2.fillPoly(img, [np.int32(hd)], RED, cv2.LINE_AA)
    if action == "measure" and len(pts) > 1 and reveal > 0:
        if TERRAIN:
            _dashed(img, pts[0], pts[1], reveal, (8, 10, 12), 7)
        _dashed(img, pts[0], pts[1], reveal, BONE if not TERRAIN else (255, 255, 255), 3)
    pulse = None
    if action in ("locate", "pin", "fill", "border_change", "measure") and reveal > 0:
        for k, q in enumerate(pts[:6]):
            r = ease((t - t0 - 0.25 * k) / 0.6)
            if r <= 0:
                continue
            if k == 0 and action == "locate":
                if TERRAIN:
                    cv2.ellipse(img, tuple(map(int, q)), (60, 60), -90, 0, 360 * r, (8, 10, 12), 7, cv2.LINE_AA)
                cv2.ellipse(img, tuple(map(int, q)), (60, 60), -90, 0, 360 * r, RED, 3 if not TERRAIN else 4, cv2.LINE_AA)
                if r >= 1 and TERRAIN:
                    pulse = (q, ((t - t0 - 0.6) % 1.9) / 1.9)
                taken.append((q[0] - 64, q[1] - 64, q[0] + 64, q[1] + 64))
            else:
                dq = (q[0], q[1] - (1 - r) * 120) if TERRAIN else q  # the pin drops in
                c = tuple(map(int, dq))
                if TERRAIN:
                    cv2.circle(img, c, int(11 * r), (8, 10, 12), -1, cv2.LINE_AA)
                    cv2.circle(img, c, int(18 * r), (8, 10, 12), 5, cv2.LINE_AA)
                cv2.circle(img, c, int(9 * r), RED, -1, cv2.LINE_AA)
                cv2.circle(img, c, int(16 * r), RED, 2, cv2.LINE_AA)
                taken.append((q[0] - 24, q[1] - 24, q[0] + 24, q[1] + 24))
    for q in pts[:6]:
        if action in ("route", "flow"):
            taken.append((q[0] - 20, q[1] - 20, q[0] + 20, q[1] + 20))
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    if pulse:
        (qx, qy), ph = pulse
        rr = 62 + 70 * ph
        d.ellipse((qx - rr, qy - rr, qx + rr, qy + rr), outline=(*RED, int(210 * (1 - ph))), width=4)
    lab = spec.get("label")
    la = ease((t - te) / 0.6)
    if lab and la > 0:
        f = inter(40)
        tw = d.textlength(str(lab).upper(), font=f) + len(str(lab)) * 3.2 + 40
        taken.append((90, 120, 90 + tw, 192))
    if TERRAIN:
        # every label's place is worked out once per shot, at the final view, and pinned to its spot on the map, so the
        # names ride with the zoom instead of hopping between spots (owner, 13 Nov 2026: "the labels dance")
        lkey = json.dumps(spec, sort_keys=True) + f"|{dur:.2f}"
        cosl0, k0 = np.cos(np.radians(lat0)), H / span_now

        def _L(kind, cls, d, x, y, text, f, a, **kw):
            if _REC is not None:
                _REC.append({"kind": kind, "cls": cls, "lon": lon0 + (x - W / 2) / (cosl0 * k0), "lat": lat0 - (y - H / 2) / k0,
                             "text": text, "size": f.size if f else 0, "kw": kw})
            if kind == "plate":
                _plate(d, x, y, text, f, a)
            elif kind == "spaced":
                _spaced(d, (x, y), text, f, kw["fill"], a, track=kw.get("track", 0.08))
            else:
                d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255, int(255 * a)), outline=(20, 20, 20, int(255 * a)), width=2)
        if _REC is None and lkey not in _LAYOUT:
            _record(spec, dur, i)
        if _REC is None and lkey in _LAYOUT:
            arrive = ease((t - te * 0.85) / 0.5)  # names appear as the camera arrives, not over the wide view
            alpha = {"place": reveal, "country": ease((t - 1.0) / 0.8), "sea": na_ * 0.85, "city": na_ * 0.9}
            for lb in _LAYOUT[lkey]:
                x, y = P(lb["lon"], lb["lat"])
                a = arrive * (alpha.get(lb["cls"], 1.0) if lb["cls"] != "country" else alpha["country"] * lb["kw"].get("k", 1.0))
                if a <= 0:
                    continue
                f = inter(lb["size"]) if lb["size"] else None
                if lb["kind"] == "plate":
                    _plate(d, x, y, lb["text"], f, a)
                elif lb["kind"] == "spaced":
                    _spaced(d, (x, y), lb["text"], f, lb["kw"]["fill"], a, track=lb["kw"].get("track", 0.08))
                else:
                    d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255, int(255 * a)), outline=(20, 20, 20, int(255 * a)), width=2)
        else:
            # names: the story's places first (next to their markers), then the focus countries, then the biggest neighbours
            fP, fC = inter(30), inter(24)
            if reveal > 0:
                cnames = {c["name"].upper() for c in WORLD["countries"]}
                for q, p in list(zip(pts, places))[:6]:
                    text = str(p.get("name", "")).upper()
                    if not text or text in cnames:  # a country is named by the country labels (13 Nov 2026: IRAN twice)
                        continue
                    w, h = _tw(d, text, fP) + 22, fP.size + 14
                    cands = [(q[0] + 30, q[1] - h / 2), (q[0] - 30 - w, q[1] - h / 2), (q[0] - w / 2, q[1] - 34 - h), (q[0] - w / 2, q[1] + 34),
                             (q[0] + 30, q[1] - h - 20), (q[0] + 30, q[1] + 20), (q[0] - 30 - w, q[1] - h - 20), (q[0] - 30 - w, q[1] + 20),
                             (q[0] - w / 2, q[1] - 90 - h), (q[0] - w / 2, q[1] + 90)]
                    for x, y in cands:
                        r = (x - 6, y - 6, x + w + 6, y + h + 6)
                        if 10 < x and x + w < W - 10 and 10 < y and y + h < H - 10 and not _hit(r, taken):
                            break
                    else:
                        x, y = cands[0]
                        r = (x - 6, y - 6, x + w + 6, y + h + 6)
                    _L('plate', 'place', d, x, y, text, fP, reveal)
                    taken.append(r)
            ca = ease((t - 1.0) / 0.8) * ease((120 - span_now) / 30)
            if ca > 0:
                hwd, hhd = W / (2 * (H / span_now) * np.cos(np.radians(lat0))) + 2, span_now / 2 + 2
                shown = [c for c in WORLD["countries"] if not (c["bbox"][2] < lon0 - hwd or c["bbox"][0] > lon0 + hwd or c["bbox"][3] < lat0 - hhd or c["bbox"][1] > lat0 + hhd)]
                for isfocus, mx, c, (cx, cy) in _country_spots(shown, P, focus):
                    f = inter(30) if isfocus else fC
                    text = c["name"].upper()
                    w, h = _tw(d, text, f) + 22, f.size + 14
                    for dx, dy in ((0, 0), (0, h + 10), (0, -h - 10), (w * 0.3, 0), (-w * 0.3, 0), (0, 2 * (h + 10)), (0, -2 * (h + 10))):
                        x, y = cx - w / 2 + dx, cy - h / 2 + dy
                        r = (x - 6, y - 6, x + w + 6, y + h + 6)
                        if 10 < x and x + w < W - 10 and 10 < y and y + h < H - 10 and not _hit(r, taken) and w < mx * 8 * 1.9 + 60:
                            _L('plate', 'country', d, x, y, text, f, ca * (1.0 if isfocus else 0.85), k=1.0 if isfocus else 0.85)
                            taken.append(r)
                            break
            # the other names in view (owner, 13 Nov 2026: "if the major areas visible are also mapped"): seas and gulfs on
            # the water, then the biggest cities, as many as the zoom allows and never over another label
            na = ease((t - 1.4) / 0.8)
            if na > 0:
                hwd, hhd = W / (2 * (H / span_now) * np.cos(np.radians(lat0))), span_now / 2
                inview = lambda x: abs(x["lon"] - lon0) < hwd * 0.92 and abs(x["lat"] - lat0) < hhd * 0.9
                told = {str(p.get("name", "")).upper() for p in places}
                sea_rank = 1 if span_now > 60 else 3 if span_now > 20 else 6
                for sea in sorted((x for x in PLACES["seas"] if inview(x) and x["rank"] <= sea_rank), key=lambda x: x["rank"])[:4]:
                    text = sea["name"].upper()
                    if text in told:
                        continue
                    f = inter(26 if span_now > 20 else 22)
                    x0s, y0s = P(sea["lon"], sea["lat"])
                    w = _tw(d, text, f) + 10
                    for dx, dy in ((0, 0), (0, 40), (0, -40), (60, 20), (-60, 20), (0, 80), (0, -80), (90, 0), (-90, 0)):
                        x, y = x0s + dx, y0s + dy
                        r = (x - w / 2 - 6, y - 18, x + w / 2 + 6, y + 18)
                        if 10 < r[0] and r[2] < W - 10 and 10 < r[1] and r[3] < H - 10 and not _hit(r, taken):
                            _L('spaced', 'sea', d, x - w / 2, y - 14, text, f, na * 0.85, fill=(230, 240, 250), track=0.18)
                            taken.append(r)
                            break
                city_rank = 2 if span_now > 60 else 4 if span_now > 25 else 6 if span_now > 10 else 8
                cities = sorted((x for x in PLACES["cities"] if inview(x) and x["rank"] <= city_rank and x["name"].upper() not in told),
                                key=lambda x: (not x["capital"], x["rank"], -x["pop"]))
                fc, n_c = inter(20), 0
                for cty in cities:
                    if n_c >= 8:
                        break
                    x, y = P(cty["lon"], cty["lat"])
                    text = cty["name"].upper()
                    w, h = _tw(d, text, fc) + 18, fc.size + 10
                    for lx, ly in ((x + 12, y - h / 2), (x - 12 - w, y - h / 2), (x - w / 2, y - h - 12), (x - w / 2, y + 12)):
                        r = (min(lx, x - 8) - 4, min(ly, y - 8) - 4, max(lx + w, x + 8) + 4, max(ly + h, y + 8) + 4)
                        if 10 < r[0] and r[2] < W - 10 and 10 < r[1] and r[3] < H - 10 and not _hit(r, taken):
                            _L('dot', 'city', d, x, y, '', None, na)
                            _L('plate', 'city', d, lx, ly, text, fc, na * 0.9)
                            taken.append(r)
                            n_c += 1
                            break
    elif reveal > 0:
        for q, p in list(zip(pts, places))[:6]:
            _spaced(d, (q[0] + 26, q[1] - 16), str(p.get("name", "")).upper(), inter(26), BONE, reveal)
    if lab and la > 0:
        f = inter(40)
        tw = d.textlength(str(lab).upper(), font=f) + len(str(lab)) * 3.2 + 40
        x0, y0 = 90, 120
        d.rectangle((x0, y0, x0 + tw, y0 + 72), fill=(10, 12, 14, int(225 * la)))
        d.line((x0, y0, x0 + tw, y0), fill=(*ORANGE, int(255 * la)), width=3)
        _spaced(d, (x0 + 20, y0 + 14), str(lab).upper(), f, BONE, la, halo=False)
    yrs = [y for y in spec.get("years") or [] if isinstance(y, (int, float))]
    if action == "border_change" and yrs:
        k = min(len(yrs) - 1, int(u * len(yrs)))
        _spaced(d, (W - 330, 110), str(int(yrs[k])), ImageFont.truetype(str(REPO / "kit" / "fonts" / "BarlowCondensed-Bold.ttf"), 110),
                ORANGE, 1.0, halo=False)
    if spec.get("disputed"):
        _spaced(d, (90, H - 120), "DISPUTED TERRITORY · BORDERS SHOWN ARE NOT AN ENDORSEMENT", inter(18), (150, 156, 160) if not TERRAIN else (255, 255, 255), 1.0)
    return np.asarray(im)
