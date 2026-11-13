"""The State Maps' "Slate Atlas" map shots, drawn from a script's map spec (kit/build.py): dark slate land, steel-blue
neighbours, the story's places in muted orange, every marker in red (owner, 11 Nov 2026). Generalised from the
studio's experiments/style-test/slate_map.py (the Hormuz test the owner liked).

    frame(spec, t, dur, i) -> RGB array, t seconds into a shot of dur seconds
spec: {"action": locate|measure|route|fill|border_change|pin|flow, "center": [lat, lon], "span": <degrees tall>,
       "places": [{"name", "lat", "lon"}], "label": "..", "years": [..], "disputed": bool}
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
OCEAN, LAND, BONE, ORANGE, STEEL = (14, 19, 23), (38, 45, 52), (216, 210, 196), (240, 138, 36), (79, 127, 166)
RED = (226, 58, 46)
WORLD = json.loads((REPO / "brand" / "maps" / "world.json").read_text())
for c in WORLD["countries"]:
    a = np.concatenate([np.array(r) for r in c["rings"]])
    c["bbox"] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
_F = {}


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


def frame(spec, t, dur, i=0):
    places = [p for p in spec.get("places") or [] if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))]
    lat0, lon0 = (spec.get("center") or [places[0]["lat"], places[0]["lon"]] if places else [20, 0])[:2]
    span = float(spec.get("span") or 20)
    action = spec.get("action") or "locate"
    u = t / max(dur, 0.1)
    if action == "locate":  # the whole region first, eased in log space toward the place
        e = ease(t / min(3.4, dur * 0.6))
        s0 = min(150.0, span * 6)
        span_now = float(np.exp(np.log(s0) * (1 - e) + np.log(span) * e))
    else:
        span_now = span * (1.04 - 0.04 * ease(u))  # a slow push
    focus = _focus(spec)
    img, P = base(lat0, lon0, span_now, focus, fill=ease((t - (2.0 if action == "locate" else 0.3)) / 1.0))
    pts = [P(p["lon"], p["lat"]) for p in places]
    reveal = ease((t - (2.6 if action == "locate" else 0.4)) / 1.2)
    if action in ("route", "flow") and len(pts) > 1 and reveal > 0:
        seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
        todo = sum(seg) * reveal
        for (a, b), L in zip(zip(pts, pts[1:]), seg):
            if todo <= 0:
                break
            f = min(1.0, todo / max(L, 1))
            todo -= L
            _dashed(img, a, b, f)
            if action == "flow" and f >= 1:
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                hd = [(int(b[0]), int(b[1])), (int(b[0] - 26 * math.cos(ang - 0.45)), int(b[1] - 26 * math.sin(ang - 0.45))),
                      (int(b[0] - 26 * math.cos(ang + 0.45)), int(b[1] - 26 * math.sin(ang + 0.45)))]
                cv2.fillPoly(img, [np.int32(hd)], RED, cv2.LINE_AA)
    if action == "measure" and len(pts) > 1 and reveal > 0:
        _dashed(img, pts[0], pts[1], reveal, BONE, 3)
    if action in ("locate", "pin", "fill", "border_change", "measure") and reveal > 0:
        for k, q in enumerate(pts[:6]):
            r = ease((t - (2.6 if action == "locate" else 0.4) - 0.25 * k) / 0.6)
            if r <= 0:
                continue
            if k == 0 and action == "locate":
                cv2.ellipse(img, tuple(map(int, q)), (60, 60), -90, 0, 360 * r, RED, 3, cv2.LINE_AA)
            else:
                cv2.circle(img, tuple(map(int, q)), int(9 * r), RED, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(map(int, q)), int(16 * r), RED, 2, cv2.LINE_AA)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    if reveal > 0:
        for q, p in list(zip(pts, places))[:6]:
            _spaced(d, (q[0] + 26, q[1] - 16), str(p.get("name", "")).upper(), inter(26), BONE, reveal)
    lab = spec.get("label")
    la = ease((t - (3.2 if action == "locate" else 0.8)) / 0.6)
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
        _spaced(d, (90, H - 120), "DISPUTED TERRITORY · BORDERS SHOWN ARE NOT AN ENDORSEMENT", inter(18), (150, 156, 160), 1.0)
    return np.asarray(im)
