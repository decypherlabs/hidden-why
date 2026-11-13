"""Point of Ruin and The State Maps: a script (studio/global_longform.py format) -> the finished episode, on GitHub
Actions (.github/workflows/kit.yml), in parallel parts (owner, 13 Nov 2026: "the whole on GitHub... in parallel as
much as we can").

    python kit/build.py prep  episodes/<slug>/script.json out/<slug>   # voice, stock clips, the timeline
    python kit/build.py part  out/<slug> <i> <n>                        # frames of part i of n -> part-<i>.mp4
    python kit/build.py join  out/<slug>                                # parts + the mix -> video.mp4, chapters.txt

Each shot's "kind" is drawn by the edit kit (kit/kit.py) or the map engine (kit/maps.py):
footage / archive (Pixabay clip, graded, label plate, source tag), still (the Kaggle painting shot-NN.png, pushed),
stat, headline, document, quote, chart (drawn), map (drawn). Kinetic captions on every shot, the wordmark, a chapter
card before each chapter after the first (none in the first minute), an end card. Voice: Kokoro am_adam at 0.95.
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kit  # noqa: E402
from kit import FPS, H, W, ease_out  # noqa: E402

GAP = 0.25  # seconds of air after each line
CARD = 2.2  # a chapter card
END = 8.0  # the end card (YouTube end screen elements sit on it)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
LOOK = {
    "point-of-ruin": {"name": "POINT OF RUIN", "bg": (20, 21, 24), "accent": (212, 90, 102), "tint": "#5B6770",
                      "headfont": ("PlayfairDisplay", "Black"), "style": "por", "paper": "#E9E2D0", "hl": "#C8862B"},
    "the-state-maps": {"name": "THE STATE MAPS", "bg": (14, 19, 23), "accent": (240, 138, 36), "tint": "#3E5566",
                       "headfont": ("BarlowCondensed-Bold", None), "style": "tsm", "paper": "#E3E0D6", "hl": "#F08A24"},
}
BONE = (216, 210, 196)


def shots_of(sc):
    out, n = [], 0
    for si, sec in enumerate(sc["sections"]):
        for k, x in enumerate(sec.get("shots") or []):
            n += 1
            out.append({**x, "n": n, "section": sec["id"], "si": si, "first": k == 0, "card": sec.get("card") if k == 0 else None})
    return out


def look_of(sc):
    return LOOK["the-state-maps" if sc["slug"].startswith("tsm-") else "point-of-ruin"]


# ---------------------------------------------------------------- prep
def voice(shots, out):
    from kokoro_onnx import Kokoro
    m = Path(os.environ.get("KOKORO_DIR", "models"))
    k = Kokoro(str(m / "kokoro-v1.0.onnx"), str(m / "voices-v1.0.bin"))
    import soundfile as sf
    durs = {}
    for x in shots:
        dest = out / f"vo-{x['n']:03d}.wav"
        if not dest.exists():
            a, sr = k.create(x["vo"], voice="am_adam", speed=0.95, lang="en-us")
            sf.write(dest, a, sr)
        info = sf.info(str(dest))
        durs[f"{x['n']:03d}"] = info.frames / info.samplerate
        print(f"voice {x['n']}: {durs[f'{x['n']:03d}']:.1f}s", flush=True)
    return durs


def pixabay(q, used):
    key = os.environ.get("PIXABAY_KEY", "")
    u = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode({"key": key, "q": q[:100], "per_page": 30, "safesearch": "true"})
    hits = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()).get("hits") or []
    real = [h for h in hits if "ai generated" not in h["tags"].lower() and "ai-generated" not in h["tags"].lower()]
    fresh = [h for h in real if h["id"] not in used] or real
    return fresh[0] if fresh else None


def fetch(h, dest):
    vs = sorted((v for v in h["videos"].values() if v.get("url")), key=lambda v: v["width"])
    v = next((v for v in vs if v["width"] >= 1920), vs[-1])
    with urllib.request.urlopen(urllib.request.Request(v["url"], headers=UA), timeout=300) as r:
        dest.write_bytes(r.read())
    return {"source": "Pixabay", "id": h["id"], "page": h["pageURL"], "user": h["user"],
            "licence": "Pixabay Content License", "duration": h["duration"]}


def stock(shots, out):
    """One clip per footage/archive shot (and per still with no painting), never the same clip twice in a row; a
    search with no hit is tried again with fewer words."""
    used, log = set(), {}
    (out / "stock").mkdir(exist_ok=True)
    for x in shots:
        q = None
        if x.get("kind") in ("footage", "archive"):
            q = x.get("search") or ""
            if x["kind"] == "archive":
                q = f"{q} vintage"
        elif x.get("kind") == "still" and not (out / f"shot-{x['n']:02d}.png").exists():
            q = " ".join((x.get("prompt") or "").replace(",", " ").split()[:5])
        if not q:
            continue
        dest = out / "stock" / f"{x['n']:03d}.mp4"
        if dest.exists():
            continue
        words = q.split()
        h = None
        for cut in (len(words), 3, 2, 1):
            try:
                h = pixabay(" ".join(words[:cut]), used)
            except Exception as e:  # noqa: BLE001
                print(f"pixabay {q}: {e}", flush=True)
            if h:
                break
        if not h:
            print(f"no clip for shot {x['n']} ({q})", flush=True)
            continue
        log[str(x["n"])] = {**fetch(h, dest), "query": q, "tags": h.get("tags", "")}
        used.add(h["id"])
        print(f"clip {x['n']}: {q} -> {h['id']}", flush=True)
    (out / "stock" / "sources.json").write_text(json.dumps(log, indent=1))


def prep(script_path, out):
    out.mkdir(parents=True, exist_ok=True)
    sc = json.loads(Path(script_path).read_text())
    shots = shots_of(sc)
    durs = voice(shots, out)
    stock(shots, out)
    t, items, chapters = 0.0, [], []
    for x in shots:
        if x["card"] and x["si"] >= 2:  # a chapter card from the second chapter on (nothing in the first minute)
            items.append({"kind": "card", "text": x["card"], "start": t, "dur": CARD})
            t += CARD
        if x["first"] and x["card"]:
            chapters.append((t, x["card"].title()))
        d = durs[f"{x['n']:03d}"] + GAP
        items.append({**x, "start": t, "dur": d})
        t += d
    items.append({"kind": "end", "start": t, "dur": END})
    t += END
    if not chapters or chapters[0][0] > 0:
        chapters.insert(0, (0.0, "Intro"))
    (out / "timeline.json").write_text(json.dumps({"slug": sc["slug"], "look": look_of(sc), "sec": t, "items": items,
                                                   "chapters": chapters}, indent=1))
    print(f"timeline: {t / 60:.1f} min, {len(items)} items", flush=True)


# ---------------------------------------------------------------- drawing
class Ctx:
    def __init__(self, out):
        self.out = out
        tl = json.loads((out / "timeline.json").read_text())
        self.items, self.sec, self.L = tl["items"], tl["sec"], tl["look"]
        self.clips, self.cache = {}, {}

    def item_at(self, t):
        for it in self.items:
            if it["start"] <= t < it["start"] + it["dur"]:
                return it
        return self.items[-1]


def backdrop(L, i):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = L["bg"]
    return img


def footage(ctx, it, t, i, n):
    p = ctx.out / "stock" / f"{it['n']:03d}.mp4"
    if not p.exists():
        return None
    key = it["n"]
    if key not in ctx.clips:
        ctx.clips = {key: kit.Clip(p, 0.0, it["dur"] + 1)}  # one open clip at a time (memory)
    rgb = ctx.clips[key].frame(i, n, drift=1 if it["n"] % 2 else -1)
    if rgb is None:
        return None
    rgb = kit.grade(rgb, sat=0.55 if it.get("kind") == "archive" else 0.8, tint=ctx.L["tint"])
    if it.get("kind") == "archive":  # duotone-ish archive treatment
        g = rgb.mean(axis=2, keepdims=True)
        rgb = np.clip(g * np.array([1.05, 0.98, 0.86]), 0, 255).astype(np.uint8)
    return rgb


def still(ctx, it, t, i, n):
    p = ctx.out / f"shot-{it['n']:02d}.png"
    if not p.exists():
        return footage(ctx, it, t, i, n)
    if ("still", it["n"]) not in ctx.cache:
        im = Image.open(p).convert("RGB")
        im = im.resize((int(W * 1.08), int(H * 1.08)), Image.LANCZOS)
        ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != "still"}
        ctx.cache[("still", it["n"])] = np.asarray(im)
    src = ctx.cache[("still", it["n"])]
    u = i / max(1, n - 1)
    z = 1.0 + 0.06 * u
    cw, ch = int(W * 1.08 / z), int(H * 1.08 / z)
    x0 = (src.shape[1] - cw) // 2
    y0 = (src.shape[0] - ch) // 2
    import cv2
    return kit.grade(cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR), tint=ctx.L["tint"])


def stat(ctx, it, t, i, n):
    s = it.get("stat") or {}
    img = backdrop(ctx.L, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    v = s.get("value")
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = None
    c = ease_out(t / 0.8)
    pre, suf = str(s.get("prefix") or ""), str(s.get("suffix") or "")
    if v is not None:
        dec = 0 if abs(v) >= 100 or float(v).is_integer() else 1
        txt = pre + f"{v * c:,.{dec}f}" + (" " + suf if suf and suf[0].isalpha() else suf)
    else:
        txt = pre + str(s.get("value", "")) + suf
    f = kit.font("BarlowCondensed-Bold", 230)
    tw = d.textlength(txt, font=f)
    a = ease_out(t / 0.25)
    d.text((W / 2 - tw / 2, H / 2 - 210), txt, font=f, fill=(*ctx.L["accent"], int(255 * a)))
    lab = str(s.get("label") or "").upper()
    lf = kit.font("Inter", 40)
    d.line([(W / 2 - 300 * c, H / 2 + 70), (W / 2 + 300 * c, H / 2 + 70)], fill=(*ctx.L["accent"], int(255 * a)), width=3)
    y = H / 2 + 100
    for ln in wrap(d, lab, lf, W - 400)[:3]:  # a long label wraps instead of running off the frame
        lw = sum(d.textlength(ch, font=lf) + 40 * 0.08 for ch in ln)
        kit.spaced(d, (W / 2 - lw / 2, y), ln, lf, (*BONE, int(255 * ease_out((t - 0.3) / 0.4))))
        y += 56
    return kit.over(img, layer)


def paper_card(ctx, it, t, key, dateline, headline, hl):
    if (key, it["n"]) not in ctx.cache:
        ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != key}
        ctx.cache[(key, it["n"])] = kit.Clipping(dateline, headline, hl, paper=ctx.L["paper"], accent=ctx.L["hl"],
                                                 style=ctx.L["style"], seed=it["n"])
    c = ctx.cache[(key, it["n"])]
    img = backdrop(ctx.L, 0)
    layer = kit.blank()
    p = kit.back_out(t / (10 / FPS))
    kit.place(layer, c.render(t), (W / 2, H / 2 + (1 - p) * 300), -2 + 2 * (it["n"] % 3), (1.06 - 0.06 * min(1, t / 0.4)) * (1 + 0.02 * t / max(it["dur"], 1)))
    return kit.over(img, layer)


def headline(ctx, it, t, i, n):
    h = it.get("headline") or {}
    text = str(h.get("text") or it.get("vo") or "")
    words = text.split()
    return paper_card(ctx, it, t, "headline", str(h.get("date") or ""), text, " ".join(words[:2]) if words else None)


def document(ctx, it, t, i, n):
    d = it.get("document") or {}
    hl = str(d.get("highlight") or "")
    title = str(d.get("title") or "")
    title = title if len(title) <= 62 else title[:60].rsplit(" ", 1)[0] + "…"  # the typed dateline fits one line
    return paper_card(ctx, it, t, "document", title, hl or str(d.get("title") or ""),
                      " ".join(hl.split()[:3]) if hl else None)


def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        nxt = (cur + " " + w).strip()
        if d.textlength(nxt, font=f) <= maxw:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def quote(ctx, it, t, i, n):
    q = it.get("quote") or {}
    img = backdrop(ctx.L, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    text = f"“{q.get('text') or it.get('vo')}”"
    f = kit.font("SourceSerif4", 64)
    lines = wrap(d, text, f, W - 520)
    y = H / 2 - len(lines) * 42
    shown = int(len(text) * min(1.0, t / max(0.6, it["dur"] * 0.45)))
    k = 0
    for ln in lines:
        part = ln[:max(0, shown - k)]
        k += len(ln) + 1
        d.text((260, y), part, font=f, fill=(*BONE, 255))
        y += 84
    a = ease_out((t - 0.8) / 0.4)
    who = " · ".join(x for x in (str(q.get("who") or ""), str(q.get("source") or "")) if x).upper()
    d.line([(260, y + 20), (360, y + 20)], fill=(*ctx.L["accent"], int(255 * a)), width=3)
    kit.spaced(d, (260, y + 44), who[:90], kit.font("Inter", 26), (*BONE, int(200 * a)))
    return kit.over(img, layer)


def chart(ctx, it, t, i, n):
    c = it.get("chart") or {}
    pts = [(str(p[0]), float(p[1])) for p in c.get("points") or [] if isinstance(p, (list, tuple)) and len(p) == 2
           and isinstance(p[1], (int, float))]
    img = backdrop(ctx.L, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    kit.spaced(d, (180, 120), str(c.get("title") or "").upper(), kit.font("Inter", 46), (*BONE, 255))
    if c.get("unit"):
        kit.spaced(d, (180, 186), str(c["unit"]).upper(), kit.font("Inter", 24), (150, 156, 160, 255))
    if pts:
        x0, x1, y0, y1 = 220, W - 220, 300, H - 200
        hi = max(v for _, v in pts) or 1
        p = ease_out(t / 1.4)
        lf = kit.font("Inter", 24)
        mark = str(c.get("mark") or "")
        if (c.get("type") or "bar") == "line":
            xy = [(x0 + (x1 - x0) * k / max(1, len(pts) - 1), y1 - (y1 - y0) * v / hi) for k, (_, v) in enumerate(pts)]
            m = max(2, int(len(xy) * p + 0.999))
            d.line(xy[:m], fill=(*BONE, 255), width=5)
            for (lab, v), (x, y) in zip(pts, xy[:m]):
                col = (226, 58, 46) if lab == mark else BONE
                d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(*col, 255))
                d.text((x, y1 + 20), lab, font=lf, fill=(150, 156, 160, 255), anchor="mt")
        else:
            bw = (x1 - x0) / len(pts) * 0.62
            for k, (lab, v) in enumerate(pts):
                cx = x0 + (x1 - x0) * (k + 0.5) / len(pts)
                hgt = (y1 - y0) * v / hi * ease_out((t - 0.12 * k) / 0.8)
                col = (226, 58, 46) if lab == mark else ctx.L["accent"]
                d.rectangle((cx - bw / 2, y1 - hgt, cx + bw / 2, y1), fill=(*col, 255))
                d.text((cx, y1 + 20), lab, font=lf, fill=(150, 156, 160, 255), anchor="mt")
                if hgt > 0:
                    d.text((cx, y1 - hgt - 14), f"{v:,.4g}", font=lf, fill=(*BONE, 255), anchor="mb")
        d.line([(x0, y1), (x1, y1)], fill=(*BONE, 120), width=2)
    if c.get("source"):
        kit.source_tag(d, f"Source: {c['source']}"[:110])
    return kit.over(img, layer)


def mapshot(ctx, it, t, i, n):
    import maps
    return maps.frame(it.get("map") or {}, t, it["dur"], i)


def card(ctx, it, t, i, n):
    img = backdrop(ctx.L, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    a = ease_out(t / 0.3) * (1 - ease_out((t - it["dur"] + 0.3) / 0.3))
    name, var = ctx.L["headfont"]
    f = kit.font(name, 110, var)
    txt = str(it["text"]).upper()
    tw = d.textlength(txt, font=f)
    d.text((W / 2 - tw / 2, H / 2 - 80), txt, font=f, fill=(*BONE, int(255 * a)))
    rw = 160 * ease_out(t / 0.5)
    d.line([(W / 2 - rw, H / 2 + 70), (W / 2 + rw, H / 2 + 70)], fill=(*ctx.L["accent"], int(255 * a)), width=4)
    return kit.over(img, layer)


def endcard(ctx, it, t, i, n):
    img = backdrop(ctx.L, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    a = ease_out(t / 0.5)
    name, var = ctx.L["headfont"]
    f = kit.font(name, 84, var)
    tw = d.textlength(ctx.L["name"], font=f)
    d.text((W / 2 - tw / 2, 150), ctx.L["name"], font=f, fill=(*BONE, int(255 * a)))
    for box in ((260, 380, 900, 740), (1020, 380, 1660, 740)):  # where YouTube's end-screen videos go
        d.rectangle(box, outline=(*ctx.L["accent"], int(140 * a)), width=2)
    return kit.over(img, layer)


DRAW = {"footage": footage, "archive": footage, "still": still, "stat": stat, "headline": headline,
        "document": document, "quote": quote, "chart": chart, "map": mapshot, "card": card, "end": endcard}


def plate_ok(ctx, it):
    """A place label goes on a stock clip only when the clip's own Pixabay tags name a word of that place (13 Nov 2026:
    a Hong Kong cruise ship came out labelled "Bandar Abbas"); a plate with no place in it (a year, a phrase) stays."""
    if "sources" not in ctx.cache:
        ctx.cache["sources"] = json.loads((ctx.out / "stock" / "sources.json").read_text()) if (ctx.out / "stock" / "sources.json").exists() else {}
    tags = str((ctx.cache["sources"].get(str(it["n"])) or {}).get("tags", "")).lower()
    words = [w.strip(".,·:;()").lower() for w in str(it["plate"]).replace("·", " ").split()]
    names = [w for w in words if w.isalpha() and len(w) > 3 and w[0].isalpha()]
    caps = [w for w in str(it["plate"]).replace("·", " ").split() if w[:1].isupper()]
    if not caps:  # no proper name in the label: nothing to get wrong
        return True
    return any(w in tags for w in names)


def render_frame(ctx, fi):
    t = fi / FPS
    it = ctx.item_at(t)
    lt = t - it["start"]
    n = max(1, int(round(it["dur"] * FPS)))
    li = int(round(lt * FPS))
    try:
        rgb = DRAW.get(it.get("kind"), stat)(ctx, it, lt, li, n)
    except Exception as e:  # noqa: BLE001 - one bad element never stops the episode: its shot falls back to a card
        print(f"shot {it.get('n')} ({it.get('kind')}): {e}", flush=True)
        rgb = None
    if rgb is None:
        rgb = backdrop(ctx.L, fi)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    if it.get("vo"):
        kit.captions(layer, it["vo"].split(), lt, it["dur"] - GAP, accent=ctx.L["accent"])
    if it.get("kind") in ("footage", "archive") and it.get("plate") and plate_ok(ctx, it):
        kit.label_plate(layer, (90, 110), str(it["plate"])[:48], lt)
    if it.get("kind") in ("footage", "archive"):
        kit.source_tag(d, "Pixabay")
    if it.get("kind") not in ("card", "end"):
        kit.watermark(d, ctx.L["name"])
    return kit.finish(kit.over(rgb, layer), fi)


def part(out, k, n):
    ctx = Ctx(out)
    total = int(ctx.sec * FPS)
    a, b = total * k // n, total * (k + 1) // n
    w = kit.Writer(out / f"part-{k:02d}.mp4")
    for fi in range(a, b):
        w.write(render_frame(ctx, fi))
        if (fi - a) % 300 == 0:
            print(f"part {k}: {(fi - a) / FPS:.0f}s of {(b - a) / FPS:.0f}s", flush=True)
    w.close()


# ---------------------------------------------------------------- join
def join(out):
    out = out.resolve()  # ffmpeg runs inside out/: every path absolute (13 Nov 2026: the first join looked in out/out/)
    tl = json.loads((out / "timeline.json").read_text())
    parts = sorted(out.glob("part-*.mp4"))
    (out / "parts.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(out / "parts.txt"), "-c", "copy",
                    str(out / "silent.mp4")], check=True, cwd=out)
    mix = kit.Mix(tl["sec"])
    items = tl["items"]
    whoosh, tick, thud, swipe = kit.synth("whoosh"), kit.synth("tick"), kit.synth("thud"), kit.synth("swipe")
    page, typing = kit.sfx("page"), kit.sfx("typing")
    prev = None
    for it in items:
        k, at = it.get("kind"), it["start"]
        if it.get("vo"):
            mix.vo(kit.load_audio(out / f"vo-{it['n']:03d}.wav"), at)
        # owner, 13 Nov 2026 ("SFX are also less"): a sound on every change of picture, and one per element
        if prev is not None and k != "card" and (k != prev.get("kind") or k in ("footage", "archive", "still")):
            mix.add(whoosh, max(0, at - 0.18), 0.32)
        if k in ("headline", "document"):
            mix.add(thud, at + 0.2, 0.55)
            mix.add(page, at + 0.25, 0.55)
            mix.add(typing[: int(kit.SR * 0.8)], at + 0.35, 0.3)
            mix.add(swipe, at + 1.0, 0.4)
        elif k == "stat":
            for j in range(8):  # the count-up ticks, then a hit
                mix.add(tick, at + 0.05 + j * 0.09, 0.45)
            mix.add(thud, at + 0.8, 0.5)
        elif k == "chart":
            mix.add(swipe, at + 0.1, 0.4)
            for j in range(len((it.get("chart") or {}).get("points") or [])):
                mix.add(tick, at + 0.15 + j * 0.12, 0.35)
        elif k == "quote":
            mix.add(typing[: int(kit.SR * 1.5)], at + 0.1, 0.22)
        elif k == "map":
            act = (it.get("map") or {}).get("action") or "locate"
            mix.add(kit.synth("riser", 2.0) if act == "locate" else whoosh, at, 0.25)
            first = 2.6 if act == "locate" else 0.4
            for j in range(min(6, len((it.get("map") or {}).get("places") or []))):
                mix.add(tick, at + first + 0.25 * j, 0.5)  # each pin or ring lands
            if act in ("route", "flow", "measure"):
                mix.add(swipe, at + first, 0.45)
        elif k == "card":
            mix.add(kit.synth("riser", 1.4), max(0, at - 1.2), 0.45)
            mix.add(thud, at, 0.75)
        elif k == "end":
            mix.add(kit.synth("riser", 1.6), max(0, at - 1.4), 0.3)
        prev = it
    # music by section (owner, 13 Nov 2026: the one reflective bed all through was "not suiting"): tension under the
    # cold open, a different licensed bed per chapter, each faded in and out, levelled to one loudness, a short drop
    # of silence before every chapter card
    beds = ["quiz-suspense", "mixkit-silent-descent-614", "mixkit-river-flow-594", "mixkit-skyline-601",
            "mixkit-epical-drums-04-679"]
    spans, cur = [], None
    for it in items:
        sec = it.get("section") or ("_end" if it.get("kind") == "end" else cur)
        if it.get("kind") == "card":
            sec = f"_card{it['start']}"
        if not spans or spans[-1][0] != sec:
            spans.append([sec, it["start"], it["start"] + it["dur"]])
        else:
            spans[-1][2] = it["start"] + it["dur"]
        cur = it.get("section") or cur
    drops = [(it["start"] - 0.6, it["start"]) for it in items if it.get("kind") == "card"]
    k = 0
    for sec, a0, a1 in spans:
        if str(sec).startswith("_card"):
            continue
        name = beds[0] if sec == "cold_open" else "mixkit-silent-descent-614" if sec in ("outro", "_end") else beds[1 + k % 4]
        if sec not in ("cold_open", "outro", "_end"):
            k += 1
        bed = kit.load_audio(HERE / "music" / f"{name}.mp3")
        rms = float(np.sqrt(np.mean(bed ** 2))) or 0.1
        mix.music(bed * (0.12 / rms), a0, a1, gain=0.24, duck=0.38, fade=1.2, silence=drops)
    mix.write(out / "mix.wav")
    # one real encode at the end (13 Nov 2026: the near-lossless parts concatenated made 2.5 GB for 10 minutes)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "silent.mp4"), "-i", str(out / "mix.wav"),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-maxrate", "12M", "-bufsize", "24M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
                    str(out / "video.mp4")], check=True)
    (out / "chapters.txt").write_text("\n".join(f"{int(s // 60):02d}:{int(s % 60):02d} {name}" for s, name in tl["chapters"]))
    print(f"video {tl['sec'] / 60:.1f} min -> {out / 'video.mp4'}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], Path(sys.argv[3]))
    elif cmd == "part":
        part(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "join":
        join(Path(sys.argv[2]))
