"""Story of Giants and The State Maps: a script (studio/global_longform.py format) -> the finished episode, on GitHub
Actions (.github/workflows/kit.yml), in parallel parts (owner, 13 Nov 2026: "the whole on GitHub... in parallel as
much as we can").

    python kit/build.py prep  episodes/<slug>/script.json out/<slug>   # voice, stock clips, the timeline
    python kit/build.py part  out/<slug> <i> <n>                        # frames of part i of n -> part-<i>.mp4
    python kit/build.py join  out/<slug>                                # parts + the mix -> video.mp4, chapters.txt

Each shot's "kind" is drawn by the edit kit (kit/kit.py) or the map engine (kit/maps.py):
footage / archive (Pixabay clip, graded, label plate, source tag), still (the Kaggle painting shot-NN.png, pushed),
stat, headline, document, quote, chart (drawn), map (drawn). Kinetic captions on every shot, the wordmark, a chapter
card before each chapter after the first (none in the first minute), an end card. Voice: Kokoro am_michael at 1.15 (owner, 13 Nov 2026).
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
# the edit (owner, 13 Nov 2026: "the videos feel very plain going... the hooks has to be some different, edit,
# effects"): the hook cuts every HOOK_BEAT s (each painting in wide / close / medium framings, a flash and a hit on every
# cut, the line's number or year slammed on screen), a title slam after it, and no picture held past BODY_BEAT s after
HOOK_BEAT, BODY_BEAT, TITLE = 3.0, 4.0, 2.6
XF = 8  # frames of crossfade on every scene change (owner, 13 Nov 2026: "no transitions during scene change")  # owner, 13 Nov 2026: 1.6 s cuts, flashes and hits read as noise
BIG = __import__("re").compile(r"\$\s?\d[\d,.]*(?:\s?(?:billion|million|thousand|bn|m)\b)?|\b\d[\d,.]*\s?(?:%|percent|billion|million|"
                               r"kilometres|kilometers|km|miles|barrels|stores|ships)\b|\b(?:1[5-9]|20)\d\d\b", 2)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
LOOK = {
    "point-of-ruin": {"name": "STORY OF GIANTS", "bg": (20, 21, 24), "accent": (212, 90, 102), "tint": "#5B6770",
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
            a, sr = k.create(x["vo"], voice="am_michael", speed=1.15, lang="en-us")  # owner, 13 Nov 2026: Michael at 1.15 on every English channel
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


def photos(shots, out):
    """The real photographs studio/kit_visuals.py picked (Wikimedia Commons / Pixabay), saved as photos/NNN.jpg."""
    (out / "photos").mkdir(parents=True, exist_ok=True)
    for x in shots:
        ph = x.get("photo") or {}
        dest = out / "photos" / f"{x['n']:03d}.jpg"
        if x.get("kind") != "photo" or not ph.get("src") or dest.exists():
            continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(ph["src"], headers={"User-Agent": "EvolveBetterStudio/1.0 (decypherlabs.io@gmail.com)"}),
                                          timeout=60).read()
            Image.open(__import__("io").BytesIO(data)).convert("RGB").save(dest, quality=92)
        except Exception as e:  # noqa: BLE001 - the shot falls back to its painting, else a card
            print(f"photo {x['n']}: {e}", flush=True)


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
        if x.get("clip"):  # the clip Claude picked for this line on the Mac (studio/kit_visuals.py): exactly that one
            try:
                u = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode({"key": os.environ.get("PIXABAY_KEY", ""), "id": x["clip"]["id"]})
                h = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())["hits"][0]
                log[str(x["n"])] = {**fetch(h, dest), "query": "picked", "tags": h.get("tags", "")}
                used.add(h["id"])
                continue
            except Exception as e:  # noqa: BLE001 - falls back to a search
                print(f"picked clip {x['clip']} for shot {x['n']}: {e}", flush=True)
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
    photos(shots, out)
    t, items, chapters = 0.0, [], []
    title = str(sc.get("intro_title") or sc.get("title") or "").split(":")[0].split(",")[0].split("|")[0].strip()
    for x in shots:
        if x["si"] == 1 and x["first"] and title:  # the title slam closes the hook, before chapter 1
            items.append({"kind": "title", "text": title[:40], "start": t, "dur": TITLE, "section": "title"})
            t += TITLE
        if x["card"] and x["si"] >= 2:  # a chapter card from the second chapter on (nothing in the first minute)
            items.append({"kind": "card", "text": x["card"], "start": t, "dur": CARD})
            t += CARD
        if x["first"] and x["card"]:
            chapters.append((t, x["card"].title()))
        d = durs[f"{x['n']:03d}"] + GAP
        hook = x["si"] == 0
        big = x.get("title") or (BIG.search(x.get("vo") or "") or [None])[0] if hook and x.get("kind") in ("still", "footage", "archive") else None
        items.append({**x, "start": t, "dur": d, **({"hook": True} if hook else {}), **({"big": str(big).replace("*", "").upper()} if big else {})})
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


def backdrop(L, i, ctx=None, it=None):
    """Behind a card: the last painting before it, blurred and dimmed (13 Nov 2026: cards on flat black read as plain);
    flat ground when there is none."""
    if ctx is not None and it is not None:
        k = ctx.items.index(it) if it in ctx.items else -1
        prev = next((x for x in reversed(ctx.items[:max(0, k)]) if x.get("kind") == "still"
                     and (ctx.out / f"shot-{x['n']:02d}.png").exists()), None)
        ph = next((x for x in reversed(ctx.items[:max(0, k)]) if x.get("kind") == "photo"
                   and (ctx.out / "photos" / f"{x['n']:03d}.jpg").exists()), None)
        if ph is not None and (prev is None or ctx.items.index(ph) > ctx.items.index(prev)):
            prev = {**ph, "_photo": True}
        if prev is not None:
            key = ("bg", prev["n"], bool(prev.get("_photo")))
            if key not in ctx.cache:
                import cv2
                src = ctx.out / ("photos" if prev.get("_photo") else "") / (f"{prev['n']:03d}.jpg" if prev.get("_photo") else f"shot-{prev['n']:02d}.png")
                im = np.asarray(Image.open(src).convert("RGB").resize((W // 4, H // 4)))
                im = cv2.resize(cv2.GaussianBlur(im, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
                ctx.cache = {a: b for a, b in ctx.cache.items() if a[0] != "bg"}
                ctx.cache[key] = (im.astype(np.float32) * 0.32 + np.array(L["bg"], np.float32) * 0.5).clip(0, 255).astype(np.uint8)
            return ctx.cache[key].copy()
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = L["bg"]
    return img


def beats(it):
    """How many pictures a still or clip line is cut into: every HOOK_BEAT s in the hook, BODY_BEAT s after."""
    if it.get("kind") in ("still", "footage", "archive"):
        return max(1, round(it["dur"] / (HOOK_BEAT if it.get("hook") else BODY_BEAT)))
    return 1


FRAMES = [(1.0, 0.5, 0.5), (1.38, 0.42, 0.45), (1.18, 0.6, 0.5), (1.5, 0.55, 0.4), (1.12, 0.38, 0.55)]  # zoom, cx, cy


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
    nb = beats(it)
    if nb > 1 and int(i * nb / max(1, n)) % 2:  # every other beat a punch-in on the same clip
        import cv2
        cw, ch = int(W / 1.22), int(H / 1.22)
        rgb = cv2.resize(rgb[(H - ch) // 2:(H - ch) // 2 + ch, (W - cw) // 2:(W - cw) // 2 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
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
    nb = beats(it)  # the painting in turn as wide, close and medium shots, each with its own move
    b = min(nb - 1, int(i * nb / max(1, n)))
    bn = max(1, n // nb)
    u = (i - b * bn) / max(1, bn - 1)
    z0, cx, cy = FRAMES[(b + it["n"]) % len(FRAMES)] if nb > 1 else (1.0, 0.5, 0.5)
    move = 0.10 if it.get("hook") else 0.07
    z = z0 * (1.0 + move * (u if (b + it["n"]) % 2 == 0 else 1 - u))  # push in, then pull out on the next
    cw, ch = int(W * 1.08 / z), int(H * 1.08 / z)
    x0 = int(np.clip(src.shape[1] * cx - cw / 2, 0, src.shape[1] - cw))
    y0 = int(np.clip(src.shape[0] * cy - ch / 2, 0, src.shape[0] - ch))
    import cv2
    return kit.grade(cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR), tint=ctx.L["tint"])


def _desk(ctx, key):
    """A dark desk under prints and clippings: the channel ground, a soft light from the top-left, film grain."""
    if key not in ctx.cache:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        light = 1.0 - 0.55 * np.clip(np.hypot((xx - W * 0.3) / W, (yy - H * 0.25) / H), 0, 1)
        g = np.array(ctx.L["bg"], np.float32) * 1.6
        img = np.clip(g[None, None, :] * light[..., None] + np.random.default_rng(3).normal(0, 4, (H, W, 1)), 0, 255)
        ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != "desk"}
        ctx.cache[key] = img.astype(np.uint8)
    return ctx.cache[key].copy()


def _print_card(im, style, caption, paper):
    """The photo as an object: a newspaper clipping (newsprint, the photo toned, a caption, a torn bottom edge) or a
    white-bordered print. Returns RGBA with a soft shadow."""
    from PIL import ImageFilter, ImageOps
    tw = 1180
    ph = im.copy()
    ph.thumbnail((tw, 720), Image.LANCZOS)
    if style == "clip":
        g = ImageOps.grayscale(ph).convert("RGB")
        ph = Image.blend(ph, g, 0.55)  # newsprint: most of the colour gone
        pad, capf = 34, kit.font("SourceSerif4", 34)
        d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines = wrap(d0, caption, capf, ph.width)[:2]
        cw, chh = ph.width + 2 * pad, ph.height + 2 * pad + 46 * len(lines) + 30
        card = Image.new("RGBA", (cw, chh), (*kit.hexrgb(paper), 255))
        card.paste(ph, (pad, pad))
        dd = ImageDraw.Draw(card)
        y = pad + ph.height + 18
        for ln in lines:
            dd.text((pad, y), ln, font=capf, fill=(40, 36, 30, 255))
            y += 46
        rng = np.random.default_rng(len(caption))  # a torn bottom edge
        edge = [(x, chh - 6 - int(rng.integers(0, 10))) for x in range(0, cw + 12, 12)]
        mask = Image.new("L", card.size, 255)
        ImageDraw.Draw(mask).polygon([(0, chh)] + edge + [(cw, chh)], fill=0)
        card.putalpha(mask)
    else:
        b = 22
        card = Image.new("RGBA", (ph.width + 2 * b, ph.height + 2 * b), (246, 244, 238, 255))
        card.paste(ph, (b, b))
    sh = Image.new("RGBA", (card.width + 60, card.height + 60), (0, 0, 0, 0))
    sh.paste((0, 0, 0, 150), (36, 40, 36 + card.width, 40 + card.height), card.split()[3])
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    sh.alpha_composite(card, (24, 24))
    return sh


def photo(ctx, it, t, i, n):
    """A real photograph presented as an object on a desk (owner, 13 Nov 2026: full-screen photos and stock read as
    "very raw"; the winners show prints and clippings): newspaper clippings and white-bordered prints in turn, sliding
    in with a slight tilt, then a slow push."""
    p = ctx.out / "photos" / f"{it['n']:03d}.jpg"
    if not p.exists():
        return still(ctx, it, t, i, n) if (ctx.out / f"shot-{it['n']:02d}.png").exists() else None
    key = ("card", it["n"])
    if key not in ctx.cache:
        style = "clip" if it["n"] % 3 != 2 else "print"
        words = str(it.get("vo") or "").split()
        caption = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
        ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != "card"}
        ctx.cache[key] = _print_card(Image.open(p).convert("RGB"), style, caption, ctx.L["paper"])
    card = ctx.cache[key]
    img = _desk(ctx, ("desk", 0))
    layer = kit.blank()
    s_in = kit.back_out(t / (12 / FPS))
    sc = min(1.0, (W * 0.78) / card.width, (H * 0.86) / card.height) * (0.94 + 0.08 * (i / max(1, n - 1)))
    ang = (-2.2, 1.6, -1.0, 2.4)[it["n"] % 4]
    kit.place(layer, card, (W / 2 + (1 - s_in) * 260 * (1 if it["n"] % 2 else -1), H / 2 + 10), ang * (0.6 + 0.4 * s_in), sc)
    return kit.over(img, layer)


def stat(ctx, it, t, i, n):
    s = it.get("stat") or {}
    img = backdrop(ctx.L, i, ctx, it)
    k_i = ctx.items.index(it) if it in ctx.items else 0
    run = []  # the numbers just before this one stay on screen, small (14 Nov 2026: four stat cards in a row read as a slideshow)
    for prev in reversed(ctx.items[max(0, k_i - 3):k_i]):
        if prev.get("kind") != "stat":
            break
        run.insert(0, prev.get("stat") or {})
    if run:
        lay0 = kit.blank()
        d0 = ImageDraw.Draw(lay0)
        f0, l0 = kit.font("BarlowCondensed-Bold", 84), kit.font("Inter", 24)
        for j, ps in enumerate(run):
            y0 = 300 + j * 170
            pv = str(ps.get("value", ""))
            try:
                pv = f"{float(ps.get('value')):,.4g}"
            except (TypeError, ValueError):
                pass
            sf = str(ps.get("suffix") or "")
            pv = str(ps.get("prefix") or "") + pv + ((" " + sf) if sf[:1].isalpha() else sf)
            d0.text((140, y0), pv, font=f0, fill=(*BONE, 150))
            d0.text((140, y0 + 92), str(ps.get("label") or "").upper()[:48], font=l0, fill=(150, 156, 160, 160))
        img = kit.over(img, lay0)
    cx = W * 0.62 if run else W / 2
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
    d.text((cx - tw / 2, H / 2 - 210), txt, font=f, fill=(*ctx.L["accent"], int(255 * a)))
    lab = str(s.get("label") or "").upper()
    lf = kit.font("Inter", 40)
    d.line([(cx - 300 * c, H / 2 + 70), (cx + 300 * c, H / 2 + 70)], fill=(*ctx.L["accent"], int(255 * a)), width=3)
    y = H / 2 + 100
    for ln in wrap(d, lab, lf, min(W - 400, 2 * (W - cx) - 160))[:3]:  # a long label wraps instead of running off the frame
        lw = sum(d.textlength(ch, font=lf) + 40 * 0.08 for ch in ln)
        kit.spaced(d, (cx - lw / 2, y), ln, lf, (*BONE, int(255 * ease_out((t - 0.3) / 0.4))))
        y += 56
    return kit.over(img, layer)


def paper_card(ctx, it, t, key, dateline, headline, hl):
    if (key, it["n"]) not in ctx.cache:
        ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != key}
        ctx.cache[(key, it["n"])] = kit.Clipping(dateline, headline, hl, paper=ctx.L["paper"], accent=ctx.L["hl"],
                                                 style=ctx.L["style"], seed=it["n"])
    c = ctx.cache[(key, it["n"])]
    img = backdrop(ctx.L, 0, ctx, it)
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
    img = backdrop(ctx.L, i, ctx, it)
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
    img = backdrop(ctx.L, i, ctx, it)
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


# ---------------------------------------------------------------- animated explainers (owner, 13 Nov 2026: "we still
# don't match their level in visuals": the winners build an animated graphic for each key moment)
def _ground(ctx, it, i):
    """Behind an explainer: the blurred last picture if there is one, else the channel ground, with a faint grid."""
    img = backdrop(ctx.L, i, ctx, it).astype(np.float32)
    g = np.zeros_like(img)
    g[::90, :, :] = 10
    g[:, ::90, :] = 10
    return np.clip(img * 0.8 + g, 0, 255).astype(np.uint8)


def timeline(ctx, it, t, i, n):
    """Years on a line: the line draws itself left to right, each event pops in on its turn (year above, what happened
    below), the latest one in the accent colour; the marked year gets a ring."""
    tl = it.get("timeline") or {}
    ev = [(str(e[0]), str(e[1])) for e in tl.get("events") or [] if isinstance(e, (list, tuple)) and len(e) >= 2][:6]
    img = _ground(ctx, it, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    if tl.get("title"):
        kit.spaced(d, (160, 120), str(tl["title"]).upper(), kit.font("Inter", 40), (*BONE, int(255 * ease_out(t / 0.4))))
    if not ev:
        return kit.over(img, layer)
    x0, x1, y = 150, W - 150, H / 2 + 20
    dur = max(1.0, it["dur"] - 0.6)
    p = ease_out(t / (dur * 0.85))
    d.line([(x0, y), (x0 + (x1 - x0) * p, y)], fill=(*BONE, 220), width=7)
    slot = (x1 - x0) / len(ev) - 24
    ys = 120
    while ys > 48 and max(d.textlength(yr, font=kit.font("BarlowCondensed-Bold", ys)) for yr, _ in ev) > slot:
        ys -= 6  # long dates ("Oct 2008") and six events shrink to fit (14 Nov 2026: labels overlapped)
    yf, lf = kit.font("BarlowCondensed-Bold", ys), kit.font("Inter", 40 if len(ev) <= 4 else 32)
    whens = [(k + 0.5) / len(ev) * dur * 0.85 for k in range(len(ev))]
    latest = sum(w <= t for w in whens) - 1  # the event just reached
    for k, (yr, what) in enumerate(ev):
        x = x0 + (x1 - x0) * (k + 0.5) / len(ev)
        a = ease_out((t - whens[k]) / 0.35)
        if a <= 0:
            continue
        cur = k == latest
        col = ctx.L["accent"] if cur else BONE
        r = 18 + 8 * (1 - a) + (6 if cur else 0)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, int(255 * a)))
        if yr == str(tl.get("mark", "")):
            rr = 46 + 8 * np.sin(t * 4)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=(226, 58, 46, int(220 * a)), width=4)
        tw = d.textlength(yr, font=yf)
        d.text((x - tw / 2, y - ys - 70 - 20 * (1 - a)), yr, font=yf, fill=(*col, int(255 * a)))
        yy = y + 50
        for ln in wrap(d, what, lf, (x1 - x0) / len(ev) - 30)[:3]:
            lw = d.textlength(ln, font=lf)
            d.text((x - lw / 2, yy), ln, font=lf, fill=(*BONE, int(235 * a)))
            yy += 52
    return kit.over(img, layer)


def flow(ctx, it, t, i, n):
    """Money or control moving between parties: boxes left to right, each arrow draws itself in turn with dots
    travelling along it, the amount popping onto it."""
    fl = it.get("flow") or {}
    nodes = [str(x) for x in fl.get("nodes") or []][:4]
    links = [l for l in fl.get("links") or [] if isinstance(l, (list, tuple)) and len(l) >= 2][:4]
    img = _ground(ctx, it, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    if fl.get("title"):
        kit.spaced(d, (160, 120), str(fl["title"]).upper(), kit.font("Inter", 40), (*BONE, int(255 * ease_out(t / 0.4))))
    if len(nodes) < 2:
        return kit.over(img, layer)
    nf, af = kit.font("Inter", 48), kit.font("BarlowCondensed-Bold", 96)
    pos = {}
    for k, nm in enumerate(nodes):
        cx = 280 + (W - 560) * k / (len(nodes) - 1)
        cy = H / 2 + 60 + (90 if k % 2 else -40) * (len(nodes) > 2)
        a = ease_out((t - 0.15 * k) / 0.35)
        bw = min(440, max(280, d.textlength(nm.upper(), font=nf) + 70))
        d.rounded_rectangle((cx - bw / 2, cy - 70, cx + bw / 2, cy + 70), radius=18, fill=(16, 18, 22, int(235 * a)),
                            outline=(*BONE, int(220 * a)), width=4)
        tw = d.textlength(nm.upper(), font=nf)
        d.text((cx - tw / 2, cy - 30), nm.upper(), font=nf, fill=(*BONE, int(255 * a)))
        pos[nm] = (cx, cy, bw)
    dur = max(1.0, it["dur"] - 0.5)
    for k, l in enumerate(links):
        if str(l[0]) not in pos or str(l[1]) not in pos:
            continue
        (ax, ay, aw), (bx, by, bw2) = pos[str(l[0])], pos[str(l[1])]
        sx, ex = ax + (aw / 2 + 14) * (1 if bx > ax else -1), bx - (bw2 / 2 + 14) * (1 if bx > ax else -1)
        start = 0.6 + k * (dur - 0.6) / max(1, len(links))
        p = ease_out((t - start) / 0.9)
        if p <= 0:
            continue
        mx, my = sx + (ex - sx) * p, ay + (by - ay) * p
        d.line([(sx, ay), (mx, my)], fill=(*ctx.L["accent"], 255), width=9)
        if p >= 1:  # the head and travelling dots
            ang = np.arctan2(by - ay, ex - sx)
            hx, hy = ex, by
            d.polygon([(hx, hy), (hx - 26 * np.cos(ang - 0.45), hy - 26 * np.sin(ang - 0.45)),
                       (hx - 26 * np.cos(ang + 0.45), hy - 26 * np.sin(ang + 0.45))], fill=(*ctx.L["accent"], 255))
            for j in range(3):
                q = ((t * 0.7 + j / 3) % 1.0)
                qx, qy = sx + (ex - sx) * q, ay + (by - ay) * q
                d.ellipse((qx - 7, qy - 7, qx + 7, qy + 7), fill=(255, 255, 255, 220))
        if len(l) > 2 and p > 0.5:
            lab = str(l[2])
            a = ease_out((p - 0.5) / 0.5)
            tw = d.textlength(lab, font=af)
            lx, ly = (sx + ex) / 2 - tw / 2, min(ay, by) - 70 - 140  # clear above both boxes
            d.rounded_rectangle((lx - 18, ly - 6, lx + tw + 18, ly + 110), radius=12, fill=(0, 0, 0, int(210 * a)))
            d.text((lx, ly), lab, font=af, fill=(*ctx.L["accent"], int(255 * a)))
    return kit.over(img, layer)


def compare(ctx, it, t, i, n):
    """Two sides side by side: names on top, the numbers counting up, bars growing to scale, the winner in the accent
    colour, the other in grey."""
    c = it.get("compare") or {}
    sides = [c.get("left") or {}, c.get("right") or {}]
    img = _ground(ctx, it, i)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    if c.get("title"):
        tf = kit.font("Inter", 40)
        tt = str(c["title"]).upper()
        kit.spaced(d, (W / 2 - d.textlength(tt, font=tf) / 2 - 20, 110), tt, tf, (*BONE, int(255 * ease_out(t / 0.4))))
    vals = []
    for sd in sides:
        try:
            vals.append(float(sd.get("value")))
        except (TypeError, ValueError):
            vals.append(0.0)
    hi = max(vals) or 1
    win = c.get("winner") or ("left" if vals[0] >= vals[1] else "right")
    nf, vf, lf = kit.font("Inter", 52), kit.font("BarlowCondensed-Bold", 170), kit.font("Inter", 36)
    d.line([(W / 2, 220), (W / 2, H - 140)], fill=(*BONE, 90), width=2)
    for k, sd in enumerate(sides):
        cx = W * (0.27 if k == 0 else 0.73)
        a = ease_out((t - 0.2 * k) / 0.4)
        nm = str(sd.get("name") or "").upper()
        d.text((cx - d.textlength(nm, font=nf) / 2, 210), nm, font=nf, fill=(*BONE, int(255 * a)))
        cnt = ease_out((t - 0.4) / 1.2)
        v = vals[k] * cnt
        dec = 0 if abs(vals[k]) >= 100 or float(vals[k]).is_integer() else 1
        txt = str(sd.get("prefix") or "") + f"{v:,.{dec}f}" + str(sd.get("suffix") or "")
        col = ctx.L["accent"] if (win == ("left" if k == 0 else "right")) else (150, 156, 160)
        d.text((cx - d.textlength(txt, font=vf) / 2, 300), txt, font=vf, fill=(*col, int(255 * a)))
        bh = 330 * (vals[k] / hi) * cnt  # under the number, never into it
        d.rectangle((cx - 150, H - 200 - bh, cx + 150, H - 200), fill=(*col, int(230 * a)))
        lab = str(sd.get("label") or "")
        for j, ln in enumerate(wrap(d, lab, lf, 560)[:2]):
            d.text((cx - d.textlength(ln, font=lf) / 2, H - 180 + 44 * j), ln, font=lf, fill=(*BONE, int(220 * a)))
    return kit.over(img, layer)


def mapshot(ctx, it, t, i, n):
    import maps
    return maps.frame(it.get("map") or {}, t, it["dur"], i)


def card(ctx, it, t, i, n):
    img = backdrop(ctx.L, i, ctx, it)
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


def titleslam(ctx, it, t, i, n):
    """The episode's name slammed in after the hook: scale 1.5 -> 1 in 0.25 s with a shake, the channel above it."""
    img = backdrop(ctx.L, i, ctx, it)
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    name, var = ctx.L["headfont"]
    txt = str(it["text"]).upper()
    size = 150
    while size > 70 and d.textlength(txt, font=kit.font(name, size, var)) > W - 240:
        size -= 8
    sc = 1.0 + 0.5 * (1 - ease_out(t / 0.25))
    f = kit.font(name, int(size * sc), var)
    sh = (np.random.default_rng(i).normal(0, 9, 2) * max(0.0, 1 - t / 0.45)).astype(int)
    tw = d.textlength(txt, font=f)
    a = min(1.0, t / 0.12) * (1 - ease_out((t - it["dur"] + 0.25) / 0.25))
    d.text((W / 2 - tw / 2 + sh[0], H / 2 - size * sc * 0.62 + sh[1]), txt, font=f, fill=(*BONE, int(255 * a)),
           stroke_width=3, stroke_fill=(0, 0, 0, int(200 * a)))
    kit.spaced(d, (W / 2 - 170, H / 2 - size * 0.62 - 70), ctx.L["name"], kit.font("Inter", 30), (*ctx.L["accent"], int(255 * a)))
    rw = 240 * ease_out((t - 0.15) / 0.4)
    d.line([(W / 2 - rw, H / 2 + size * 0.62), (W / 2 + rw, H / 2 + size * 0.62)], fill=(*ctx.L["accent"], int(255 * a)), width=5)
    return kit.over(img, layer)


def big_word(layer, text, t, dur, accent):
    """The hook line's number or year, huge, popping in (back-out) with a black outline, gone before the next beat."""
    if t < 0.12 or t > min(dur, 1.9):
        return
    d = ImageDraw.Draw(layer)
    sc = kit.back_out((t - 0.12) / 0.22)
    f = kit.font("BarlowCondensed-Bold", max(20, int(260 * (0.6 + 0.4 * sc))))
    a = int(255 * min(1.0, (min(dur, 1.9) - t) / 0.2))
    tw = d.textlength(text, font=f)
    d.text((W / 2 - tw / 2, H * 0.30), text, font=f, fill=(*accent, a), stroke_width=8, stroke_fill=(0, 0, 0, a))


def endcard(ctx, it, t, i, n):
    img = backdrop(ctx.L, i, ctx, it)
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


DRAW = {"footage": footage, "archive": footage, "still": still, "photo": photo, "stat": stat, "headline": headline,
        "document": document, "quote": quote, "chart": chart, "map": mapshot, "card": card, "end": endcard,
        "timeline": timeline, "flow": flow, "compare": compare,
        "title": titleslam}


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
    import cv2
    k_i = ctx.items.index(it)
    if k_i and li < XF and it.get("kind") not in ("card", "title", "end", "headline", "document", "quote", "stat",
                                                    "timeline", "flow", "compare"):
        # a crossfade between pictures (photos, stills, footage, maps); paper and graphics slide in on their own
        # (14 Nov 2026: a headline dissolving over a photo read as a double exposure)
        prev = ctx.items[k_i - 1]
        pk = ("prevframe", k_i)
        if pk not in ctx.cache:
            pn = max(1, int(round(prev["dur"] * FPS)))
            try:
                pr = DRAW.get(prev.get("kind"), stat)(ctx, prev, prev["dur"] - 1 / FPS, pn - 1, pn)
            except Exception:  # noqa: BLE001
                pr = None
            ctx.cache = {k: v for k, v in ctx.cache.items() if k[0] != "prevframe"}
            ctx.cache[pk] = pr
        pr = ctx.cache[pk]
        if pr is not None:
            a = ease_out(li / XF)
            rgb = cv2.addWeighted(rgb, a, pr, 1 - a, 0)
    if it.get("kind") not in ("map", "end") and li < 6:  # a snap-in on every cut: 1.06 -> 1 over 6 frames
        z = 1.0 + 0.025 * (1 - ease_out(li / 6))
        cw, ch = int(W / z), int(H / z)
        rgb = cv2.resize(rgb[(H - ch) // 2:(H - ch) // 2 + ch, (W - cw) // 2:(W - cw) // 2 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    nb = beats(it)
    since = li % max(1, n // nb) if nb > 1 else li
    if nb > 1 and it.get("kind") not in ("still", "footage", "archive") and int(li * nb / max(1, n)) % 2:
        z = 1.16 - 0.04 * min(1.0, since / max(1, n // nb))  # a card: every other beat a close punch-in, easing
        cw, ch = int(W / z), int(H / z)
        rgb = cv2.resize(rgb[(H - ch) // 2:(H - ch) // 2 + ch, (W - cw) // 2:(W - cw) // 2 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    # no flash on cuts (owner, 13 Nov 2026: "white booms")
    layer = kit.blank()
    d = ImageDraw.Draw(layer)
    if it.get("big"):
        big_word(layer, it["big"], lt, it["dur"], ctx.L["accent"])
    if it.get("vo") and it["start"] >= 60:  # no running captions in the first minute: big words there (owner, 13 Nov 2026)
        kit.captions(layer, it["vo"].split(), lt, it["dur"] - GAP, accent=ctx.L["accent"])
    if it.get("kind") in ("footage", "archive") and it.get("plate") and plate_ok(ctx, it):
        kit.label_plate(layer, (90, 110), str(it["plate"])[:48], lt)
    if it.get("kind") in ("footage", "archive"):
        kit.source_tag(d, "Pixabay")
    elif it.get("kind") == "photo" and (it.get("photo") or {}).get("credit"):
        kit.source_tag(d, ("Photo: " + str(it["photo"]["credit"]).split(",")[0])[:70])
    if it.get("kind") not in ("card", "end", "title"):
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
        if k == "title":  # one soft riser into the title, no boom (owner, 13 Nov 2026)
            mix.add(kit.synth("riser", 1.6), max(0, at - 1.5), 0.3)
        # sound design (owner, 13 Nov 2026: first "SFX are also less", then "unnecessary cuts, sfx"): a soft whoosh when
        # the kind of picture changes, one quiet sound per element, nothing stacked
        if prev is not None and k not in ("card", "title", "end"):  # one simple sound on every scene change (owner, 13 Nov
            # 2026: "a mouse click or page turn"): a page turn onto paper, a soft click onto anything else
            if k in ("headline", "document", "quote", "photo"):
                mix.add(page, max(0, at - 0.05), 0.28)
            else:
                mix.add(tick, max(0, at - 0.02), 0.3)
        if k in ("headline", "document"):
            pass  # the page turn on the cut is its sound
        elif k == "stat":
            mix.add(tick, at + 0.75, 0.25)
        elif k == "chart":
            mix.add(swipe, at + 0.1, 0.2)
        elif k == "timeline":
            evs = len((it.get("timeline") or {}).get("events") or [])
            for j in range(min(6, evs)):
                mix.add(tick, at + (j + 0.5) / max(1, evs) * max(1.0, it["dur"] - 0.6) * 0.85, 0.3)
        elif k == "flow":
            nl = len((it.get("flow") or {}).get("links") or [])
            for j in range(min(4, nl)):
                mix.add(swipe, at + 0.6 + j * (max(1.0, it["dur"] - 0.5) - 0.6) / max(1, nl), 0.2)
        elif k == "compare":
            mix.add(tick, at + 1.6, 0.3)
        elif k == "quote":
            mix.add(typing[: int(kit.SR * 1.0)], at + 0.1, 0.12)
        elif k == "map":
            act = (it.get("map") or {}).get("action") or "locate"
            if act == "locate":
                mix.add(kit.synth("riser", 2.0), at, 0.15)
            for j2 in range(min(3, len((it.get("map") or {}).get("places") or []))):
                mix.add(tick, at + (2.6 if act == "locate" else 0.4) + 0.25 * j2, 0.25)
        elif k == "card":
            mix.add(kit.synth("riser", 1.4), max(0, at - 1.2), 0.25)
            mix.add(thud, at, 0.35)
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
        if it.get("kind") in ("card", "title"):
            sec = f"_card{it['start']}"
        if not spans or spans[-1][0] != sec:
            spans.append([sec, it["start"], it["start"] + it["dur"]])
        else:
            spans[-1][2] = it["start"] + it["dur"]
        cur = it.get("section") or cur
    drops = [(it["start"] - 0.6, it["start"]) for it in items if it.get("kind") in ("card", "title")]
    k = 0
    for sec, a0, a1 in spans:
        if str(sec).startswith("_card"):
            continue
        if sec != "cold_open":  # no bed under the narration after the hook (owner, 13 Nov 2026: the beds read as sad)
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
