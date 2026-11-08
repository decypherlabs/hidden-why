"""Bring finished episodes (releases episode-<slug>) into the dashboard as The Hidden Why projects.

    python3 tools/sync_dashboard.py            # every published episode not yet in the dashboard

Every script in episodes/ shows up as soon as it exists, with its live state: "script" (written, or
approved and waiting for Kaggle), "awaiting_gpu" (being made on Kaggle; dashboard/data/thw_kaggle.json),
"failed" (the Kaggle run failed), then "done" once its release is imported.

Each episode becomes videos/hidden-why/<date>/<slug>/ with long/<title-slug>.mp4 plus its .meta.json,
.thumb.jpg and .source.json, and a project.json; the dashboard lists it under the account and its
normal Upload works. Files already downloaded to channels/the-hidden-why/videos/<slug>/ are reused.
The dashboard's background poller runs this every 10 minutes.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RB = ROOT.parent
sys.path.insert(0, str(ROOT / "tools"))
import fetch as F  # noqa: E402

VIDEOS = RB / "videos" / "hidden-why"


def niche(slug):
    """Hindi editions (episodes/<slug>-hi) belong to The Hidden Why Hindi's category, dk-* to Devlok Katha's
    (owner, 31 Oct 2026)."""
    if slug.startswith("ll-"):  # Logos & Legends episodes (owner, 7 Nov 2026)
        return "logos-and-legends"
    return "devlok-katha" if slug.startswith("dk-") else "hidden-why-hindi" if slug.endswith("-hi") or slug.startswith("gk-") \
        else "hidden-why"  # gk-*: Gazab Kyun (was The Hidden Why Hindi), 8 Nov 2026


def videos(slug):
    return RB / "videos" / niche(slug)


def discarded(slug):
    """THW Shorts cut from episodes (owner: Shorts are originals), archived 3 Nov 2026: never back on Projects."""
    # THW Hindi translations (episodes/<slug>-hi), archived 8 Nov 2026 when the channel became Gazab Kyun
    return bool(re.search(r"-short-\d+$", slug)) or slug.endswith("-hi")
CACHE = RB / "channels" / "the-hidden-why" / "videos"


def api(path):
    req = urllib.request.Request(f"https://api.github.com/repos/{F.REPO}{path}",
                                 headers={"Authorization": f"Bearer {F.TOK}", "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def slugify(t, n=60):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:n].strip("-")


def project_dir(slug):
    return next((p for p in sorted(videos(slug).glob(f"*/{slug}")) if (p / "project.json").exists()), None)


def parent_of(slug):
    """A promo Short's full episode (script "full_episode", Logos & Legends teasers), else None."""
    try:
        return json.loads((ROOT / "episodes" / slug / "script.json").read_text()).get("full_episode")
    except (OSError, ValueError):
        return None


def imported(slug):
    par = parent_of(slug)
    if par:  # a promo Short lives in its episode's project (owner, 8 Nov 2026)
        d = project_dir(par)
        return bool(d) and any(i.get("episode") == slug for i in json.loads((d / "project.json").read_text()).get("items") or [])
    d = project_dir(slug)
    return bool(d) and json.loads((d / "project.json").read_text()).get("status") == "done"


TITLE_TAGS = ["#science", "#facts", "#didyouknow"]  # owner, 1 Nov 2026: popular hashtags in every THW title


def thw_title(title, short):
    """The Hidden Why's title + its hashtags (#Shorts first on a Short), as many as fit YouTube's 100 chars."""
    base = re.sub(r"\s*#\w+", "", title).strip()
    tags = (["#Shorts"] if short else []) + TITLE_TAGS
    while tags and len(f"{base} {' '.join(tags)}") > 100:
        tags.pop()
    return f"{base} {' '.join(tags)}".strip()


def track_scripts(released):
    """A project card per script that isn't published yet, with where it is now."""
    run = {}
    try:
        run = json.loads((RB / "dashboard" / "data" / "thw_kaggle.json").read_text()).get("run") or {}
    except (OSError, ValueError):
        pass
    for sp in sorted((ROOT / "episodes").glob("*/script.json")):
        slug = sp.parent.name
        if parent_of(slug):  # promo Shorts have no card of their own: they join their episode's project when built
            continue
        if slug in released or discarded(slug) or slug.startswith("preview-"):  # sample frames for the owner, never a project on a channel
            continue
        sc = json.loads(sp.read_text())
        if sc.get("voice_review") and not sc.get("voice_ok") and not project_dir(slug):
            # waiting for the owner's voiceover: it lives on the Scripts page only (owner, 8 Nov 2026: 22 teaser
            # Short scripts showed up as project cards "made on Kaggle... on GitHub", read as builds running)
            continue
        shots = [x for sec in sc.get("sections") or [] for x in sec.get("shots") or []]
        mins = sum(len(x.get("vo", "").split()) for x in shots) / 150
        about = f"{len(shots)} shots, about {mins:.0f} min"
        if slug in (run.get("episodes") or []) and run.get("status") in ("queued", "running"):
            status, step, err = "awaiting_gpu", f"Being made on Kaggle since {run.get('started')} ({about})", None
        elif slug in (run.get("episodes") or []) and run.get("status") in ("error", "collect-failed"):
            status, step = "failed", f"The Kaggle run failed ({about}); Claude is fixing it"
            err = "see dashboard/data/thw_kaggle.last.log"
        elif sc.get("approved"):
            where = ("on GitHub (no GPU: the English paintings are reused)" if slug.endswith("-hi")
                     else f"on GitHub from {sc['paint_from']}'s paintings (no GPU)" if sc.get("paint_from")
                     else "painted on Kaggle, then narrated and edited on GitHub" if slug.startswith(("dk-", "ll-", "gk-")) else "in the next Kaggle run")
            status, step, err = "script", f"Script approved: made {where} ({about})", None
        else:
            status, step, err = "script", f"Script written: awaiting review ({about})", None
        pdir = project_dir(slug) or videos(slug) / time.strftime("%Y-%m-%d") / slug
        pj = pdir / "project.json"
        old = json.loads(pj.read_text()) if pj.exists() else {}
        if old.get("status") == "done" or (old.get("status"), old.get("step"), old.get("title")) == (status, step, sc["title"]) \
                or (old.get("status") == "awaiting_review" and status == "script"):  # QC-held stays held (no rebuild loop)
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        pj.write_text(json.dumps({
            "id": f"hw-{slug}", "category": niche(slug), "created": old.get("created") or now, "updated": now,
            "status": status, "progress": None, "step": step, "error": err, "pid": None, "title": sc["title"],
            "source": {"url": f"https://github.com/{F.REPO}/blob/main/episodes/{slug}/script.json", "title": sc["title"],
                       "creator": "The Hidden Why"}, "items": []}, indent=2, ensure_ascii=False))
        print(f"{slug}: {status} - {step}")


def sync_one(rel):
    """An episode release made before Kaggle runs went straight to the Mac (older episodes)."""
    slug = rel["tag_name"][len("episode-"):]
    if imported(slug) or discarded(slug):
        return False
    src = CACHE / slug
    src.mkdir(parents=True, exist_ok=True)
    for a in rel["assets"]:  # reuse what's already on disk (same size), fetch the rest in parallel ranges
        f = src / a["name"]
        if not (f.exists() and f.stat().st_size == a["size"]):
            F.fetch(a, f)
    return bool(import_episode(slug, src, rel["published_at"][:10]))  # counted by main()


def import_local(slug, files_dir, status="done"):
    """A finished episode straight from the Kaggle run's output (studio/thw_kaggle.py): moved into
    channels/the-hidden-why/videos/<slug>/ and into the dashboard; no GitHub round trip."""
    import shutil
    src = CACHE / slug
    src.mkdir(parents=True, exist_ok=True)
    for f in Path(files_dir).iterdir():
        shutil.move(str(f), src / f.name)
    return import_episode(slug, src, time.strftime("%Y-%m-%d"), status)


def import_promo(slug, par, src, day):
    """A promo Short goes into its full episode's project, beside the long-form (owner, 8 Nov 2026: "for LL, the
    promotional shorts should be part of same project"); named by its slug, so two promos never share a file
    (ll-019's two teasers were both amul.mp4). The episode's status and title stay its own."""
    meta = json.loads((src / "meta.json").read_text())
    title = meta["youtube"]["title"]
    pdir = project_dir(par) or videos(par) / day / par
    (pdir / "shorts").mkdir(parents=True, exist_ok=True)
    mp4 = pdir / "shorts" / f"{slug}.mp4"
    if mp4.exists() and not os.path.samefile(mp4, src / "video.mp4"):  # a rebuild: the old render is archived
        (pdir / "archive").mkdir(exist_ok=True)
        os.replace(mp4, pdir / "archive" / f"{slug}.{int(mp4.stat().st_mtime)}.mp4")
    if not mp4.exists():
        os.link(src / "video.mp4", mp4)
    if (src / "thumb.jpg").exists():
        (pdir / "shorts" / f"{slug}.thumb.jpg").write_bytes((src / "thumb.jpg").read_bytes())
        meta.setdefault("selected", {})["thumbnail"] = 1
    (pdir / "shorts" / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (pdir / "shorts" / f"{slug}.source.json").write_text(json.dumps(
        {"format": "short", "niche": niche(slug), "title": title, "speaker": "", "episode": slug,
         "source_url": f"https://github.com/{F.REPO}/tree/main/episodes/{slug}"}, indent=2))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                               capture_output=True, text=True).stdout or 0)
    pj = pdir / "project.json"
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data = json.loads(pj.read_text()) if pj.exists() else {
        "id": f"hw-{par}", "category": niche(par), "created": now, "status": "script", "progress": None, "error": None,
        "step": "Promo Shorts ready; the full episode is still being made", "pid": None,
        "title": title, "source": {"url": f"https://github.com/{F.REPO}/tree/main/episodes/{par}", "title": title,
                                   "creator": "The Hidden Why"}}
    item = {"kind": "short", "file": f"shorts/{slug}.mp4", "title": title, "duration": round(dur, 1), "meta": True,
            "thumbs": 1 if (src / "thumb.jpg").exists() else 0, "episode": slug}
    data["items"] = [i for i in data.get("items") or [] if i.get("file") != item["file"]] + [item]
    data["updated"] = now
    pj.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"{slug}: in the dashboard, in {par}'s project ({pdir.relative_to(RB)})")
    return pdir


def import_episode(slug, src, day, status="done"):
    par = parent_of(slug)
    if par:
        return import_promo(slug, par, src, day)
    meta = json.loads((src / "meta.json").read_text())
    title = meta["youtube"]["title"]
    pdir = project_dir(slug) or videos(slug) / day / slug  # the script's card becomes the episode
    sc = ROOT / "episodes" / slug / "script.json"
    short = sc.exists() and json.loads(sc.read_text()).get("format") == "short"  # Shorts go to shorts/
    sub = "shorts" if short else "long"
    if niche(slug) == "hidden-why":
        meta["youtube"]["title"] = thw_title(title, short)
    (pdir / sub).mkdir(parents=True, exist_ok=True)
    # a Devanagari title slugifies to nothing: keep the clip's existing name, else the episode slug (31 Oct)
    old = [p for p in (pdir / sub).glob("*.mp4")]
    stem = slugify(re.sub(r"#\w+", "", title.split("(")[0])) or (old[0].stem if len(old) == 1 else slugify(slug))
    if len(re.sub(r"[^a-z]", "", stem)) < 4:  # a Hindi title slugs to "1526": use the episode's
        stem = slugify(slug)
    mp4 = pdir / sub / f"{stem}.mp4"
    if mp4.exists() and not os.path.samefile(mp4, src / "video.mp4"):
        # a rebuild (repaint): the old render is archived, never deleted, and the new one takes its place
        # (29 Oct: rebuilds kept the old file, so QC held the same stale video every run)
        (pdir / "archive").mkdir(exist_ok=True)
        os.replace(mp4, pdir / "archive" / f"{stem}.{int(mp4.stat().st_mtime)}.mp4")
    if not mp4.exists():
        os.link(src / "video.mp4", mp4)  # hard link: no second copy of a big file
    chap = (src / "chapters.txt").read_text().strip() if (src / "chapters.txt").exists() else ""
    if not short and chap.count("\n") >= 2:  # long-form chapters into the description (owner, 7 Nov 2026: the benchmarks all list them)
        d = meta["youtube"].get("description", "")
        block = f"⏱️ CHAPTERS\n{chap}"
        meta["youtube"]["description"] = d.replace("{CHAPTERS}", chap) if "{CHAPTERS}" in d else \
            (d if "\n00:00 " in d else f"{d.rstrip()}\n\n{block}")
    if (src / "thumb.jpg").exists():  # Shorts come without a cover
        (pdir / sub / f"{stem}.thumb.jpg").write_bytes((src / "thumb.jpg").read_bytes())
        meta.setdefault("selected", {})["thumbnail"] = 1
    for k in (2, 3):  # the owner's Test & Compare alternates, set before the video was built (set_thumbs.py)
        alt = ROOT / "episodes" / slug / f"thumb-custom-{k}.jpg"
        if alt.exists():
            (pdir / sub / f"{stem}.thumb-{k}.jpg").write_bytes(alt.read_bytes())
    (pdir / sub / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                               capture_output=True, text=True).stdout or 0)
    (pdir / sub / f"{stem}.source.json").write_text(json.dumps(
        {"format": "short" if short else "long", "niche": niche(slug), "title": title, "speaker": "", "episode": slug,
         "source_url": f"https://github.com/{F.REPO}/tree/main/episodes/{slug}"}, indent=2))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:  # the episode's promo Shorts stay in its project when the long-form is (re)built
        promos = [i for i in json.loads((pdir / "project.json").read_text()).get("items") or []
                  if i.get("episode") and i.get("episode") != slug]
    except (OSError, ValueError):
        promos = []
    (pdir / "project.json").write_text(json.dumps({
        "id": f"hw-{slug}", "category": niche(slug), "created": now, "updated": now, "finished": now,
        "status": status, "progress": 1.0, "error": None,  # "qc": the studio's quality gate runs first
        "step": "Quality check" if status == "qc" else "Episode from the hidden-why repo, ready to upload",
        "pid": None, "source": {"url": f"https://github.com/{F.REPO}/tree/main/episodes/{slug}", "title": title,
                                "creator": "The Hidden Why"},
        "items": [{"kind": "short" if short else "long", "file": f"{sub}/{stem}.mp4", "title": title, "duration": round(dur, 1), "meta": True,
                   "thumbs": 1}] + promos}, indent=2, ensure_ascii=False))
    print(f"{slug}: in the dashboard ({pdir.relative_to(RB)})")
    return pdir


def main():
    n = 0
    rels = [r for r in api("/releases?per_page=100") if r["tag_name"].startswith("episode-") and r.get("assets")]
    for rel in rels:
        n += sync_one(rel)
    print(f"{n} new episode(s)")
    track_scripts({r["tag_name"][len("episode-"):] for r in rels} | {s for s in (p.parent.name for p in
                   (ROOT / "episodes").glob("*/script.json")) if imported(s)})


if __name__ == "__main__":
    main()
