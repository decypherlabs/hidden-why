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
CACHE = RB / "channels" / "the-hidden-why" / "videos"


def api(path):
    req = urllib.request.Request(f"https://api.github.com/repos/{F.REPO}{path}",
                                 headers={"Authorization": f"Bearer {F.TOK}", "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def slugify(t, n=60):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:n].strip("-")


def project_dir(slug):
    return next((p for p in sorted(VIDEOS.glob(f"*/{slug}")) if (p / "project.json").exists()), None)


def imported(slug):
    d = project_dir(slug)
    return bool(d) and json.loads((d / "project.json").read_text()).get("status") == "done"


def track_scripts(released):
    """A project card per script that isn't published yet, with where it is now."""
    run = {}
    try:
        run = json.loads((RB / "dashboard" / "data" / "thw_kaggle.json").read_text()).get("run") or {}
    except (OSError, ValueError):
        pass
    for sp in sorted((ROOT / "episodes").glob("*/script.json")):
        slug = sp.parent.name
        if slug in released:
            continue
        sc = json.loads(sp.read_text())
        shots = [x for sec in sc.get("sections") or [] for x in sec.get("shots") or []]
        mins = sum(len(x.get("vo", "").split()) for x in shots) / 150
        about = f"{len(shots)} shots, about {mins:.0f} min"
        if slug in (run.get("episodes") or []) and run.get("status") in ("queued", "running"):
            status, step, err = "awaiting_gpu", f"Being made on Kaggle since {run.get('started')} ({about})", None
        elif slug in (run.get("episodes") or []) and run.get("status") in ("error", "collect-failed"):
            status, step = "failed", f"The Kaggle run failed ({about}); Claude is fixing it"
            err = "see dashboard/data/thw_kaggle.last.log"
        elif sc.get("approved"):
            status, step, err = "script", f"Script approved: made in the next Kaggle run ({about})", None
        else:
            status, step, err = "script", f"Script written: awaiting review ({about})", None
        pdir = project_dir(slug) or VIDEOS / time.strftime("%Y-%m-%d") / slug
        pj = pdir / "project.json"
        old = json.loads(pj.read_text()) if pj.exists() else {}
        if old.get("status") == "done" or (old.get("status"), old.get("step"), old.get("title")) == (status, step, sc["title"]):
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        pj.write_text(json.dumps({
            "id": f"hw-{slug}", "category": "hidden-why", "created": old.get("created") or now, "updated": now,
            "status": status, "progress": None, "step": step, "error": err, "pid": None, "title": sc["title"],
            "source": {"url": f"https://github.com/{F.REPO}/blob/main/episodes/{slug}/script.json", "title": sc["title"],
                       "creator": "The Hidden Why"}, "items": []}, indent=2, ensure_ascii=False))
        print(f"{slug}: {status} - {step}")


def sync_one(rel):
    """An episode release made before Kaggle runs went straight to the Mac (older episodes)."""
    slug = rel["tag_name"][len("episode-"):]
    if imported(slug):
        return False
    src = CACHE / slug
    src.mkdir(parents=True, exist_ok=True)
    for a in rel["assets"]:  # reuse what's already on disk (same size), fetch the rest in parallel ranges
        f = src / a["name"]
        if not (f.exists() and f.stat().st_size == a["size"]):
            F.fetch(a, f)
    return import_episode(slug, src, rel["published_at"][:10])


def import_local(slug, files_dir):
    """A finished episode straight from the Kaggle run's output (studio/thw_kaggle.py): moved into
    channels/the-hidden-why/videos/<slug>/ and into the dashboard; no GitHub round trip."""
    import shutil
    src = CACHE / slug
    src.mkdir(parents=True, exist_ok=True)
    for f in Path(files_dir).iterdir():
        shutil.move(str(f), src / f.name)
    return import_episode(slug, src, time.strftime("%Y-%m-%d"))


def import_episode(slug, src, day):
    meta = json.loads((src / "meta.json").read_text())
    title = meta["youtube"]["title"]
    pdir = project_dir(slug) or VIDEOS / day / slug  # the script's card becomes the episode
    (pdir / "long").mkdir(parents=True, exist_ok=True)
    stem = slugify(title.split("(")[0])
    mp4 = pdir / "long" / f"{stem}.mp4"
    if not mp4.exists():
        os.link(src / "video.mp4", mp4)  # hard link: no second copy of a big file
    (pdir / "long" / f"{stem}.thumb.jpg").write_bytes((src / "thumb.jpg").read_bytes())
    meta.setdefault("selected", {})["thumbnail"] = 1
    (pdir / "long" / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                               capture_output=True, text=True).stdout or 0)
    (pdir / "long" / f"{stem}.source.json").write_text(json.dumps(
        {"format": "long", "niche": "hidden-why", "title": title, "speaker": "", "episode": slug,
         "source_url": f"https://github.com/{F.REPO}/tree/main/episodes/{slug}"}, indent=2))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    (pdir / "project.json").write_text(json.dumps({
        "id": f"hw-{slug}", "category": "hidden-why", "created": now, "updated": now, "finished": now,
        "status": "done", "progress": 1.0, "step": "Episode from the hidden-why repo, ready to upload", "error": None,
        "pid": None, "source": {"url": f"https://github.com/{F.REPO}/tree/main/episodes/{slug}", "title": title,
                                "creator": "The Hidden Why"},
        "items": [{"kind": "long", "file": f"long/{stem}.mp4", "title": title, "duration": round(dur, 1), "meta": True,
                   "thumbs": 1}]}, indent=2, ensure_ascii=False))
    print(f"{slug}: in the dashboard ({pdir.relative_to(RB)})")
    return True


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
