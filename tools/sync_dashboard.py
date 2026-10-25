"""Bring finished episodes (releases episode-<slug>) into the dashboard as The Hidden Why projects.

    python3 tools/sync_dashboard.py            # every published episode not yet in the dashboard

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


def imported(slug):
    return any((p / "project.json").exists() for p in VIDEOS.glob(f"*/{slug}"))


def sync_one(rel):
    slug = rel["tag_name"][len("episode-"):]
    if imported(slug):
        return False
    src = CACHE / slug
    src.mkdir(parents=True, exist_ok=True)
    for a in rel["assets"]:  # reuse what's already on disk (same size), fetch the rest in parallel ranges
        f = src / a["name"]
        if not (f.exists() and f.stat().st_size == a["size"]):
            F.fetch(a, f)
    meta = json.loads((src / "meta.json").read_text())
    title = meta["youtube"]["title"]
    day = rel["published_at"][:10]
    pdir = VIDEOS / day / slug
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
         "source_url": f"https://github.com/{F.REPO}/releases/tag/episode-{slug}"}, indent=2))
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
    for rel in api("/releases?per_page=100"):
        if rel["tag_name"].startswith("episode-") and rel.get("assets"):
            n += sync_one(rel)
    print(f"{n} new episode(s)")


if __name__ == "__main__":
    main()
