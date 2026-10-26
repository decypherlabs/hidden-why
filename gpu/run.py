"""Kaggle/Colab worker: makes whole episodes on the notebook machine and publishes them as GitHub
releases. No GitHub Actions needed (they're slow); the Mac downloads the result (tools/fetch.py).

    python gpu/run.py [slug ...]                     # every APPROVED episode without an episode-<slug> release
    python gpu/run.py --repaint <slug> 26 30 31      # repaint those shots, then rebuild that episode

Per episode: paint (SDXL, one process per GPU) -> release paint-<slug> (paintings, kept for repaints)
-> narration (Kokoro on the GPU) -> diagrams + edit -> thumbnail -> metadata with real chapters ->
release episode-<slug> with video.mp4, thumb.jpg, meta.json, chapters.txt.
Needs GH_TOKEN (Kaggle/Colab secret or env) with contents write on this repo, or OUTBOX mode (Kaggle
run started by the Mac's API push, which can't use notebook secrets): the repo comes from a private
Kaggle dataset with the earlier paint packs in packs/, every "release" is written to $OUTBOX/<tag>/
instead, and the Mac publishes those releases after the run (studio/thw_kaggle.py).
"""
import json
import os
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "decypherlabs/hidden-why"
ROOT = Path(__file__).resolve().parent.parent
OUTBOX = Path(os.environ["OUTBOX"]) if os.environ.get("OUTBOX") else None
TOK = None if OUTBOX else os.environ["GH_TOKEN"]
PACKS = ROOT / "packs"  # OUTBOX mode: paint-<slug>.tar packs shipped with the repo snapshot
PY = sys.executable


def gh(method, path, body=None, data=None, ctype="application/json", accept="application/vnd.github+json"):
    url = path if path.startswith("http") else f"https://api.github.com/repos/{REPO}{path}"
    req = urllib.request.Request(url, method=method, data=data if data is not None else (json.dumps(body).encode() if body else None),
                                 headers={"Authorization": f"Bearer {TOK}", "Accept": accept, "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            raw = r.read()
            if accept != "application/vnd.github+json":
                return raw
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"GitHub {method} {path}: {e.code} {e.read().decode()[:200]}") from None


def release(tag):
    """The release for tag (created if missing)."""
    if OUTBOX:
        return {"tag": tag}
    return gh("GET", f"/releases/tags/{tag}") or gh("POST", "/releases", {"tag_name": tag, "name": tag, "prerelease": True})


def upload(rel, path, ctype):
    """Upload (or replace) one asset."""
    if OUTBOX:
        import shutil
        (OUTBOX / rel["tag"]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, OUTBOX / rel["tag"] / path.name)
        print(f"  saved {rel['tag']}/{path.name} ({path.stat().st_size / 1e6:.0f} MB) for the Mac to publish", flush=True)
        return
    for a in gh("GET", f"/releases/{rel['id']}/assets?per_page=100") or []:
        if a["name"] == path.name:
            gh("DELETE", f"/releases/assets/{a['id']}")
    gh("POST", rel["upload_url"].split("{")[0] + f"?name={path.name}", data=path.read_bytes(), ctype=ctype)
    print(f"  uploaded {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def step(script_py, *args):
    print(f"\n== {Path(script_py).stem}", flush=True)
    subprocess.run([PY, str(script_py), *map(str, args)], check=True)


def expected(script):
    """File names of every painting this episode needs (diagram shots are drawn in code, not painted)."""
    d, n, names = json.loads(Path(script).read_text()), 0, []
    for sec in d["sections"]:
        for sh in sec.get("shots") or []:
            n += 1
            if sh.get("prompt") and not sh.get("diagram"):
                names.append(f"shot-{n:02d}.png")
    th = d.get("thumbnail") or {}
    return names + [f for f, k in (("thumb-plane.png", "plane"), ("thumb-zoom.png", "zoom")) if th.get(k)]


def paint(slug, script, out):
    """SDXL on every GPU at once (Kaggle has two T4s): each process paints every n-th picture.
    Only missing pictures are painted; with a complete pack SDXL isn't even loaded."""
    missing = [f for f in expected(script) if not (out / f).exists()]
    if not missing:
        print(f"{slug}: all {len(expected(script))} paintings present (from the paint-{slug} release): nothing to paint",
              flush=True)
        return
    print(f"{slug}: painting {len(missing)}: {', '.join(missing)}", flush=True)
    import torch
    gpus = max(1, torch.cuda.device_count())
    procs = [subprocess.Popen([PY, str(ROOT / "gpu/paint.py"), str(script), str(out), str(k), str(gpus)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in range(gpus)]
    if any(p.wait() for p in procs):
        raise SystemExit(f"{slug}: painting failed (see above)")
    tar = out.parent / f"paint-{slug}.tar"
    with tarfile.open(tar, "w") as t:
        for f in sorted(out.glob("*.png")):
            t.add(f, arcname=f.name)
    upload(release(f"paint-{slug}"), tar, "application/x-tar")


def fetch_paintings(slug, out):
    """Earlier paintings of this episode (for a repaint): release paint-<slug>."""
    if OUTBOX:
        pack = PACKS / f"paint-{slug}.tar"
        if not pack.exists():
            raise SystemExit(f"{slug}: no paintings to repaint from (packs/paint-{slug}.tar missing)")
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(pack) as t:
            t.extractall(out)
        return
    rel = gh("GET", f"/releases/tags/paint-{slug}")
    asset = next((a for a in (rel or {}).get("assets", []) if a["name"].endswith(".tar")), None)
    if not asset:
        raise SystemExit(f"{slug}: no paintings to repaint from (release paint-{slug} missing)")
    out.mkdir(parents=True, exist_ok=True)  # also creates out/, where the pack is saved
    tar = out.parent / asset["name"]
    tar.write_bytes(gh("GET", f"/releases/assets/{asset['id']}", accept="application/octet-stream"))
    with tarfile.open(tar) as t:
        t.extractall(out)


def finish(slug, script, out):
    """Narration, diagrams + edit, thumbnail, metadata -> release episode-<slug>."""
    # Kokoro needs Python < 3.13 (Kaggle runs 3.13): the notebook makes a 3.12 env for it (TTS_PY)
    print("\n== narrate", flush=True)
    subprocess.run([os.environ.get("TTS_PY", PY), str(ROOT / "pipeline/narrate.py"), str(script), str(out)], check=True)
    step(ROOT / "pipeline/build.py", script, out)
    step(ROOT / "pipeline/thumb.py", script, out)
    meta = json.loads((script.parent / "meta.json").read_text())
    meta["youtube"]["description"] = meta["youtube"]["description"].replace("{CHAPTERS}", (out / "chapters.txt").read_text())
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    rel = release(f"episode-{slug}")
    for name, ctype in (("thumb.jpg", "image/jpeg"), ("meta.json", "application/json"),
                        ("chapters.txt", "text/plain"), ("video.mp4", "video/mp4")):
        upload(rel, out / name, ctype)
    print(f"\n{slug}: DONE -> https://github.com/{REPO}/releases/tag/episode-{slug}", flush=True)


def main():
    args = sys.argv[1:]
    if args[:1] == ["--repaint"]:
        slug, shots = args[1], [int(n) for n in args[2:]]
        script, out = ROOT / "episodes" / slug / "script.json", ROOT / "out" / slug
        fetch_paintings(slug, out)
        for n in shots:
            (out / f"shot-{n:02d}.png").unlink(missing_ok=True)
        print(f"{slug}: repainting shots {shots}", flush=True)
        paint(slug, script, out)
        finish(slug, script, out)
        return
    slugs = args or sorted(p.name for p in (ROOT / "episodes").iterdir() if (p / "script.json").exists())
    for slug in slugs:
        if not json.loads((ROOT / "episodes" / slug / "script.json").read_text()).get("approved"):
            print(f"{slug}: script not approved yet (review it first; then \"approved\": true): skipped")
            continue
        if not OUTBOX and gh("GET", f"/releases/tags/episode-{slug}"):  # OUTBOX: the Mac sends only due ones
            print(f"{slug}: already published as a release (use --repaint to redo shots)")
            continue
        script, out = ROOT / "episodes" / slug / "script.json", ROOT / "out" / slug
        if (PACKS / f"paint-{slug}.tar").exists() if OUTBOX else gh("GET", f"/releases/tags/paint-{slug}"):
            fetch_paintings(slug, out)  # painted before: reuse, paint only what's missing
        paint(slug, script, out)
        finish(slug, script, out)


if __name__ == "__main__":
    main()
