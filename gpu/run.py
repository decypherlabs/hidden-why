"""Colab/Kaggle worker: paint every episode that has no paint-<slug> release yet, hand the paintings to
GitHub and start the 'build' workflow (narration, diagrams, edit, thumbnail).

    python gpu/run.py [slug ...]
Needs GH_TOKEN (Colab secret / Kaggle secret / env) with contents + actions write on this repo.
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
TOK = os.environ["GH_TOKEN"]


def gh(method, path, body=None, data=None, ctype="application/json"):
    url = path if path.startswith("http") else f"https://api.github.com/repos/{REPO}{path}"
    req = urllib.request.Request(url, method=method, data=data if data is not None else (json.dumps(body).encode() if body else None),
                                 headers={"Authorization": f"Bearer {TOK}", "Accept": "application/vnd.github+json",
                                          "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"GitHub {method} {path}: {e.code} {e.read().decode()[:200]}") from None


def main():
    slugs = sys.argv[1:] or sorted(p.name for p in (ROOT / "episodes").iterdir() if (p / "script.json").exists())
    for slug in slugs:
        if gh("GET", f"/releases/tags/paint-{slug}"):
            print(f"{slug}: already painted")
            continue
        out = ROOT / "out" / slug
        # one painter per GPU (Kaggle gives two T4s), each on its own card
        import torch
        gpus = max(1, torch.cuda.device_count())
        procs = [subprocess.Popen([sys.executable, str(ROOT / "gpu/paint.py"), str(ROOT / "episodes" / slug / "script.json"),
                                   str(out), str(k), str(gpus)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)})
                 for k in range(gpus)]
        if any(p.wait() for p in procs):
            raise SystemExit(f"{slug}: painting failed (see above)")
        tar = out.parent / f"paint-{slug}.tar"
        with tarfile.open(tar, "w") as t:
            for f in sorted(out.glob("*.png")):
                t.add(f, arcname=f.name)
        rel = gh("POST", "/releases", {"tag_name": f"paint-{slug}", "name": f"paint-{slug}", "prerelease": True})
        gh("POST", rel["upload_url"].split("{")[0] + f"?name={tar.name}", data=tar.read_bytes(), ctype="application/x-tar")
        gh("POST", "/actions/workflows/build.yml/dispatches", {"ref": "main", "inputs": {"slug": slug}})
        print(f"{slug}: paintings handed to GitHub; the build is running")


if __name__ == "__main__":
    main()
