"""Download a finished episode (release episode-<slug>) to the Mac, fast.

    python3 tools/fetch.py 01-round-windows [dest]     # default: ../channels/the-hidden-why/videos/<slug>/

GitHub's release storage gives one connection only ~30-50 KB/s here, so big files come down in 16
parallel byte ranges; sizes are checked after.
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "decypherlabs/hidden-why"
ROOT = Path(__file__).resolve().parent.parent
TOK = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def storage_url(asset_id):
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/assets/{asset_id}",
                                 headers={"Authorization": f"Bearer {TOK}", "Accept": "application/octet-stream"})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 307):
            return e.headers["Location"]
        raise
    raise RuntimeError("expected a redirect to storage")


def get_range(url, a, b, dest):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == b - a + 1:
                with open(dest, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                return
        except Exception:  # noqa: BLE001 - retried
            pass
    raise RuntimeError(f"range {a}-{b} failed")


def fetch(asset, dest):
    size, url = asset["size"], storage_url(asset["id"])
    with open(dest, "wb") as f:
        f.truncate(size)
    chunk = max(1 << 20, size // 16 + 1)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda a: get_range(url, a, min(size, a + chunk) - 1, dest), range(0, size, chunk)))
    if dest.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: size mismatch")
    print(f"  {dest.name} ({size / 1e6:.0f} MB)")


def main():
    slug = sys.argv[1]
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "channels/the-hidden-why/videos" / slug
    dest.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/tags/episode-{slug}",
                                 headers={"Authorization": f"Bearer {TOK}", "Accept": "application/vnd.github+json"})
    rel = json.loads(urllib.request.urlopen(req, timeout=60).read())
    for a in rel["assets"]:
        fetch(a, dest / a["name"])
    print(f"{slug} -> {dest}")


if __name__ == "__main__":
    main()
