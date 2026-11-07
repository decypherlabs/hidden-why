"""The fingerprint of an episode's narration (owner, 7 Nov 2026: hear the voice before any GPU is spent): every
spoken line plus the brand's voice settings. A narration made for one fingerprint is reused only by the same one,
so a rewritten line or a new voice description always narrates again. No dependencies: the studio and the
dashboard import it too.

    python tools/voicehash.py episodes/<slug>/script.json      ->  10 hex characters
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def voice(script):
    b = json.loads((ROOT / "brand" / "brand.json").read_text())
    if script.get("brand"):
        b = {**b, **json.loads((ROOT / "brand" / script["brand"] / "brand.json").read_text())}
    return script.get("voice") or b.get("voice") or {}


def of(script):
    lines = [s["vo"] for sec in script["sections"] for s in sec.get("shots") or []]
    v = {k: x for k, x in voice(script).items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps([lines, v], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]


if __name__ == "__main__":
    print(of(json.loads(Path(sys.argv[1]).read_text())))
