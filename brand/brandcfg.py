"""The active brand: brand/brand.json (The Hidden Why), overlaid by brand/<HW_BRAND>/brand.json when a script
names another channel ("brand": "devlok-katha"). No PIL here: narrate.py runs in the TTS env.
Assets (logo.png, music/<mood>.mp3) come from the brand's folder, else from brand/."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = os.environ.get("HW_BRAND") or ""
B = json.loads((HERE / "brand.json").read_text())
if NAME:
    B = {**B, **json.loads((HERE / NAME / "brand.json").read_text())}


def asset(rel):
    p = HERE / NAME / rel
    return p if NAME and p.exists() else HERE / rel
