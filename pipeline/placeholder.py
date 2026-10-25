"""Test helper: flat gradient stand-ins for every painting (no GPU), so the edit can be checked."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

script = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
n = 0
for sec in script["sections"]:
    for s in sec.get("shots") or []:
        n += 1
        if s.get("prompt"):
            img = Image.new("RGB", (1344, 768), (15, 59, 68))
            ImageDraw.Draw(img).text((40, 360), f"shot {n}: {s['prompt'][:90]}", fill=(242, 238, 227))
            img.save(out / f"shot-{n:02d}.png")
Image.new("RGB", (1344, 768), (40, 60, 90)).save(out / "thumb-plane.png")
Image.new("RGB", (1024, 1024), (90, 120, 160)).save(out / "thumb-zoom.png")
