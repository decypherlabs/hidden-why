"""The Hidden Why Hindi: an English episode re-voiced in Hindi (owner, 31 Oct 2026).

    python3 tools/hindi.py <slug> <hindi.json>     # writes episodes/<slug>-hi/{script,meta}.json

hindi.json: {"title", "hook_text", "cover_text", "description", "tags": [...],
             "shots": [{"vo": "<Devanagari, read by the voice>", "caption": "<Hinglish, shown on screen>"}, ...]}
one entry per shot, in order. The copy keeps every painting prompt and paints nothing: "paint_from" makes
the build reuse the English episode's paintings (GitHub Actions, no GPU; studio/thw_kaggle.py hindi()).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICE = {"voice": "hm_omega", "lang": "hi", "lang_code": "h", "speed": 1.15}  # Kokoro's Hindi male narrator


def main(slug, hindi_path):
    src = ROOT / "episodes" / slug
    sc = json.loads((src / "script.json").read_text())
    hi = json.loads(Path(hindi_path).read_text())
    shots = [s for sec in sc["sections"] for s in sec.get("shots") or []]
    if len(hi["shots"]) != len(shots):
        raise SystemExit(f"{slug}: {len(shots)} shots, but {len(hi['shots'])} Hindi lines")
    for s, h in zip(shots, hi["shots"]):
        s["vo"], s["caption"] = h["vo"], h["caption"]
    sc.pop("repaint", None)
    sc.pop("voice_speed", None)
    sc.update(slug=f"{slug}-hi", lang="hi", paint_from=slug, voice=VOICE, approved=True,
              title=hi["title"], hook_text=hi["hook_text"], cover_text=hi.get("cover_text") or hi["hook_text"])
    dst = ROOT / "episodes" / f"{slug}-hi"
    dst.mkdir(exist_ok=True)
    (dst / "script.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False) + "\n")
    tags = ["the hidden why hindi", "shorts", "hindi facts", "science in hindi", *hi.get("tags", [])]
    (dst / "meta.json").write_text(json.dumps({
        "youtube": {"title": f"{hi['title']} #Shorts", "description": hi["description"], "tags": tags,
                    "category_id": "27", "privacy": "public", "made_for_kids": False, "default_language": "hi"},
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"episodes/{slug}-hi: {len(shots)} shots")


if __name__ == "__main__":
    main(*sys.argv[1:3])
