"""GPU step: one cinematic painting per shot (brand art style), plus the thumbnail's plane and zoom.

    python gpu/paint.py episodes/<slug>/script.json out/<slug>/

Writes shot-NN.png (1344x768, SDXL's 16:9 bucket), thumb-plane.png and thumb-zoom.png. Fixed seeds, and
existing files are kept, so a rerun after a Colab disconnect continues where it stopped.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BRAND = json.loads((ROOT / "brand" / "brand.json").read_text())
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def model_path():
    """SDXL from an attached Kaggle dataset (any /kaggle/input/** folder holding an SDXL model_index.json),
    else SDXL_PATH, else the Hugging Face download."""
    if os.environ.get("SDXL_PATH"):
        return os.environ["SDXL_PATH"]
    for idx in sorted(Path("/kaggle/input").glob("**/model_index.json")) if Path("/kaggle/input").exists() else []:
        if "StableDiffusionXLPipeline" in idx.read_text():
            return str(idx.parent)
    return HF_MODEL


def shots(script):
    n = 0
    for sec in script["sections"]:
        for s in sec.get("shots") or []:
            n += 1
            if s.get("prompt"):
                yield n, s["prompt"]


def main(script_path, out_dir, part=0, parts=1):
    """part/parts: this process paints every parts-th job (one process per GPU on Kaggle's T4 x2)."""
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from diffusers import StableDiffusionXLPipeline
    src = model_path()
    dev = os.environ.get("PAINT_DEVICE", "cuda")  # cpu only for the smoke test (tiny model)
    kw = {"variant": "fp16", "use_safetensors": True} if src == HF_MODEL else {}
    pipe = StableDiffusionXLPipeline.from_pretrained(
        src, torch_dtype=torch.float16 if dev == "cuda" else torch.float32, **kw).to(dev)
    print(f"  SDXL from {src}", flush=True)
    pipe.enable_vae_slicing()
    style, neg = BRAND["art_style"], BRAND["art_negative"]
    th = script.get("thumbnail") or {}
    jobs = [(out / f"shot-{n:02d}.png", f"{p}, {style}", 1344, 768, 1000 + n) for n, p in shots(script)]
    if th.get("plane"):  # the thumbnail keeps its subject bright and readable, so a lighter style
        jobs.append((out / "thumb-plane.png", f"{th['plane']}, cinematic digital painting, dramatic light, "
                     "high contrast, highly detailed, no text", 1344, 768, 7))
    if th.get("zoom"):
        jobs.append((out / "thumb-zoom.png", f"{th['zoom']}, cinematic digital painting, crisp detail, no text",
                     1024, 1024, 8))
    for i, (p, prompt, w, h, seed) in enumerate(jobs, 1):
        if (i - 1) % parts != part or p.exists():
            continue
        g = torch.Generator(dev).manual_seed(seed)
        img = pipe(prompt, negative_prompt=neg, width=w, height=h, num_inference_steps=30, guidance_scale=6.5,
                   generator=g).images[0]
        img.save(p)
        print(f"  {i}/{len(jobs)} {p.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(v) for v in sys.argv[3:5]))
