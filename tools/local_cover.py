"""Shorts covers painted on the Mac (owner, 31 Oct 2026: "for already created ones, generate thumbnails on
this machine"): script["cover"]["prompt"] -> out/<slug>/cover.png with a realistic SDXL Lightning model on
Apple's GPU (MPS), one picture at a time (8 GB Mac), then pipeline/thumb.py adds the title question.

    HW_BRAND=devlok-katha ~/img-py/bin/python tools/local_cover.py dk-01-neelkanth dk-02-ganesh-gajmukh ...

Writes out/<slug>/cover.png and out/<slug>/thumb.jpg; nothing in the dashboard changes.
"""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brand"))
from brandcfg import B  # noqa: E402  (HW_BRAND overlay)

MODEL = "SG161222/RealVisXL_V5.0_Lightning"  # realistic, 4-6 steps: fits an 8 GB Mac
NEG = "text, letters, words, watermark, logo, blood, gore, deformed, extra limbs, extra fingers, blurry, cartoon, flat"


def main(slugs):
    style = B.get("cover_style") or B["art_style"]
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.float16).to("mps")
    pipe.enable_attention_slicing()
    for slug in slugs:
        script = ROOT / "episodes" / slug / "script.json"
        prompt = json.loads(script.read_text())["cover"]["prompt"]
        out = ROOT / "out" / slug
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "cover.png").exists():
            g = torch.Generator("cpu").manual_seed(9)
            img = pipe(prompt=prompt, prompt_2=f"{prompt}, {style}", negative_prompt=NEG, width=768, height=1344,
                       num_inference_steps=6, guidance_scale=1.5, generator=g).images[0]
            img.save(out / "cover.png")
            gc.collect()
            torch.mps.empty_cache()
        if not features.check("raqm"):  # without it Devanagari matras come out broken: paint only
            print(f"{slug}: painted; no raqm (brew install libraqm), cover text skipped", flush=True)
            continue
        subprocess.run([sys.executable, str(ROOT / "pipeline" / "thumb.py"), str(script), str(out)], check=True)
        print(f"{slug}: {out / 'thumb.jpg'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
