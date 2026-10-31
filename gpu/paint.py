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
sys.path.insert(0, str(ROOT / "brand"))
from brandcfg import B as BRAND  # noqa: E402  (HW_BRAND: another channel's art style)
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
            if s.get("prompt") and not s.get("diagram"):
                yield n, s["prompt"], s.get("negative", "")


def main(script_path, out_dir, part=0, parts=1):
    """part/parts: this process paints every parts-th job (one process per GPU on Kaggle's T4 x2)."""
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from diffusers import StableDiffusionXLPipeline
    # a brand's own model (Devlok Katha, owner 31 Oct 2026: realistic RealVisXL Lightning, 6 steps), else SDXL base
    src = BRAND.get("model") or model_path()
    dev = os.environ.get("PAINT_DEVICE", "cuda")  # cpu only for the smoke test (tiny model)
    kw = {"variant": "fp16", "use_safetensors": True} if src == HF_MODEL else {}
    if BRAND.get("vae"):
        from diffusers import AutoencoderKL
        kw["vae"] = AutoencoderKL.from_pretrained(BRAND["vae"], torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        src, torch_dtype=torch.float16 if dev == "cuda" else torch.float32, **kw).to(dev)
    print(f"  SDXL from {src}", flush=True)
    pipe.enable_vae_slicing()
    style, neg = BRAND["art_style"], BRAND["art_negative"]
    th = script.get("thumbnail") or {}
    # SDXL has two text encoders (77 tokens each). The second (OpenCLIP bigG) also gives the pooled
    # embedding that sets the overall composition, so it must carry the SUBJECT too: subject alone on the
    # first, subject + brand style on the second (style-only there drew generic hangar scenes, seen on
    # Kaggle). Shot-specific negatives (e.g. "airplane" for the island) keep the subject from turning into
    # the story's strongest noun.
    sw, sh = (768, 1344) if script.get("format") == "short" else (1344, 768)  # Shorts are painted vertical
    if script.get("format") == "short" and BRAND.get("short_size"):  # DK (31 Oct): 768x1344 stacked/merged figures
        sw, sh = BRAND["short_size"]
    jobs = [(out / f"shot-{n:02d}.png", p, f"{p}, {style}", f"{neg_extra}, {neg}" if neg_extra else neg, sw, sh, 1000 + n)
            for n, p, neg_extra in shots(script)]
    thumb_style = "cinematic digital painting, dramatic light, high contrast, highly detailed, sharp focus, no text"
    if th.get("plane"):  # the thumbnail keeps its subject bright and readable, so a lighter style
        jobs.append((out / "thumb-plane.png", th["plane"], f"{th['plane']}, {thumb_style}", neg, 1344, 768, 7))
    cv = (script.get("cover") or {}).get("prompt")
    if cv:  # a Short's own cover painting: the one image that says the title (owner, 31 Oct: DK covers didn't match)
        cs = BRAND.get("cover_style") or style  # DK: realistic, not the folk style (owner: "realistic, engaging, hooking")
        jobs.append((out / "cover.png", cv, f"{cv}, {cs}", f"blood, gore, deformed, extra limbs, {neg}", 768, 1344, 9))
    if th.get("zoom"):
        jobs.append((out / "thumb-zoom.png", th["zoom"], f"{th['zoom']}, {thumb_style}", neg, 1024, 1024, 8))
    for i, (p, prompt, prompt2, negative, w, h, seed) in enumerate(jobs, 1):
        if (i - 1) % parts != part or p.exists():
            continue
        g = torch.Generator(dev).manual_seed(seed)
        img = pipe(prompt=prompt, prompt_2=prompt2, negative_prompt=negative, negative_prompt_2=negative,
                   width=w, height=h, num_inference_steps=int(BRAND.get("steps", 30)), guidance_scale=float(BRAND.get("guidance", 7.0)),
                   generator=g).images[0]
        if BRAND.get("face_fix"):
            img = face_fix(pipe, img, prompt, prompt2, negative, g)
        img.save(p)
        print(f"  {i}/{len(jobs)} {p.name}", flush=True)


def face_fix(pipe, img, prompt, prompt2, negative, g):
    """Repaint each face larger, then paste it back (owner, 31 Oct: DK eyes came out garbled; a face is only
    ~150 px of an 832x1216 frame, too few for 6 Lightning steps to draw eyes). OpenCV's face detector; no
    face found = the painting is kept as it is."""
    import cv2
    import numpy as np
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from PIL import Image, ImageDraw, ImageFilter
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 6, minSize=(48, 48))
    if not len(faces):
        return img
    i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    for x, y, w, h in faces[:3]:
        c = int(max(w, h) * 1.8)  # the face plus hair and chin, as a square
        cx, cy = x + w // 2, y + h // 2
        box = (max(0, cx - c // 2), max(0, cy - c // 2), min(img.width, cx + c // 2), min(img.height, cy + c // 2))
        crop = img.crop(box)
        big = crop.resize((1024, 1024), Image.LANCZOS)
        new = i2i(prompt=f"detailed face, clear symmetrical eyes, {prompt}", prompt_2=prompt2, negative_prompt=f"bad eyes, crossed eyes, {negative}",
                  negative_prompt_2=negative, image=big, strength=0.4, num_inference_steps=int(BRAND.get("steps", 30)) * 2,
                  guidance_scale=float(BRAND.get("guidance", 7.0)), generator=g).images[0].resize(crop.size, Image.LANCZOS)
        m = Image.new("L", crop.size, 0)  # feathered edge: no visible seam
        ImageDraw.Draw(m).ellipse((crop.width * 0.08, crop.height * 0.08, crop.width * 0.92, crop.height * 0.92), fill=255)
        img.paste(new, box[:2], m.filter(ImageFilter.GaussianBlur(crop.width * 0.06)))
    return img


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(v) for v in sys.argv[3:5]))
