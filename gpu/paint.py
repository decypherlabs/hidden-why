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
    """(key, prompt, negative, seed) per painting: key "NN" for a shot's own picture, "NN-2", "NN-3" for the extra
    pictures in its "more" list (owner, 7 Nov 2026: a new picture every ~4 s, like The Quiet Story). "paint_limit": N
    paints only the first N shots (the owner's preview before the whole episode is painted)."""
    n, limit = 0, script.get("paint_limit")
    for sec in script["sections"]:
        for s in sec.get("shots") or []:
            n += 1
            if limit and n > limit:
                return
            if s.get("prompt") and not s.get("diagram"):
                # "seed": a repaint after a failed shot check draws a new image, not the same one (1 Nov)
                yield f"{n:02d}", s["prompt"], s.get("negative", ""), int(s.get("seed") or 1000 + n)
                for k, extra in enumerate(s.get("more") or [], 2):
                    yield f"{n:02d}-{k}", extra, s.get("negative", ""), int(s.get("seed") or 1000 + n) + 100 * k


def main(script_path, out_dir, part=0, parts=1):
    """part/parts: this process paints every parts-th job (one process per GPU on Kaggle's T4 x2)."""
    script = json.loads(Path(script_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if BRAND.get("engine") == "flux":
        return paint_flux(script, out, part, parts)
    from diffusers import StableDiffusionXLPipeline
    # a brand's own model (Devlok Katha, owner 31 Oct 2026: realistic RealVisXL Lightning, 6 steps), else SDXL base
    src = os.environ.get("SDXL_PATH") or BRAND.get("model") or model_path()  # the smoke test's tiny model wins
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
    style = script.get("art_style") or style  # a series' own look (DK ranking / Kalyug, 2 Nov 2026)
    th = script.get("thumbnail") or {}
    # SDXL has two text encoders (77 tokens each). The second (OpenCLIP bigG) also gives the pooled
    # embedding that sets the overall composition, so it must carry the SUBJECT too: subject alone on the
    # first, subject + brand style on the second (style-only there drew generic hangar scenes, seen on
    # Kaggle). Shot-specific negatives (e.g. "airplane" for the island) keep the subject from turning into
    # the story's strongest noun.
    sw, sh = (768, 1344) if script.get("format") == "short" else (1344, 768)  # Shorts are painted vertical
    if script.get("format") == "short" and BRAND.get("short_size"):  # DK (31 Oct): 768x1344 stacked/merged figures
        sw, sh = BRAND["short_size"]
    jobs = [(out / f"shot-{n}.png", p, f"{p}, {style}", f"{neg_extra}, {neg}" if neg_extra else neg, sw, sh, seed)
            for n, p, neg_extra, seed in shots(script)]
    # a brand may set its own thumbnail look (DK, 7 Nov 2026: fiery and saturated like the niche's winners)
    thumb_style = BRAND.get("thumb_style") or "cinematic digital painting, dramatic light, high contrast, highly detailed, sharp focus, no text"
    flux_thumb = BRAND.get("thumb_engine") == "flux" and th.get("plane")  # DK (7 Nov 2026): the thumbnail on FLUX
    if th.get("plane") and not flux_thumb:  # the thumbnail keeps its subject bright and readable, so a lighter style
        jobs.append((out / "thumb-plane.png", th["plane"], f"{th['plane']}, {thumb_style}", neg, 1344, 768, 7))
    cv = (script.get("cover") or {}).get("prompt")
    if cv:  # a Short's own cover painting: the one image that says the title (owner, 31 Oct: DK covers didn't match)
        cs = script.get("art_style") or BRAND.get("cover_style") or style  # DK: realistic, not the folk style (owner: "realistic, engaging, hooking")
        jobs.append((out / "cover.png", cv, f"{cv}, {cs}", f"blood, gore, deformed, extra limbs, {neg}", sw, sh, int((script.get("cover") or {}).get("seed") or 9)))
    if th.get("zoom"):
        jobs.append((out / "thumb-zoom.png", th["zoom"], f"{th['zoom']}, {thumb_style}", neg, 1024, 1024, 8))
    for i, (p, prompt, prompt2, negative, w, h, seed) in enumerate(jobs, 1):
        if (i - 1) % parts != part or p.exists():
            continue
        g = torch.Generator(dev).manual_seed(seed)
        img = pipe(prompt=prompt, prompt_2=prompt2, negative_prompt=negative, negative_prompt_2=negative,
                   width=w, height=h, num_inference_steps=int(BRAND.get("steps", 30)), guidance_scale=float(BRAND.get("guidance", 7.0)),
                   generator=g).images[0]
        if BRAND.get("refine"):
            img = refine(pipe, img, prompt, prompt2, negative, g, float(BRAND["refine"]))
        if BRAND.get("face_fix"):
            img = face_fix(pipe, img, prompt, prompt2, negative, g)
        img.save(p)
        print(f"  {i}/{len(jobs)} {p.name}", flush=True)
    if flux_thumb and part == 0 and not (out / "thumb-plane.png").exists():
        del pipe
        torch.cuda.empty_cache()
        flux_images([(out / "thumb-plane.png", f"{th['plane']}, {thumb_style}", 7)], 1344, 768)


def flux_images(jobs, w, h):
    """FLUX.1-schnell for a few images (jobs: [(path, prompt, seed)]): the thumbnail of a brand that wants the
    premium look while its shots stay on SDXL (owner, 7 Nov 2026: "doesn't match the quality they have")."""
    import subprocess
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
    tok = next(Path("/kaggle/input").rglob("hf_token.txt"), None) if Path("/kaggle/input").exists() else None
    if tok and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = tok.read_text().strip()
    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    repo = BRAND.get("flux_model") or "black-forest-labs/FLUX.1-schnell"
    tr = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.float16,
                                                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                                       bnb_4bit_compute_dtype=torch.float16))
    pipe = FluxPipeline.from_pretrained(repo, transformer=tr, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    for p, prompt, seed in jobs:
        img = pipe(prompt, width=w, height=h, num_inference_steps=int(BRAND.get("flux_steps", 4)), guidance_scale=0.0,
                   max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        img.save(p)
        print(f"  FLUX {p.name}", flush=True)
    del pipe, tr
    torch.cuda.empty_cache()


def deity(prompt):
    """A god in the prompt gets the brand's fixed beautiful look (owner, 8 Nov 2026: "every God character is beautiful
    and attractive"; FLUX drew Ram as a villager and baby Krishna dark grey). One look per deity, first match wins."""
    import re
    looks = {k: v for k, v in (BRAND.get("deity_looks") or {}).items() if not k.startswith("_")}
    low, added = prompt.lower(), []
    for pat, look in looks.items():
        m = re.search(rf"\b(?:{pat})", low)
        if m and look not in added and len(added) < 2:  # two gods at most: FLUX reads ~256 tokens
            added.append(look)
            low = low.replace(m.group(0), "")  # "baby krishna" must not also pick up the grown-up Krishna
    return f"{prompt}, {', '.join(added)}" if added else prompt


def paint_flux(script, out, part=0, parts=1):
    """FLUX.1-schnell (Apache-2.0; owner, 6 Nov 2026: "more beautiful and premium", it paints what SDXL dropped:
    Kaliya, blue Krishna). 4 steps, no negative prompt, 4-bit transformer, CPU offload: ~41 s an image on a T4.
    The model is gated: the read-only token comes in the private input dataset (hf_token.txt)."""
    import subprocess
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
    tok = next(Path("/kaggle/input").rglob("hf_token.txt"), None) if Path("/kaggle/input").exists() else None
    if tok and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = tok.read_text().strip()
    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    repo = BRAND.get("flux_model") or "black-forest-labs/FLUX.1-schnell"
    style = script.get("art_style") or BRAND["art_style"]
    # more pixels than SDXL's canvas: crisper on a 1080p frame (owner, 7 Nov 2026: "the best, clear shorts")
    w, h = BRAND.get("flux_short_size", (864, 1536)) if script.get("format") == "short" else BRAND.get("flux_size", (1536, 864))
    jobs = [(out / f"shot-{n}.png", f"{deity(p)}, {style}", seed) for n, p, _, seed in shots(script)]
    th = script.get("thumbnail") or {}
    if th.get("plane"):  # a long-form's thumbnail painting (Logos & Legends on FLUX, 7 Nov 2026)
        jobs.append((out / "thumb-plane.png", f"{th['plane']}, {BRAND.get('thumb_style') or BRAND.get('cover_style') or style}", 7))
    cv = (script.get("cover") or {}).get("prompt")
    if cv:
        jobs.append((out / "cover.png", f"{cv}, {script.get('art_style') or BRAND.get('cover_style') or style}",
                     int((script.get("cover") or {}).get("seed") or 9)))
    jobs = [j for i, j in enumerate(jobs) if i % parts == part and not j[0].exists()]
    if not jobs:
        return
    steps = int(BRAND.get("flux_steps", 4))
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    try:  # fast: all prompts through the text encoders first, then the 4-bit transformer + VAE alone on the GPU
        enc = FluxPipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=torch.float16).to("cuda")
        with torch.no_grad():
            embeds = [tuple(t.cpu() for t in enc.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)[:2])
                      for _, prompt, _ in jobs]
        del enc
        torch.cuda.empty_cache()
        tr = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.float16, quantization_config=q)
        pipe = FluxPipeline.from_pretrained(repo, transformer=tr, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                            tokenizer_2=None, torch_dtype=torch.float16).to("cuda")
        print(f"  FLUX from {repo} (all on the GPU)", flush=True)
        for i, ((p, _, seed), (pe, ppe)) in enumerate(zip(jobs, embeds), 1):
            img = pipe(prompt_embeds=pe.to("cuda"), pooled_prompt_embeds=ppe.to("cuda"), width=w, height=h,
                       num_inference_steps=steps, guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            img.save(p)
            print(f"  {i}/{len(jobs)} {p.name}", flush=True)
        return
    except Exception as e:  # noqa: BLE001  the proven CPU-offload path below paints whatever is still missing
        print(f"  FLUX fast path failed ({type(e).__name__}: {e}); falling back to CPU offload", flush=True)
        enc = pipe = tr = None  # noqa: F841  free the GPU before the fallback loads
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    jobs = [j for j in jobs if not j[0].exists()]
    tr = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.float16, quantization_config=q)
    pipe = FluxPipeline.from_pretrained(repo, transformer=tr, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    print(f"  FLUX from {repo}", flush=True)
    for i, (p, prompt, seed) in enumerate(jobs, 1):
        img = pipe(prompt, width=w, height=h, num_inference_steps=int(BRAND.get("flux_steps", 4)), guidance_scale=0.0,
                   max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        img.save(p)
        print(f"  {i}/{len(jobs)} {p.name}", flush=True)


def refine(pipe, img, prompt, prompt2, negative, g, scale):
    """Hi-res pass over the whole painting (owner, 1 Nov: eyes still garbled): upscaled `scale`x and repainted
    lightly, so every face, turned or small, gets more pixels; shrunk back to the canvas size."""
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from PIL import Image
    i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    i2i.enable_vae_tiling()
    big = img.resize((int(img.width * scale) // 8 * 8, int(img.height * scale) // 8 * 8), Image.LANCZOS)
    out = i2i(prompt=f"sharp detailed faces, clear natural eyes, {prompt}", prompt_2=prompt2,
              negative_prompt=f"bad eyes, crossed eyes, {negative}", negative_prompt_2=negative, image=big, strength=0.3,
              num_inference_steps=int(BRAND.get("steps", 30)) * 3, guidance_scale=float(BRAND.get("guidance", 7.0)),
              generator=g).images[0]
    return out.resize(img.size, Image.LANCZOS)


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
    faces = list(det.detectMultiScale(gray, 1.1, 4, minSize=(40, 40)))
    side = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")  # turned faces too
    faces += [f for f in side.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
              if not any(abs(f[0] - g0[0]) < g0[2] and abs(f[1] - g0[1]) < g0[3] for g0 in faces)]
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
