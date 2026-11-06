"""Divya variants (owner, 6 Nov 2026: "make Divya a bit faster, more deep and confident"): the same lines in a few
written descriptions; speed and pitch tweaks are applied afterwards with ffmpeg."""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
M = "ai4bharat/indic-parler-tts"
LINES = json.loads(Path("episodes/dk-165-draupadi-endless-saree/script.json").read_text())["sections"][0]["shots"][:6]
CLEAR = " The recording is very clear and close-sounding, with no background noise."
DESC = {
    "a-now": "Divya speaks in a calm, expressive storytelling voice at a moderate pace." + CLEAR,
    "e1-animated": "Divya speaks in a very expressive and animated storytelling voice with high pitch variation, stressing the key words, at a moderate pace." + CLEAR,
    "e2-dramatic": "Divya speaks in a dramatic, intense and highly expressive voice, like a gripping storyteller building suspense, with strong emphasis and varied pitch." + CLEAR,
    "e3-per-line": None,  # the mood follows each line: a question surprised, the turn suspenseful, the rest animated
}
MOOD = {"?": "Divya speaks in a surprised, curious and very expressive tone, with strong emphasis and high pitch variation.",
        "रुकिए": "Divya speaks in a dramatic, suspenseful whisper-like tone, slow and intense, with strong emphasis.",
        "": "Divya speaks in a very expressive and animated storytelling voice with high pitch variation, stressing the key words."}
model = ParlerTTSForConditionalGeneration.from_pretrained(M)
tok, dtok = AutoTokenizer.from_pretrained(M), AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
for name, desc in DESC.items():
    parts = []
    for s in LINES:
        mood = next(m for k, m in MOOD.items() if k in s["vo"]) + CLEAR
        d = dtok(desc or mood, return_tensors="pt")
        torch.manual_seed(7)
        p = tok(s["vo"], return_tensors="pt")
        a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                           prompt_attention_mask=p.attention_mask).cpu().float().squeeze().numpy()
        parts += [a, torch.zeros(int(0.25 * model.config.sampling_rate)).numpy()]
    import numpy as np
    sf.write(OUT / f"{name}.wav", np.concatenate(parts), model.config.sampling_rate)
    print(name, flush=True)
