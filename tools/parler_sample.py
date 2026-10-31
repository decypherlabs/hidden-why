"""Indic Parler-TTS samples (AI4Bharat, Apache-2.0): free Hindi narration for Devlok Katha (owner, 31 Oct 2026).

    python tools/parler_sample.py out/voices
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

TEXT = ["समुद्र मंथन चल रहा था।", "देवता और असुर अमृत की खोज में थे।",
        "लेकिन अमृत से पहले निकला हलाहल, ऐसा विष जो पूरी सृष्टि को जला सकता था।",
        "तब महादेव आगे आए, और उन्होंने वो विष पी लिया।",
        "माता पार्वती ने उनका गला थाम लिया, ताकि विष नीचे न उतरे।", "और इसीलिए, महादेव कहलाए नीलकंठ।"]
VOICES = {
    "rohit": "Rohit speaks in a deep, calm and expressive storytelling voice at a moderate pace. The recording is very clear and close-sounding, with no background noise.",
    "karan": "Karan speaks in a deep, warm and dramatic narrator voice at a slightly slow pace. The recording is very clear and close-sounding, with no background noise.",
    "divya": "Divya speaks in a calm, expressive storytelling voice at a moderate pace. The recording is very clear and close-sounding, with no background noise.",
}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts")
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
sr = model.config.sampling_rate
for name, desc in VOICES.items():
    d = dtok(desc, return_tensors="pt")
    parts = []
    for i, line in enumerate(TEXT):
        torch.manual_seed(7)
        p = tok(line, return_tensors="pt")
        a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                           prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        parts += [a.cpu().numpy().squeeze(), np.zeros(int(sr * 0.25), np.float32)]
    sf.write(out / f"{name}.wav", np.concatenate(parts), sr)
    print(name, "done", flush=True)
