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
if __import__("os").environ.get("SET") == "bold":  # long-form narrator (owner, 7 Nov 2026: "a bold male voice")
    TEXT = ["उन्नीस सौ अट्ठानवे में, एक आदमी बिना पैसे और बिना डिग्री के बैंक में गया।", "सबने उसका मज़ाक उड़ाया।",
            "लेकिन उसने हार नहीं मानी।", "दस साल बाद, वही बैंक उसके लिए काम कर रहा था।",
            "और ये कहानी, यहीं से शुरू होती है।"]
    BOLD = ("speaks in a deep, bold, low-pitched and commanding documentary narrator voice, with confident, slightly "
            "dramatic expression, at a moderate pace. The recording is very clear and close-sounding, with no background noise.")
    VOICES = {n.lower(): f"{n} {BOLD}" for n in ("Rohit", "Karan", "Aman", "Vikas")}
if __import__("os").environ.get("SET") == "dk-aman":  # Aman on a ~35 s Devlok Katha passage (owner, 7 Nov 2026)
    TEXT = ["हस्तिनापुर के राजा शांतनु, एक मछुआरे की बेटी सत्यवती से विवाह करना चाहते थे।",
            "लेकिन सत्यवती के पिता ने एक शर्त रखी।", "राजगद्दी पर सिर्फ़ सत्यवती का पुत्र ही बैठेगा।",
            "राजा उदास होकर लौट आए। उनके पुत्र देवव्रत से यह देखा नहीं गया।",
            "देवव्रत उस मछुआरे के पास गए, और सबके सामने प्रतिज्ञा ली।",
            "मैं कभी राजा नहीं बनूँगा। और मैं जीवन भर विवाह भी नहीं करूँगा।",
            "उस पल आकाश से फूल बरसे, और देवताओं ने कहा, भीष्म!",
            "उसी दिन से देवव्रत, भीष्म कहलाए।"]
    VOICES = {
        "aman-bold": ("Aman speaks in a deep, bold, low-pitched and commanding documentary narrator voice, with confident, "
                      "slightly dramatic expression, at a moderate pace. The recording is very clear and close-sounding, "
                      "with no background noise."),  # LL's exact voice
        "aman-epic": ("Aman speaks in a deep, resonant and reverent storytelling voice, with dramatic, emotional expression, "
                      "at a slightly slow pace. The recording is very clear and close-sounding, with no background noise."),
        "aman-warm": ("Aman speaks in a warm, calm and expressive storytelling voice at a moderate pace. The recording is "
                      "very clear and close-sounding, with no background noise."),
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
