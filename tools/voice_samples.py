"""Voice samples for a brand's narrator (owner, 15 Nov 2026: "DK's competitors' audio is a heavy voice with some
bass... give me samples of voices for DK on the first few lines of Jagannath's script"): the episode's first lines
read by each candidate Parler description, with the narration's own sentence pauses, one wav per voice.

    python tools/voice_samples.py <script.json> <out dir> [lines]       # .github/workflows/voice_samples.yml
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import narrate as N  # noqa: E402

CLEAR = "The recording is very clear and close-sounding, with no background noise."
VOICES = {
    "1-aman-now": None,  # the brand's current description
    "2-aman-deep": "Aman speaks in a very deep, heavy, bass-rich and resonant baritone voice, slow and grave, like a "
                   "temple documentary narrator, with powerful and dramatic expression. " + CLEAR,
    "3-rohit-deep": "Rohit speaks in a very deep, heavy, bass-rich and resonant baritone voice, slow and grave, like a "
                    "temple documentary narrator, with powerful and dramatic expression. " + CLEAR,
}


def main(script_path, out, n):
    import torch
    import torchaudio.functional as AF
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    script = json.loads(Path(script_path).read_text())
    v = N.BRAND["voice"]
    model = ParlerTTSForConditionalGeneration.from_pretrained(v["model"])
    tok = AutoTokenizer.from_pretrained(v["model"])
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    conf = N.BRAND.get("narration") or {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    texts = [t for _, t in N.lines(script)][:n]
    for name, desc in VOICES.items():
        d = dtok(desc or v["description"], return_tensors="pt")
        parts = []
        for text in texts:
            for piece, _verse, pause in N.pieces(text, None, conf):
                torch.manual_seed(N.SEEDS[0])
                p = tok(piece, return_tensors="pt")
                a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask).cpu().float().squeeze()
                x = AF.resample(a.unsqueeze(0), model.config.sampling_rate, N.SR).squeeze().numpy()
                parts += [N.trim(x), np.zeros(int(max(pause, 0.3) * N.SR), np.float32)]
            parts.append(np.zeros(int(0.45 * N.SR), np.float32))  # the breath between lines
        sf.write(out / f"{name}.wav", np.concatenate(parts), N.SR)
        print(name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
