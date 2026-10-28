"""Vertical Shorts cut from a finished, approved episode: its own narration lines, re-painted 9:16.

    python3 tools/cut_shorts.py 04-coca-cola-red [--count 4]     # writes episodes/<ep>-short-<n>/
    python3 tools/cut_shorts.py --all                             # every approved long episode not cut yet

DEPRECATED 28 Oct: the user rejected these cuts (no context, repetitive). THW Shorts are written as originals.

Claude (one headless call per episode, no tools) picks 3-4 self-contained 30-55 s stories from the
episode's narration and writes a vertical painting prompt per line plus the Shorts metadata. The words
are the episode's own, copied line for line and checked here, so they're as fact-checked as the
episode was: each cut is approved at once (approved: true, cut_from: <episode>). Diagram shots are
never used (they're drawn 16:9).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPS = ROOT / "episodes"
CLAUDE_GLOB = "/Users/admin/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
NEG = "text, letters, words, numbers, logo, watermark, brand name"
STYLE = ", vertical composition, dramatic cinematic lighting, highly detailed"

PROMPT = """You cut YouTube Shorts / Instagram Reels for "The Hidden Why", an English explainer channel
(US/UK audience) that answers why ordinary things are the way they are.

Episode "{title}". Its narration, one numbered line per shot:
{lines}

Pick {count} DIFFERENT Shorts. Each is one self-contained mini-story of 30-55 seconds (about 75-140
words at our pace), built ONLY from these lines, used verbatim, in their original order (you may skip
lines). Line 1 of each Short must hook in one second (a surprising claim, question or "you've never
noticed" moment) and the last line must land the answer/payoff. No line may depend on something the
Short left out ("that rule", "as we saw"). Never use a line marked [diagram].

Reply with ONLY JSON:
{{"shorts": [{{
 "slug": "<3-5 word kebab-case topic>",
 "title": "<Shorts title, 35-60 chars, a why-question or bold claim, one emoji at most, ends with #Shorts>",
 "lines": [<line numbers>],
 "prompts": ["<for each chosen line: a vivid painting for a 9:16 frame, one subject, no text>"],
 "description": "<YouTube description, 300-600 chars: the hook, the answer in 2-3 sentences, 'Full episode on the channel.', then 4-6 hashtags starting #Shorts and ending #thehiddenwhy>",
 "tags": ["<6-10 search phrases>"],
 "caption": "<Instagram caption 150-400 chars, ends with 3-5 hashtags including #thehiddenwhy>"
}}]}}"""


def claude(prompt):
    exe = sorted(glob.glob(CLAUDE_GLOB))[-1]
    r = subprocess.run([exe, "-p", prompt, "--output-format", "json", "--model", os.environ.get("CUT_MODEL", "sonnet"),
                        "--setting-sources", "user", "--strict-mcp-config", "--permission-mode", "dontAsk",
                        "--no-session-persistence", "--tools", ""], capture_output=True, text=True, timeout=900)
    res = json.loads(r.stdout or "{}").get("result") or ""
    m = re.search(r"\{.*\}", res, re.S)
    return json.loads(m.group(0)) if m else {}


def shots_of(script):
    return [sh for sec in script["sections"] for sh in sec.get("shots") or []]


def cut(ep, count=4):
    script = json.loads((EPS / ep / "script.json").read_text())
    if not script.get("approved") or script.get("format") == "short":
        return []
    shots = shots_of(script)
    lines = "\n".join(f"{i}. {'[diagram] ' if sh.get('diagram') else ''}{sh['vo']}" for i, sh in enumerate(shots, 1))
    plan = claude(PROMPT.format(title=script["title"], lines=lines, count=count))
    made = []
    for n, s in enumerate(plan.get("shorts") or [], 1):
        idx = s.get("lines") or []
        ok = (idx == sorted(set(idx)) and all(1 <= i <= len(shots) and not shots[i - 1].get("diagram") for i in idx)
              and len(s.get("prompts") or []) == len(idx))
        words = sum(len(shots[i - 1]["vo"].split()) for i in idx) if ok else 0
        if not ok or not 60 <= words <= 160 or len(s.get("description") or "") < 300:
            print(f"  {ep} short {n}: rejected (lines {idx}, {words} words)")
            continue
        slug = f"{ep}-short-{n}"
        d = EPS / slug
        d.mkdir(exist_ok=True)
        (d / "script.json").write_text(json.dumps({
            "slug": slug, "format": "short", "title": s["title"].replace(" #Shorts", ""), "approved": True,
            "cut_from": ep, "voice_speed": 1.05, "music": {"intense": ["main"]},
            "sections": [{"id": "main", "card": None, "shots": [
                {"vo": shots[i - 1]["vo"], "prompt": p.rstrip(". ") + STYLE, "move": "push", "negative": NEG}
                for i, p in zip(idx, s["prompts"])]}]}, indent=1, ensure_ascii=False))
        (d / "meta.json").write_text(json.dumps({
            "youtube": {"title": s["title"], "description": s["description"], "tags": ["the hidden why", "shorts", *s.get("tags", [])][:12],
                        "category_id": "27", "privacy": "public", "made_for_kids": False},
            "instagram": {"caption": s["caption"], "share_to_feed": True}}, indent=1, ensure_ascii=False))
        first = json.loads((d / "script.json").read_text())
        sh0 = first["sections"][0]["shots"][0]
        # a cut opens cold: drop a leading connector the episode needed ("Now, Santa." -> "Santa.")
        v = re.sub(r"^(?:Now|So|And|But|Then|Well|Okay|Still),?\s+", "", sh0["vo"])
        sh0["vo"] = v[:1].upper() + v[1:]
        (d / "script.json").write_text(json.dumps(first, indent=1, ensure_ascii=False))
        made.append(slug)
        print(f"  {slug}: {s['title']} ({words} words)")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--count", type=int, default=4)
    a = ap.parse_args()
    eps = [a.episode] if a.episode else []
    if a.all:
        cut_already = {json.loads(p.read_text()).get("cut_from") for p in EPS.glob("*/script.json")}
        eps = [p.parent.name for p in sorted(EPS.glob("*/script.json"))
               if p.parent.name not in cut_already and json.loads(p.read_text()).get("format") != "short"
               and json.loads(p.read_text()).get("approved")]
    if not eps:
        sys.exit("nothing to cut")
    for ep in eps:
        print(ep)
        cut(ep, a.count)


if __name__ == "__main__":
    main()
