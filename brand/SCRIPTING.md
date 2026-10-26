# Writing a Hidden Why episode

Benchmarks (2026-10-26, /Users/admin/rb/benchmarks/NOTES.md): The Infographics Show for long-form
(median 396K views), Zack D. Films for Shorts (median 10M).

## Voice
The narrator is the smartest friend in the room: second person, short sentences, no lecture.
Open a loop in the first 5 seconds ("If you just said Santa, you're not alone. And it's backwards."),
and close it only in THE REVEAL. Every section ends on a question or a turn ("But here's the problem.").
Shocking is good, untrue never: every fact checked against 2+ sources, listed in meta.json.

## Structure (sections and music stay as they are)
cold_open (3-4 shots, the myth + the stakes) -> sting -> THE ORDINARY (why nobody notices) ->
THE STORY (people, places, dates; time anchors like "By 9:45 AM...") -> THE REVEAL (the twist,
shown with a code-drawn diagram) -> TODAY -> outro (one-line answer, "That's the hidden why", next episode tease).

## Length
Target 8-12 minutes from episode 5 on (about 1,200-1,700 words, 50-80 shots); a new picture every 6-10 s.

## Pictures
Shot-specific prompts: name the subject, place, era, light and camera; negatives for logos/text/brands.
Numbers, dates, maps and labels are never painted: use a diagram (pipeline/diagram.py).

## Title and thumbnail
Title: a plain question or a stakes statement with a time anchor; 40-70 characters.
Thumbnail: `"layout": "face"`: `plane` is ONE person with an extreme emotion (shock, panic, disbelief)
filling the frame, painted for the topic; `text` is a 2-4 word statement, one *word* in yellow
("IT WAS *NEVER* SANTA"); optional zoom bubble with the key detail.
