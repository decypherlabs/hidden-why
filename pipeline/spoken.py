"""English numbers as words before the voice reads them (owner, 15 Nov 2026: "17000 is narrated as 17, zero zero zero"
on THW, Story of Giants and The State Maps: the espeak phonemizer under Kokoro reads a thousands comma as a pause and
the zeros one by one). Only what it misreads changes: 1,500 / 17,000 / 145300 / $6,000 become words; years (1975),
small numbers, decimals and percentages stay, since it reads those right. Captions keep the written figures."""
import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]


def words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + words(n % 100) if n % 100 else "")
    for v, name in SCALES:
        if n >= v:
            return words(n // v) + " " + name + (" " + words(n % v) if n % v else "")
    return str(n)


ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
       "twelve": "twelfth"}


def ordinal(n):
    w = words(n)
    head, _, last = w.rpartition(" ") if " " in w else ("", "", w)
    pre, dash, tail = last.rpartition("-")
    tail = ORD.get(tail) or (tail[:-1] + "ieth" if tail.endswith("y") else tail + "th")
    return (head + " " if head else "") + (pre + dash if dash else "") + tail


def year(n):
    if n % 1000 < 10 and n // 1000 == 2:  # 2000, 2008
        return words(n)
    hi, lo = divmod(n, 100)
    return words(hi) + (" hundred" if lo == 0 else " oh " + ONES[lo] if lo < 10 else " " + words(lo))


TOKEN = re.compile(r"(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|s\b|st\b|nd\b|rd\b|th\b)?")
SCALE = re.compile(r"\s*(thousand|million|billion|trillion)\b")


FOLD = str.maketrans({"Đ": "D", "đ": "d", "Ø": "O", "ø": "o", "Ł": "L", "ł": "l", "ß": "ss", "Æ": "Ae", "æ": "ae",
                      "Œ": "Oe", "œ": "oe", "Þ": "Th", "þ": "th", "ı": "i", "’": "'", "‘": "'", "“": '"', "”": '"'})


def plain(text):
    """Accented Latin letters as plain ones for the English voice (owner, 23 Nov 2026: TSM Vietnam at 3:04, "Bạch
    Đằng" came out as random sounds; Vietnamese, Turkish, Polish... names everywhere): Bach Dang, Dong Hoi, Ca Mau.
    Other scripts are left alone. Captions and on-screen words keep the real spelling."""
    import unicodedata
    t = unicodedata.normalize("NFKD", text.translate(FOLD))
    return unicodedata.normalize("NFC", "".join(c for c in t if not (unicodedata.combining(c) and
                                                                    unicodedata.name(c, "").startswith("COMBINING"))))


def say(text):
    """Every figure as words (owner, 15 Nov 2026: "and other numbers too"): 17,000 / 6.75 / 90% / 1975 / 1980s /
    3rd / $6,000 / $2 billion. Accented names as plain letters (plain)."""
    text = plain(text)

    def rep(m):
        dollar, digits, dec, suf = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        n = int(digits.replace(",", ""))
        if suf in ("st", "nd", "rd", "th") and not dec:
            return ordinal(n)
        if suf == "s" and not dec and "," not in digits and 1000 <= n <= 2099:  # the 1980s
            w = year(n)
            return w[:-1] + "ies" if w.endswith("y") else w + "s"
        if not dollar and not dec and "," not in digits and len(digits) == 4 and 1000 <= n <= 2099 and suf != "%":
            w = year(n)
        else:
            w = words(n) + (" point " + " ".join(ONES[int(c)] for c in dec[1:]) if dec else "")
        if suf == "%":
            w += " percent"
        elif suf == "s":
            w += "s"
        if dollar:
            scale = SCALE.match(text, m.end())
            if scale:  # "$2 billion": the dollars after the scale word
                return w + " " + scale.group(1) + " dollars" + "\x00" * len(scale.group(0))
            w += " dollar" if n == 1 and not dec else " dollars"
        return w
    out = TOKEN.sub(rep, text)
    return re.sub("\x00+" + SCALE.pattern, "", out).replace("\x00", "")


if __name__ == "__main__":
    for t in ["It cost 17,000 dollars.", "$6,000 a year", "145300 people", "In 1975, 90% of film", "$6.75 billion in debt",
              "1,500 malls", "$2 billion", "a 12,000-mile route", "in the 1980s", "the 3rd and 21st", "2008 and 2000",
              "33 km wide, 2.5 miles", "1905 and 1066", "$1"]:
        print(t, "->", say(t))
