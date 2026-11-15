"""The owner's card look for a picture (pipeline/build.py photos_in; the dashboard's shot slider shows the same)."""
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def card(im, tw, th, frac):
    """The owner's card look (15 Nov 2026, Jagannath): the picture as a rounded card with a thin gold edge and a soft
    shadow, frac of the frame (smaller for a small source), over a blurred, darkened copy of itself."""
    sc = max(tw / im.width, th / im.height)
    bg = im.resize((int(im.width * sc) + 1, int(im.height * sc) + 1), Image.LANCZOS)
    x0, y0 = (bg.width - tw) // 2, (bg.height - th) // 2
    bg = ImageEnhance.Brightness(bg.crop((x0, y0, x0 + tw, y0 + th)).filter(ImageFilter.GaussianBlur(35))).enhance(0.5)
    sc = min(tw * frac / im.width, th * frac / im.height)
    fg = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    x, y, r = (tw - fg.width) // 2, (th - fg.height) // 2, int(tw * 0.01)
    shade = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(shade).rounded_rectangle((x + 10, y + 18, x + fg.width + 10, y + fg.height + 18), r, fill=170)
    bg.paste(Image.new("RGB", (tw, th)), (0, 0), shade.filter(ImageFilter.GaussianBlur(26)))
    m = Image.new("L", fg.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, fg.width - 1, fg.height - 1), r, fill=255)
    bg.paste(fg, (x, y), m)
    ImageDraw.Draw(bg).rounded_rectangle((x, y, x + fg.width - 1, y + fg.height - 1), r, outline=(235, 200, 120), width=4)
    return bg
