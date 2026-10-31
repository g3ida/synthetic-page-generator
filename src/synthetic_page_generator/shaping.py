"""Measuring and laying out Arabic text: what a face can draw, and lines that fit."""

from __future__ import annotations

import os
import re

from PIL import ImageFont

TAG = re.compile(r"<[^>]+>")

# Harakat, tanween, sukun, superscript alef and the Qur'anic marks. Tatweel
# (U+0640) is deliberately NOT here: it is a letter-joining stretch, part of
# the word's shape, not a diacritic -- removing it would change the glyphs.
TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def font_cmap(path):
    """Every codepoint a face can actually draw.

    PIL has no fallback: a character the font lacks is drawn as .notdef -- an
    empty box -- while the text the page was built from still has it, so the
    image no longer shows what the page claims to say. It is common: several
    Arabic faces lack the ornate Qur'anic parentheses U+FD3E/U+FD3F, which
    classical texts use often.
    """
    from fontTools.ttLib import TTFont
    cs = set()
    for t in TTFont(path, fontNumber=0)["cmap"].tables:
        cs |= set(t.cmap)
    return cs


def drawable(text, cmap):
    """True if every visible character of the page exists in the face.

    Tags are stripped first: they are never drawn, so a font lacking Latin
    letters must not be rejected over the "p" in "<p>".
    """
    body = TAG.sub("", text)
    return not any(ord(c) not in cmap for c in body if not c.isspace())


def strip_tashkeel(text):
    """Undiacritised copy of a page's markup.

    Safe to run over raw markup: every codepoint in TASHKEEL is Arabic, so tag
    names and attributes cannot be touched. Stripping the SOURCE means the
    image and any text recorded with it come from the same string, so the page
    still says exactly what is drawn on it.
    """
    return TASHKEEL.sub("", text)



def font_bundle(path, fs):
    """regular / bold / superscript / heading / footnote faces for one family.

    Bold comes from the variable weight axis where the family has one, else from
    a -Bold.ttf beside the regular file. A family with neither gets `bold` None,
    and callers fall back to the regular face rather than faking a weight.
    """
    R = ImageFont.Layout.RAQM
    reg = ImageFont.truetype(path, fs, layout_engine=R)
    bold = None
    try:
        b = ImageFont.truetype(path, fs, layout_engine=R)
        axes = b.get_variation_axes()
        if axes and any(b"eight" in a["name"] for a in axes):
            b.set_variation_by_axes([min(700, a["maximum"]) if b"eight" in a["name"]
                                     else a["default"] for a in axes])
            bold = b
    except Exception:
        bold = None
    if bold is None:
        alt = path[:-4] + "-Bold.ttf"
        if os.path.exists(alt):
            try:
                bold = ImageFont.truetype(alt, fs, layout_engine=R)
            except Exception:
                bold = None
    return {"normal": reg, "bold": bold,
            "sup": ImageFont.truetype(path, max(10, int(fs * 0.62)), layout_engine=R),
            "head": ImageFont.truetype(path, int(fs * 1.35), layout_engine=R),
            "foot": ImageFont.truetype(path, int(fs * 0.78), layout_engine=R)}



def space_width(draw, font):
    """Word gap in pixels, with a floor.

    Some faces ship a broken space glyph -- Qahiri.ttf advances 0.00px for
    U+0020, and Alkalami effectively swallows it -- so drawing word by word
    with the font's own metric sets the whole line solid, and nothing on the
    page says where one word ends and the next begins. A floor of 0.30 of a
    meem keeps the word gap every printed page has.
    """
    sp = draw.textlength(" ", font=font, direction="rtl")
    ref = draw.textlength("\u0645", font=font, direction="rtl")   # meem
    return max(sp, 0.30 * ref)


def wrap_styled(draw, words, fonts, base, max_w):
    """Greedy RTL wrap over styled words. Arabic does not join across spaces,
    so measuring word by word is safe."""
    lines, cur, w_cur = [], [], 0.0
    space = space_width(draw, fonts[base])
    for word, style in words:
        if style == "break":
            if cur:
                lines.append(cur)
            cur, w_cur = [], 0.0
            continue
        f = fonts["sup"] if style == "sup" else (
            fonts["bold"] if style == "bold" and fonts["bold"] else fonts[base])
        wd = draw.textlength(word, font=f, direction="rtl")
        if cur and w_cur + space + wd > max_w:
            lines.append(cur)
            cur, w_cur = [(word, style)], wd
        else:
            if cur:
                w_cur += space
            cur.append((word, style))
            w_cur += wd
    if cur:
        lines.append(cur)
    return lines, space
