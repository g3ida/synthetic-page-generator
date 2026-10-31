#!/usr/bin/env python3
"""Synthetic Arabic page generator that emits LAYOUT GROUND TRUTH.

Why this exists: general-purpose layout detectors rarely recognise an Arabic
footnote block, a bayt of poetry or a fahras row. Rather than hunt for a
detector that has seen classical Arabic books, generate pages where the boxes
are known BY CONSTRUCTION because the generator placed every block itself.

Every region carries its reading order, assigned as it is drawn: columns
right-to-left, top-to-bottom inside a column, articles top-to-bottom. That is
the signal a detector built for left-to-right pages gets wrong.

Text is real -- sampled from a pool of passages taken from books (prose,
headings, footnotes, table rows, verse, list items) -- so glyph statistics,
tashkeel density and line lengths match real books rather than lorem ipsum.
"""
import math
import os
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import assets
from .regions import Region
from .scan import FINE_STROKE, rng_normal
from .shaping import drawable, font_bundle, font_cmap, strip_tashkeel, wrap_styled

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_SNIPPETS = ["Cairo", "Brill", "MS. 214", "cf. Gibb", "Leiden", "vol. II",
                  "op. cit.", "EI2", "s.v.", "no. 47", "Wensinck", "GAL I"]
# Real page geometries. A fixed 1024x1536 teaches a detector that the page is
# always that shape; scans arrive at every size and DPI.
PAGE_SIZES = [(1024, 1536), (900, 1350), (1240, 1754), (800, 1200),
              (1400, 2000), (1180, 1620), (1600, 2200), (760, 1140)]


# Section separators as Arabic books actually set them. Kept as text so they
# inherit the page's face and degradation rather than being drawn as vectors.
ORNAMENTS = ["* * *", "* * * * *", "٭ ٭ ٭", "* * * * * * *",
             "۞", "﴿ ۞ ﴾", "❊ ❊ ❊", "◆ ◆ ◆", "✿ ✿ ✿",
             ". . . . . . . . . . . . . . . . . . . .",
             "..............................",
             "ـــــــــــــــــــــــــــــ"]

# Faces used for a display banner (basmala, chapter opener): the cursive and
# calligraphic families. The first six are FINE_STROKE, the faces kept out of
# stroke thickening because their counters close under a MinFilter.
CALLIGRAPHIC = ("Nastaliq", "Gulzar", "ArefRuqaa", "Qahiri", "Alkalami",
                "Amiri", "Lotus", "trado", "majalla", "Naskh")

BANNERS = ["بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيمِ",
           "وَبِهِ نَسْتَعِينُ", "الحَمْدُ لِلَّهِ رَبِّ العَالَمِينَ",
           "كِتَابُ الطَّهَارَةِ", "بَابُ الصَّلَاةِ", "فَصْلٌ"]

STAMP_TEXTS = [["مكتبة", "الأوقاف"], ["دار الكتب", "المصرية"],
               ["وقف", "لله تعالى"], ["المكتبة", "الأزهرية"],
               ["خزانة", "القرويين"], ["مكتبة", "عارف حكمت"],
               ["المكتبة", "الظاهرية"], ["وقف على", "طلبة العلم"],
               ["مكتبة", "چester Beatty"], ["فهرس", "المخطوطات"],
               ["رقم", "١٤٧٢"], ["مراجعة", "التصحيح"]]

# Illumination palette: the pigments Arabic manuscript decoration actually uses
# -- lapis, vermilion, verdigris, gold, and the reds/browns of iron-gall inks.
ILLUM = [(38, 78, 152), (176, 58, 38), (52, 122, 84), (186, 145, 42),
         (120, 40, 96), (196, 108, 32), (30, 30, 30), (140, 30, 40)]

# Chart series colours. Muted enough to survive greyscale reproduction, which is
# how most of these pages were actually printed.
CHART_COLORS = [(46, 96, 148), (192, 96, 58), (86, 138, 88), (176, 148, 54),
                (128, 84, 140), (72, 140, 152), (176, 92, 122), (110, 110, 110)]

CHART_LABELS = ["الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس",
                "الحجاز", "الشام", "مصر", "العراق", "المغرب", "اليمن",
                "النحو", "الفقه", "الحديث", "التفسير", "اللغة", "المنطق"]

CHART_TITLES = ["توزيع المخطوطات حسب القرن", "عدد النسخ في كل مكتبة",
                "نسبة الأبواب في الكتاب", "إحصاء الرواة", "مقارنة الطبعات",
                "أعداد المسائل في كل باب"]

STAMP_SHAPES = ["oval", "circle", "rect", "double_rect", "rosette", "shield",
                "text_only", "oval"]

# Ink colours real seals are struck in. Violet aniline and faded red dominate
# in early 20th-century Arabic library practice; black is a modern rubber stamp.
STAMP_INKS = [(120, 30, 30), (30, 50, 120), (40, 40, 40), (95, 45, 105),
              (150, 60, 40), (35, 85, 60), (70, 60, 130), (160, 100, 30)]

WATERMARKS = ["نسخة إلكترونية", "للاطلاع فقط", "مكتبة المصطفى", "غير مخصص للبيع",
              "نسخة مصورة", "www.waqfeya.net", "PDF", "مكتبة صيد الفوائد"]


_POOL_CACHE = {}


def pool_for_font(pool, cmap, key):
    """Text this particular face can actually draw.

    PIL renders a missing glyph as a blank or a .notdef box and says nothing, so
    a page can look finished and carry tofu. Real Arabic text is full of
    characters no single face covers: U+FDFA (the ligature for صلى الله عليه
    وسلم) is missing from many faces, the ornate Quranic parentheses
    U+FD3E/FD3F from several, and Farsi yeh and Persian digits from more.

    Rather than strip those characters -- which would quietly change the text,
    so the pages no longer read like the books they imitate -- each page keeps
    only the passages ITS chosen face can render. Cached per face.
    """
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    def ok(t):
        return drawable(t, cmap)
    out = {}
    for k, v in pool.items():
        if not v:
            out[k] = v
            continue
        if isinstance(v[0], str):
            keep = [t for t in v if ok(t)]
        else:                                   # table rows: list of cells
            keep = [t for t in v if all(ok(c) for c in t)]
        # never let a category empty out -- a face that can draw almost nothing
        # should produce fewer pages, not crash the generator
        out[k] = keep if len(keep) >= 8 else v
    _POOL_CACHE[key] = out
    return out


def degrade_tracked(im, rng, font_name="", max_tier=None, color=False):
    """Make a clean render look scanned, and say how far it was rotated.

    Four tiers rather than independent per-effect draws, because independent
    draws almost never leave a page genuinely clean, and clean pages exist:

      clean  20%  untouched
      light  40%  sub-degree skew, faint blur, little noise -- a good flatbed scan
      medium 30%  visible blur and grain, ink weight drift -- a routine scan
      heavy  10%  speckle and blur -- a poor but READABLE copy

    The rotation angle is RETURNED, so region boxes can be rotated with the
    page: boxes recorded before the skew, on a page rotated after them, sit a
    few pixels off their text -- worst at the page corners, where the lever arm
    is longest. Nothing non-rigid is applied: a rotation is invertible, a
    displacement field is not, and no box could follow one.
    """
    tier = rng.choices(["clean", "light", "medium", "heavy"],
                       weights=[20, 40, 30, 10], k=1)[0]
    if max_tier:
        order = ["clean", "light", "medium", "heavy"]
        if order.index(tier) > order.index(max_tier):
            tier = max_tier
    if tier == "clean":
        return im, tier, 0.0
    cfg = {"light":  dict(rot=0.25, blur=(0.0, 0.45), noise=(1.0, 3.0),
                          gamma=(0.94, 1.08), speckle=0.0, morph=0.0),
           "medium": dict(rot=0.6,  blur=(0.3, 0.7),  noise=(2.0, 4.5),
                          gamma=(0.88, 1.16), speckle=0.12, morph=0.10),
           "heavy":  dict(rot=0.9,  blur=(0.45, 0.9), noise=(3.0, 6.0),
                          gamma=(0.84, 1.22), speckle=0.30, morph=0.16)}[tier]
    # Colour scans exist and are common for anything printed after ~1980.
    # Working in L would wash out every chart, tint and illuminated ornament
    # on the page, so colour is kept on request.
    if not color:
        im = im.convert("L")
    ang = rng.uniform(-cfg["rot"], cfg["rot"])
    im = im.rotate(ang, resample=Image.BICUBIC, fillcolor=255, expand=False)
    im = im.point(lambda v, g=rng.uniform(*cfg["gamma"]): min(255, int(255 * (v / 255) ** g)))
    if not any(k in font_name for k in FINE_STROKE) and rng.random() < cfg["morph"]:
        # MinFilter thickens (over-inked), MaxFilter thins. Thinning is the
        # destructive one for Arabic: a 1px stroke under MaxFilter(3) is erased
        # outright, and the FINE_STROKE exemption only names six families, so
        # any other thin face slips past it. Bias hard toward thickening and
        # let the contrast guard below catch what still goes wrong.
        im = im.filter(ImageFilter.MinFilter(3) if rng.random() < 0.85
                       else ImageFilter.MaxFilter(3))
    b = rng.uniform(*cfg["blur"])
    if b > 0.05:
        im = im.filter(ImageFilter.GaussianBlur(b))
    arr = np.asarray(im, dtype=np.int16)
    a = arr + rng_normal(arr.shape, rng, rng.uniform(*cfg["noise"]))
    if rng.random() < cfg["speckle"]:
        g = np.random.default_rng(rng.getrandbits(32))
        m = g.random(a.shape[:2]) < rng.uniform(0.0003, 0.0012)
        if a.ndim == 3:
            a[m] = g.integers(0, 90, (int(m.sum()), a.shape[2]))
        else:
            a[m] = g.integers(0, 90, int(m.sum()))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8),
                          "RGB" if a.ndim == 3 else "L")
    return out, tier, ang


def speckle(im, rng):
    """Patchy dust and isolated blobs, the way a dirty platen actually prints.

    `degrade` adds UNIFORM noise, which is not what real scans look like: dust
    clusters, and a few large blobs sit far from the rest. The isolated blobs
    matter most: if the only dark round marks on empty paper are seals, a
    detector learns that any such mark is a stamp. No region is emitted: dirt
    is not content.
    """
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    # a few dense patches rather than an even wash
    for _ in range(rng.randint(2, 6)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rad = min(W, H) * rng.uniform(0.06, 0.30)
        for _ in range(rng.randint(40, 260)):
            a = rng.uniform(0, 6.283)
            rr = rad * (rng.random() ** 0.5)
            x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
            r = rng.uniform(0.4, 1.8)
            g = rng.randint(40, 150)
            d.ellipse([x - r, y - r, x + r, y + r],
                      fill=(g, g, g, rng.randint(90, 220)))
    # isolated blobs -- these are what get mistaken for seals
    for _ in range(rng.randint(0, 5)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = min(W, H) * rng.uniform(0.004, 0.013)
        g = rng.randint(10, 70)
        d.ellipse([x - r, y - r * rng.uniform(0.6, 1.4),
                   x + r * rng.uniform(0.6, 1.4), y + r],
                  fill=(g, g, g, rng.randint(150, 245)))
    return im


def scan_artifacts(im, rng, kinds):
    """Damage a real scanner leaves. None of it produces a region: it is not
    content, and a detector that boxes a punch hole as a `figure` is worse than
    one that ignores it."""
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    if "gutter" in kinds:
        # Binding shadow: the page curves into the spine and goes dark.
        side = rng.choice(["l", "r"])
        gw = int(W * rng.uniform(0.05, 0.13))
        for i in range(gw):
            a = int(150 * (1 - i / gw) ** 2)
            x = i if side == "l" else W - 1 - i
            d.line([(x, 0), (x, H)], fill=(0, 0, 0, a))
    if "edge" in kinds:
        # Scanner lid open: black beyond the paper edge.
        t = int(min(W, H) * rng.uniform(0.012, 0.045))
        for e in rng.sample(["t", "b", "l", "r"], rng.randint(1, 2)):
            box = {"t": [0, 0, W, t], "b": [0, H - t, W, H],
                   "l": [0, 0, t, H], "r": [W - t, 0, W, H]}[e]
            d.rectangle(box, fill=(rng.randint(0, 40),) * 3 + (255,))
    if "holes" in kinds:
        side = rng.choice([0.045, 0.955])
        for k in range(rng.randint(2, 3)):
            cy = H * (0.25 + 0.25 * k) + rng.uniform(-20, 20)
            r = int(min(W, H) * 0.016)
            d.ellipse([side * W - r, cy - r, side * W + r, cy + r],
                      fill=(255, 255, 255, 255), outline=(90, 90, 90, 255), width=2)
    if "stain" in kinds:
        for _ in range(rng.randint(1, 3)):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            r = rng.uniform(0.04, 0.14) * min(W, H)
            d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8],
                      fill=(150, 120, 70, rng.randint(18, 46)))
    if "tear" in kinds:
        cx = rng.choice([0, W]); cy = rng.choice([0, H])
        sz = rng.uniform(0.06, 0.16) * min(W, H)
        pts = [(cx, cy),
               (cx + (sz if cx == 0 else -sz), cy),
               (cx, cy + (sz if cy == 0 else -sz))]
        d.polygon(pts, fill=(255, 255, 255, 255), outline=(120, 120, 120, 255))
    return im


def text_contrast(gray, regions):
    """Median (paper - ink) across text regions. The number that says whether a
    human could still read the page."""
    TXT = {"paragraph", "footnote", "hashiya", "header", "footer", "page_number",
           "heading", "verse", "list"}
    vals = []
    for r in regions:
        if r.type not in TXT:
            continue
        x0, y0, x1, y1 = r.bbox
        p = gray[y0:y1, x0:x1]
        if p.size < 200:
            continue
        # Measure against the region's OWN ink, not a fixed percentile. A wide
        # box around a short centred title is mostly paper, so a 2nd-percentile
        # test reads it as blank and reports contrast 0 on a perfectly good
        # heading.
        ink = p[p < 200]
        if ink.size < 40:
            continue
        vals.append(float(np.percentile(p, 90)) - float(np.median(ink)))
    return float(np.median(vals)) if vals else 255.0


class BadPage(Exception):
    """This page came out wrong; render another one instead of shipping it.

    A page whose boxes contradict each other teaches the detector a
    contradiction, and a page costs milliseconds to redraw.
    """


def ink_frac(gray, bbox, delta=40):
    """Fraction of a box that differs from its OWN background.

    A plain `pixel < 200` test assumes white paper: on a dark cover ground every
    pixel counts as ink, so a title the cover fill had painted over would still
    pass as inked and ship an empty heading box.
    """
    x0, y0, x1, y1 = (int(v) for v in bbox)
    p = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if p.size < 200:
        return 0.0
    return float((np.abs(p.astype(np.int16) - int(np.median(p))) > delta).mean())


# Classes whose boxes may never overlap each other: two of them on the same ink
# is a contradiction in the ground truth. Furniture (stamp, ornament, figure) is
# left out - a seal really does land on top of text in a scanned book.
NO_OVERLAP = {"paragraph", "heading", "footnote", "header", "footer",
              "page_number", "caption", "text", "table", "toc", "verse", "list",
              "hashiya", "calligraphy"}


def resolve_overlaps(regions, tol=0.02, trim_frac=0.30):
    """Keep text boxes off one another; raise BadPage when that is not possible.

    Blocks that advance by a nominal leading instead of by the ink they drew
    end up overlapping - title over subtitle, banner over the column beneath
    it, a folio inside a paragraph. A thin overlap is trimmed off the lower
    box; anything deeper means the two blocks were genuinely drawn on top of
    each other.
    """
    rs = sorted([r for r in regions if r.type in NO_OVERLAP],
                key=lambda r: (r.bbox[1], r.bbox[0]))
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            ax0, ay0, ax1, ay1 = a.bbox
            bx0, by0, bx1, by1 = b.bbox
            ow, oh = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if ow <= 0 or oh <= 0:
                continue
            small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
            if not small or (ow * oh) / small <= tol:
                continue
            # b starts lower (sorted): pull its top below a, but only when that
            # costs b less than a third of its height and leaves a real block.
            hb = by1 - by0
            if ay1 < by1 and (ay1 - by0) <= trim_frac * hb and (by1 - ay1) > 8:
                b.bbox = [bx0, ay1 + 1, bx1, by1]
                continue
            raise BadPage(f"{a.type} and {b.type} overlap "
                          f"{(ow * oh) / small:.0%} of the smaller box")


def rotate_pts(pts, cx, cy, ang):
    """Where content lands after PIL's im.rotate(ang): R(-ang) about the centre.
    Verified empirically against a fiducial dot to under 0.5 px."""
    t = math.radians(-ang); c, s_ = math.cos(t), math.sin(t)
    return [[cx + (x - cx) * c - (y - cy) * s_,
             cy + (x - cx) * s_ + (y - cy) * c] for x, y in pts]

# Title colours seen in printed classical Arabic books: rubrication (red), and
# the dark blue/green used by modern Beirut editions. Never a full-saturation
# web colour.
TITLE_COLORS = [(0, 0, 0), (150, 26, 26), (26, 60, 130), (20, 90, 60), (120, 60, 10)]

# Magazine / modern-edition article tints. Kept pale: a wash behind body text in
# a real publication is a tint, not a fill, and a detector must not learn that
# "coloured rectangle" means "figure".
BAND_TINTS = [(247, 243, 233), (238, 244, 248), (245, 240, 245), (240, 247, 240),
              (250, 245, 235), (243, 243, 246)]
# Borderless tables carry their structure in alternating row wash instead of rules.
# Saturated full-bleed grounds for covers -- real covers are printed in flat
# spot colours, which is nothing like the paper-white the generator otherwise
# produces.
# Real furniture -- transparent cut-outs, grime overlays, calligraphy -- comes
# from assets.py. Without transparent art every figure is an opaque rectangle,
# and a detector learns "figure = bordered box" and misses cartoon cut-outs on
# bare paper, photographs bleeding into the page, art with no frame to find.

# Pale grounds for tinted pages -- a coloured stock the text sits on. Distinct
# from COVER_GROUNDS, which are saturated full-bleed cover colours.
PAGE_TINTS = [(252, 246, 228), (245, 240, 250), (240, 248, 244),
              (253, 241, 235), (238, 244, 252), (250, 250, 235),
              (246, 236, 236), (236, 246, 246)]

COVER_GROUNDS = [(28, 122, 58), (18, 74, 132), (140, 26, 34), (196, 148, 44),
                 (44, 46, 52), (238, 232, 214), (92, 40, 110), (12, 96, 104)]
TABLE_TINTS = [(240, 244, 249), (246, 243, 236), (243, 246, 241), (247, 241, 241)]

# ---------------------------------------------------------------------------
# `text` = standalone text that is not a title, by WHAT THE LINE IS. On real
# pages it is an imprint block, a URL, an ownership inscription or a citation
# line. These pools give the class that content.
# ---------------------------------------------------------------------------
PUBLISHERS = ["دار الكتب العلمية", "دار السلام للطباعة والنشر", "مؤسسة الرسالة",
              "دار ابن حزم", "المكتبة العصرية", "دار الفكر", "دار الغرب الإسلامي",
              "دار المنهاج", "مكتبة الرشد", "دار طيبة للنشر والتوزيع"]
IMPRINT_CITIES = ["بيروت - لبنان", "القاهرة - مصر", "الرياض - السعودية",
                  "دمشق - سوريا", "عمّان - الأردن", "الدار البيضاء - المغرب",
                  "تونس - تونس", "بغداد - العراق"]
IMPRINT_DOMAINS = ["dar-alsalam.com", "ktab-online.com", "alresalah-pub.com",
                   "maktaba-asriya.com", "ibnhazm-books.com", "almanhaj-pub.com"]
INSCRIPTIONS = ["مِنْ كُتُبِ الفَقِيرِ إِلَى اللهِ تَعَالَى", "وَقْفٌ لِلَّهِ تَعَالَى",
                "مَلَكَهُ بِالشِّرَاءِ الشَّرْعِيِّ", "بِسْمِ اللهِ وَبِهِ نَسْتَعِينُ",
                "الحَمْدُ لِلَّهِ الَّذِي هَدَانَا لِهَذَا"]
BLURB_OPENERS = ["هذا الكتاب", "يتناول هذا الكتاب", "دراسة تهدف إلى",
                 "بين يديك", "كتاب في"]


def imprint_block(rng, cmap=None):
    """An imprint / colophon: publisher, city, phone, e-mail, site. `text`."""
    pub = rng.choice(PUBLISHERS)
    dom = rng.choice(IMPRINT_DOMAINS)
    phone = "هاتف: " + arabize(rng.randint(1000000, 9999999), rng, cmap)
    out = [pub, rng.choice(IMPRINT_CITIES)]
    if rng.random() < 0.7:
        out.append(phone)
    lines = [out]
    web = []
    if rng.random() < 0.8:
        web.append("www." + dom)
    if rng.random() < 0.5:
        web.append("email:info@" + dom)
    if web:
        lines.append(web)          # a separate block: printed apart from the address
    return lines


CITATION_MARKS = ("أخرجه", "رواه", "انظر:", "ينظر:", "صحيح البخاري", "سنن",
                  "المصدر السابق", "تقدم تخريجه")


def is_citation(s):
    """A citation / reference line: `text`, never a heading, whatever its size."""
    s = s.strip()
    if not s:
        return False
    if s[0] in "(([" and any(ch.isdigit() or "٠" <= ch <= "٩" for ch in s[:6]):
        return True
    return any(k in s for k in CITATION_MARKS)


def split_heads(heads):
    """The `heads` pool is scraped text and mixes true titles with citation
    lines. Separating them is what makes the `text`/`heading` distinction real
    in the ground truth instead of a coin flip."""
    titles = [h for h in heads if not is_citation(h)]
    cites = [h for h in heads if is_citation(h)]
    return titles or heads, cites


def arabize(n, rng, cmap=None, force=None):
    """Page/list numerals: Arabic-Indic or Western. Both occur in real books.

    Falls back to Western when the face has no Arabic-Indic digit glyphs: PIL
    draws a missing U+0660-0669 as nothing at all, which would leave a page
    number box over blank paper.
    """
    s = str(n)
    use_ar = rng.random() < 0.6 if force is None else force
    if not use_ar:
        return s
    ar = "".join(ARABIC_DIGITS[int(c)] for c in s)
    if cmap is not None and not drawable(ar, cmap):
        return s
    return ar


def draw_line_c(d, line, fonts, base, x_right, y, fs, space, fill=(0, 0, 0)):
    """Draw one wrapped line right to left, in a colour: headings, rubrication
    and cover text are not always black."""
    x = x_right
    for word, style in line:
        f = fonts["sup"] if style == "sup" else (
            fonts["bold"] if style == "bold" and fonts["bold"] else fonts[base])
        d.text((x, y), word, font=f, fill=fill, direction="rtl", anchor="ra")
        x -= d.textlength(word, font=f, direction="rtl") + space


def line_w(d, line, fonts, base, space):
    """Ink width of one wrapped line, matching draw_line_c's advance exactly."""
    w = 0.0
    for word, style in line:
        f = fonts["sup"] if style == "sup" else (
            fonts["bold"] if style == "bold" and fonts["bold"] else fonts[base])
        w += d.textlength(word, font=f, direction="rtl") + space
    return max(0.0, w - space)


def hbox(d, lines, fonts, base, space, x_right, y0, y1, pad=6):
    """Tight box around CENTRED/right-aligned heading lines.

    A heading is drawn centred or flush right, so its ink fills only part of
    the column. Boxing the full column `(mx_l, y0, W - mx_r, y)` would get the
    height right and the width far too wide, so the box follows the ink.
    draw_line_c lays each line out ending at x_right and advancing leftwards,
    so a line spans [x_right - w, x_right].
    """
    w = max((line_w(d, l, fonts, base, space) for l in lines), default=0.0)
    return (x_right - w - pad, y0 - pad, x_right + pad, y1 + pad)


class Page:
    """Draws blocks and records a box + reading order for each."""

    def __init__(self, W, H, rng):
        self.W, self.H, self.rng = W, H, rng
        self.im = Image.new("RGB", (W, H), (255, 255, 255))
        self.d = ImageDraw.Draw(self.im)
        self.regions, self._order = [], 0
        self.ink = (0, 0, 0)      # page ink colour; light on a dark cover
        self.cmap = None
        self.figure_paths = []
        self.map_font = None
        self.map_names = ["دمشق", "بغداد", "القاهرة", "قرطبة", "نيسابور", "البصرة",
                          "حلب", "فاس", "سمرقند", "الموصل", "طنجة", "شيراز"]

    def add(self, bbox, rtype, column=None):
        x0, y0, x1, y1 = bbox
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        self.regions.append(Region(id=len(self.regions),
                                   bbox=[max(0, x0), max(0, y0),
                                         min(self.W, x1), min(self.H, y1)],
                                   type=rtype, score=1.0, native=rtype,
                                   order=self._order, column=column))
        self._order += 1

    def hr(self, x0, x1, y, width=2):
        self.d.line([(x0, y), (x1, y)], fill=(0, 0, 0), width=width)
        self.add((x0, y - width - 1, x1, y + width + 1), "rule")
        return y + width + 2

    def vrule(self, x, y0, y1):
        """Vertical separator between columns. No region: it is furniture."""
        self.d.line([(x, y0), (x, y1)], fill=(90, 90, 90), width=1)

    # ---- block drawers. each returns the new y, or None if it did not fit ----

    def para(self, text, fonts, xr, colw, y, fs, lead, max_y, rtype="paragraph",
             base="normal", fill=(0, 0, 0), col=None, justify=True):
        words = [(w, "normal") for w in text.split()]
        if any(self.d.textlength(w, font=fonts[base], direction="rtl") > colw
               for w, _ in words):
            return None           # unbreakable word wider than the column
        lines, space = wrap_styled(self.d, words, fonts, base, colw)
        if not lines or y + lead * len(lines) > max_y:
            return None
        y0 = y
        for i, ln in enumerate(lines):
            jw = colw if (justify and i < len(lines) - 1 and len(ln) > 2) else None
            if jw:
                ink = sum(self.d.textlength(w, font=fonts[base], direction="rtl")
                          for w, _ in ln)
                wide = (jw - ink) / max(1, len(ln) - 1)
                sp = wide if space <= wide <= space * 3 else space
            else:
                sp = space
            draw_line_c(self.d, ln, fonts, base, xr, y, fs, sp, fill)
            y += lead
        self.add((xr - colw, y0 - lead // 5, xr, y), rtype, col)
        return y + lead // 3

    def poem(self, baits, fonts, xr, colw, y, fs, lead, max_y, style, col=None):
        """A whole poem as ONE region -- that is how a reader sees it.

        Real verse height is bimodal -- a single quoted bayt, or a whole-page
        poem -- so callers draw both ends of that range, not just the middle.

        style 'inline'  sadr and ajz share a baseline, two centred half-columns.
        style 'offset'  ajz on the next line, indented -- the other convention.
        """
        y0, half, gut = y, colw * 0.44, colw * 0.10
        drew = 0
        # One region per poem: a poem reads as one block and is boxed as one.
        # A short verse box comes from a short poem -- a one- or two-bayt
        # quotation inside prose -- not from splitting a long poem bayt by
        # bayt. The per-bayt spans are still recorded; `per_bayt` boxes each
        # bayt on its own instead.
        per_bayt = False
        spans = []
        for sadr, ajz in baits:
            need = lead if style == "inline" else lead * 2
            if y + need > max_y:
                break
            if style == "inline":
                # Both hemistichs must fit their half-column on ONE line, or the
                # bayt is not drawn at all. Advancing y for a bayt that did not
                # render would leave an empty `verse` box -- a box the image
                # does not back up, and a fault that only shows on the PNG.
                pair = []
                for txt, right in ((sadr, xr), (ajz, xr - half - gut)):
                    ln, sp = wrap_styled(self.d, [(w, "normal") for w in txt.split()],
                                         fonts, "normal", half)
                    if len(ln) != 1:
                        pair = None
                        break
                    w = sum(self.d.textlength(t, font=fonts["normal"], direction="rtl")
                            for t, _ in ln[0]) + sp * (len(ln[0]) - 1)
                    pair.append((ln[0], sp, right - (half - w) / 2))
                if not pair:
                    continue
                bs = y
                for ln0, sp, x in pair:
                    draw_line_c(self.d, ln0, fonts, "normal", x, y, fs, sp)
                y += lead
                spans.append((bs, y))
                drew += 1
            else:
                pair = []
                for k, txt in enumerate((sadr, ajz)):
                    ln, sp = wrap_styled(self.d, [(w, "normal") for w in txt.split()],
                                         fonts, "normal", colw * 0.8)
                    if len(ln) != 1:
                        pair = None
                        break
                    pair.append((ln[0], sp, xr - (0 if k == 0 else colw * 0.18)))
                if not pair:
                    continue
                bs = y
                for ln0, sp, x in pair:
                    draw_line_c(self.d, ln0, fonts, "normal", x, y, fs, sp)
                    y += lead
                spans.append((bs, y))
                drew += 1
        if not drew:
            return None
        if per_bayt and len(spans) > 1:
            for a, b in spans:
                self.add((xr - colw, a - lead // 6, xr, b), "verse", col)
        else:
            self.add((xr - colw, y0 - lead // 4, xr, y), "verse", col)
        return y + lead // 2

    def listing(self, items, fonts, xr, colw, y, fs, lead, max_y, ordered, col=None):
        y0 = y
        for i, it in enumerate(items, 1):
            mark = f"{arabize(i, self.rng, self.cmap)}." if ordered else "–"
            words = [(w, "normal") for w in (mark + " " + it).split()]
            if any(self.d.textlength(w, font=fonts["normal"], direction="rtl") > colw
                   for w, _ in words):
                continue          # one word wider than the column: would overflow
            lines, sp = wrap_styled(self.d, words, fonts, "normal", colw * 0.94)
            if y + lead * len(lines) > max_y:
                break
            for j, ln in enumerate(lines):
                draw_line_c(self.d, ln, fonts, "normal",
                            xr - (0 if j == 0 else colw * 0.05), y, fs, sp)
                y += lead
        if y == y0:
            return None
        # Labelled `paragraph`: on real pages a list is not reliably told apart
        # from the prose around it, so it is not a class of its own. The BLOCK
        # is still drawn for visual variety.
        self.add((xr - colw, y0 - lead // 4, xr, y), "paragraph", col)
        return y + lead // 3

    def toc(self, rows, fonts, xr, colw, y, fs, lead, max_y, col=None,
            style=None):
        """The fahras, in the three shapes real books use.

        style 'dots'       title ......... number   (leader dots)
        style 'ruled'      a bordered two-column table
        style 'borderless' the same, carried by an alternating row wash
        """
        style = style or self.rng.choices(["dots", "ruled", "borderless"],
                                          weights=[0.6, 0.2, 0.2], k=1)[0]
        y0 = y
        for rowi, cells in enumerate(rows):
            if y + lead > max_y:
                break
            title, num = cells[0], cells[-1]
            while title and self.d.textlength(title, font=fonts["normal"],
                                              direction="rtl") > colw * 0.62:
                title = title[:-1]
            if len(title) < 4:
                continue
            nw = self.d.textlength(num, font=fonts["normal"], direction="rtl")
            self.d.text((xr - colw + nw, y), num, font=fonts["normal"],
                        fill=(0, 0, 0), direction="rtl", anchor="ra")
            tw = self.d.textlength(title, font=fonts["normal"], direction="rtl")
            self.d.text((xr, y), title, font=fonts["normal"], fill=(0, 0, 0),
                        direction="rtl", anchor="ra")
            gap0, gap1 = xr - colw + nw + 8, xr - tw - 8
            if style == "dots" and gap1 > gap0:
                self.d.text(((gap0 + gap1) / 2, y), "." * int((gap1 - gap0) / 6),
                            font=fonts["normal"], fill=(120, 120, 120), anchor="ma")
            elif style == "ruled":
                self.d.rectangle([xr - colw, y - 3, xr, y + lead - 3],
                                 outline=(60, 60, 60), width=1)
            elif style == "borderless" and (rowi % 2 == 0):
                self.d.rectangle([xr - colw, y - 3, xr, y + lead - 3],
                                 fill=None, outline=None)
            y += lead
        if y == y0:
            return None
        self.add((xr - colw, y0 - lead // 4, xr, y), "toc", col)
        return y + lead // 3

    def table(self, rows, fonts, xr, colw, y, fs, lead, max_y, col=None,
              style="ruled", tint=None):
        """style 'ruled' draws a grid; 'borderless' carries the structure in an
        alternating row wash instead. The borderless form is what modern Beirut
        editions and magazines use, and it is materially harder to detect --
        there are no lines for an edge detector to find, only alignment."""
        ncell = max(len(r) for r in rows)
        cw = colw / ncell
        y0, rows = y, rows[:max(2, int((max_y - y) // lead))]
        if len(rows) < 2:
            return None
        if style == "borderless" and tint:
            self.d.rectangle([xr - colw, y - 4, xr, y + lead * len(rows) - 3],
                             fill=tuple(min(255, c + 6) for c in tint))
        for ri, r in enumerate(rows):
            if y + lead > max_y:
                break
            if style == "borderless" and tint and ri % 2 == 0:
                self.d.rectangle([xr - colw, y - 3, xr, y + lead - 3], fill=tint)
            for ci in range(ncell):
                cx = xr - ci * cw
                txt = r[ci] if ci < len(r) else ""
                # Trim to the CELL, measured, not to a fixed character count --
                # 26 chars of Amiri and 26 of Cairo are very different widths and
                # the fixed cut let text run out past the page margin.
                while txt and self.d.textlength(txt, font=fonts["foot"],
                                                direction="rtl") > cw - 10:
                    txt = txt[:-1]
                self.d.text((cx - 4, y), txt, font=fonts["foot"],
                            fill=(0, 0, 0), direction="rtl", anchor="ra")
                if style == "ruled":
                    self.d.rectangle([cx - cw, y - 3, cx, y + lead - 3],
                                     outline=(60, 60, 60), width=1)
            y += lead
        self.add((xr - colw, y0 - 4, xr, y), "table", col)
        return y + lead // 3

    def hashiya(self, texts, font_path, x0, x1, y0, y1, fs, rotated):
        """Marginal gloss in the outer margin, beside the matn.

        Not a footnote: a footnote sits below a rule at the foot of the page and
        is part of the text block's own column. A hashiya runs down the OUTER
        MARGIN alongside the matn, is set smaller, and is very often rotated 90
        degrees so the reader turns the book to read it. That rotation is why no
        general detector will ever find it -- rotated text does not look like a
        text region to a model trained on upright Latin documents.

        Drawn into its own canvas and pasted, which is the only way to get
        rotated text out of PIL, and which keeps the box exact either way.
        """
        w, h = int(x1 - x0), int(y1 - y0)
        if w < 30 or h < 60:
            return False
        sfs = max(11, int(fs * 0.62))
        sf = font_bundle(font_path, sfs)
        lead_s = int(sfs * 1.5)
        # When rotated, the gloss is typeset along the LONG axis: compose on a
        # canvas of (h x w) and turn it, so wrapping uses the page height.
        cw, ch = (h, w) if rotated else (w, h)
        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
        cd = ImageDraw.Draw(canvas)
        # A real hashiya is 2-5 SHORT notes scattered down the margin with big
        # blank gaps, not a solid column of text. Filling the margin would give
        # one box most of the page tall, a shape real glosses never have.
        rng = self.rng
        yy, drew = rng.randint(4, max(5, ch // 6)), False
        want = rng.randint(2, 5)
        for t in texts[:want]:
            words = [(a, "normal") for a in t.split()[:rng.randint(2, 9)]]
            if not words:
                continue
            if any(cd.textlength(a, font=sf["normal"], direction="rtl") > cw - 8
                   for a, _ in words):
                continue
            # Wrapping to the full reserved margin would make every note as
            # wide as the strip. A marginal gloss is a few words, not a
            # justified column.
            nw = max(24, int((cw - 8) * rng.uniform(0.30, 0.85)))
            lines, sp = wrap_styled(cd, words, sf, "normal", nw)
            if not lines or yy + lead_s * len(lines) > ch - 4:
                break
            for ln in lines:
                draw_line_c(cd, ln, sf, "normal", cw - 4, yy, sfs, sp)
                yy += lead_s
            drew = True
            yy += int(lead_s * rng.uniform(2.5, 7.0))      # the blank gap
            if yy > ch - lead_s * 2:
                break
        if not drew:
            return False
        if rotated:
            canvas = canvas.rotate(90, expand=True)
        self.im.paste(canvas, (int(x0), int(y0)))
        # Box the INK, not the whole reserved margin. Real glosses are short
        # notes scattered down the margin, so each drawn note is boxed
        # separately rather than the column as a whole.
        g = np.asarray(canvas.convert("L"))
        rows = np.where((g < 200).any(axis=1))[0]
        if rows.size == 0:
            return False
        # split the column into runs of ink separated by blank bands
        breaks, run_start = [], rows[0]
        for a, b in zip(rows, rows[1:]):
            if b - a > max(6, lead_s):
                breaks.append((run_start, a)); run_start = b
        breaks.append((run_start, rows[-1]))
        # Per-note x extent. Taking the columns over the whole strip would give
        # every note the width of the widest one.
        for r0, r1 in breaks:
            if r1 - r0 < 4:
                continue
            sub = np.where((g[r0:r1 + 1] < 200).any(axis=0))[0]
            if sub.size == 0:
                continue
            cx0, cx1 = int(sub[0]), int(sub[-1])
            self.add((x0 + cx0 - 3, y0 + r0 - 3, x0 + cx1 + 3, y0 + r1 + 3),
                     "hashiya")
        return True

    def make_map(self, w, h):
        """Procedural map: graticule, landmasses, routes, Arabic place labels.

        Generated rather than sourced because a map is visually unlike both a
        photograph and a block of text -- large pale areas, thin long strokes,
        sparse labels -- and that is exactly the texture a layout model needs to
        learn is still a `figure`. Historical atlases are common in classical
        Arabic books.
        """
        rng = self.rng
        im = Image.new("RGB", (w, h), (247, 243, 231))
        d = ImageDraw.Draw(im, "RGBA")
        for i in range(1, 6):                       # graticule
            d.line([(w * i / 6, 0), (w * i / 6, h)], fill=(170, 175, 165, 120))
            d.line([(0, h * i / 6), (w, h * i / 6)], fill=(170, 175, 165, 120))
        for _ in range(rng.randint(2, 4)):          # landmasses / provinces
            cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
            r = rng.uniform(0.12, 0.30) * min(w, h)
            pts = []
            for k in range(rng.randint(7, 12)):
                a = 2 * math.pi * k / 10
                rr = r * rng.uniform(0.6, 1.35)
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            tone = rng.choice([(214, 222, 200), (226, 214, 196), (206, 216, 222)])
            d.polygon(pts, fill=tone + (215,), outline=(110, 110, 100, 255))
        for _ in range(rng.randint(1, 3)):          # routes / rivers
            pts = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(4)]
            d.line(pts, fill=(120, 130, 150, 200), width=2)
        if self.map_font:
            for _ in range(rng.randint(3, 7)):      # settlements + labels
                x, y = rng.uniform(0.08, 0.9) * w, rng.uniform(0.08, 0.9) * h
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(60, 60, 60))
                lbl = rng.choice(self.map_names)
                d.text((x - 6, y - 4), lbl, font=self.map_font, fill=(40, 40, 40),
                       direction="rtl", anchor="ra")
        d.rectangle([0, 0, w - 1, h - 1], outline=(80, 80, 70), width=2)
        return im

    def ornament(self, fonts, x0, x1, y, fs, lead, max_y, text=None, col=None):
        """Centred section separator: * * *, dotted leader, or a calligraphic mark."""
        if y + lead * 1.6 > max_y:
            return None
        t = text or self.rng.choice(ORNAMENTS)
        if self.cmap is not None and not drawable(t, self.cmap):
            # Dingbats such as ❊ ◆ ✿ are missing from most Arabic faces and ۞
            # from many, so the fallback chain has to end somewhere every face
            # can reach.
            for alt in ("* * *", "· · ·", "- - -", ". . . . . . . ."):
                if drawable(alt, self.cmap):
                    t = alt
                    break
            else:
                return None
        # Real separators are WIDE, so a single "* * *" is far too small: the
        # token is repeated out to a sampled width and, three times in four,
        # set at display size.
        face = "head" if (self.rng.random() < 0.75 and fonts.get("head")) else "normal"
        f = fonts[face]
        target = (x1 - x0) * self.rng.uniform(0.05, 0.20)
        base = t
        for _ in range(14):
            w = self.d.textlength(t, font=f, direction="rtl")
            if w >= target:
                break
            t = t + " " + base
        w = self.d.textlength(t, font=f, direction="rtl")
        while w > (x1 - x0) and " " in t:
            t = t.rsplit(" ", 1)[0]
            w = self.d.textlength(t, font=f, direction="rtl")
        cx = (x0 + x1) / 2
        # One row of marks is thinner than many real separators. Some books set
        # a double row, which gets the height honestly rather than by padding a
        # box past its ink.
        rows = 2 if (self.rng.random() < 0.45 and y + lead * 2.6 < max_y) else 1
        yy = y
        boxes = []
        for _ in range(rows):
            self.d.text((cx + w / 2, yy), t, font=f, fill=(0, 0, 0),
                        direction="rtl", anchor="ra")
            boxes.append(self.d.textbbox((cx + w / 2, yy), t, font=f,
                                         direction="rtl", anchor="ra"))
            yy += int(f.size * 0.95)
        bb = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes))
        self.add((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), "ornament", col)
        return y + int(lead * 1.6) + (int(f.size * 0.95) if rows == 2 else 0)

    def banner(self, font_path, text, x0, x1, y, fs, max_y, fill=(0, 0, 0)):
        """Large display line in a calligraphic face -- a basmala or chapter opener."""
        size = int(fs * self.rng.uniform(1.8, 2.6))
        if y + size * 1.9 > max_y:
            return None
        try:
            bf = font_bundle(font_path, size)
        except Exception:
            return None
        w = self.d.textlength(text, font=bf["normal"], direction="rtl")
        if w > (x1 - x0) * 0.96:
            return None
        cx = (x0 + x1) / 2
        self.d.text((cx + w / 2, y), text, font=bf["normal"], fill=fill,
                    direction="rtl", anchor="ra")
        # The box and the y handed back to the caller follow the same bottom;
        # if the caller resumed above the box's foot, a banner set across the
        # head of a two-column page would overlap the column beneath it.
        bottom = y + int(size * 1.5)
        self.add((cx - w / 2 - 8, y - 8, cx + w / 2 + 8, bottom), "heading")
        return bottom + int(size * 0.45)

    def stamp(self, font_path, x, y, r, rng, shape=None):
        """Library / waqf ownership seal.

        Drawn on its own RGBA canvas and rotated, because a real stamp is never
        square to the page, then composited semi-transparently so the text
        underneath shows through. That overlap is the whole difficulty: a stamp
        sits ON content rather than in clean space, and a detector must not
        swallow the text beneath it into a `figure`.

        Worn coverage is modelled by punching random holes in the alpha channel
        -- a struck seal is never solid, and a uniformly opaque stamp is the
        giveaway of synthetic data.
        """
        shape = shape or rng.choice(STAMP_SHAPES)
        ink = rng.choice(STAMP_INKS)
        d2 = int(r * 2.4)
        c = Image.new("RGBA", (d2, d2), (0, 0, 0, 0))
        cd = ImageDraw.Draw(c)
        m, alpha = d2 // 2, rng.randint(95, 190)
        lw = max(2, r // 15)
        col = ink + (alpha,)
        if shape == "circle":
            cd.ellipse([m - r, m - r, m + r, m + r], outline=col, width=lw)
            cd.ellipse([m - r * .84, m - r * .84, m + r * .84, m + r * .84],
                       outline=col, width=max(1, lw // 2))
        elif shape == "oval":
            cd.ellipse([m - r, m - r * .74, m + r, m + r * .74], outline=col, width=lw)
            cd.ellipse([m - r * .86, m - r * .62, m + r * .86, m + r * .62],
                       outline=col, width=max(1, lw // 2))
        elif shape == "rect":
            cd.rectangle([m - r, m - r * .62, m + r, m + r * .62], outline=col, width=lw)
        elif shape == "double_rect":
            cd.rectangle([m - r, m - r * .66, m + r, m + r * .66], outline=col, width=lw)
            cd.rectangle([m - r * .88, m - r * .54, m + r * .88, m + r * .54],
                         outline=col, width=max(1, lw // 2))
        elif shape == "rosette":
            pts = []
            for k in range(48):                     # scalloped edge
                a = 2 * math.pi * k / 48
                rr = r * (0.86 + 0.14 * math.cos(8 * a))
                pts.append((m + rr * math.cos(a), m + rr * math.sin(a)))
            cd.polygon(pts, outline=col)
            for k in range(len(pts)):
                cd.line([pts[k], pts[(k + 1) % len(pts)]], fill=col, width=lw)
            cd.ellipse([m - r * .62, m - r * .62, m + r * .62, m + r * .62],
                       outline=col, width=max(1, lw // 2))
        elif shape == "shield":
            cd.polygon([(m - r * .8, m - r * .8), (m + r * .8, m - r * .8),
                        (m + r * .8, m + r * .2), (m, m + r * .9),
                        (m - r * .8, m + r * .2)], outline=col)
            cd.line([(m - r * .8, m - r * .8), (m + r * .8, m - r * .8),
                     (m + r * .8, m + r * .2), (m, m + r * .9),
                     (m - r * .8, m + r * .2), (m - r * .8, m - r * .8)],
                    fill=col, width=lw)
        # text_only draws no frame at all
        try:
            sf = font_bundle(font_path, max(9, int(r * (0.30 if shape == "text_only" else 0.23))))
            lines = rng.choice(STAMP_TEXTS)
            yy = m - int(r * 0.24)
            for ln in lines:
                if not drawable(ln, font_cmap(font_path)):
                    continue
                tw = cd.textlength(ln, font=sf["normal"], direction="rtl")
                cd.text((m + tw / 2, yy), ln, font=sf["normal"], fill=col,
                        direction="rtl", anchor="ra")
                yy += int(r * 0.30)
        except Exception:
            pass
        # worn ink: knock holes in the alpha so the strike is uneven
        if rng.random() < 0.75:
            a = np.asarray(c).copy()
            g = np.random.default_rng(rng.getrandbits(32))
            mask = g.random(a.shape[:2]) < rng.uniform(0.10, 0.40)
            a[..., 3] = np.where(mask, (a[..., 3] * 0.25).astype(a.dtype), a[..., 3])
            c = Image.fromarray(a, "RGBA")
        c = c.rotate(rng.uniform(-32, 32), resample=Image.BICUBIC, expand=True)
        px, py = int(x - c.width / 2), int(y - c.height / 2)
        self.im.paste(c, (px, py), c)
        # Box the INK, not the canvas. The canvas is square whatever shape was
        # drawn on it, while real seals are often oval or oblong; boxing the
        # canvas would label an oval or rectangular seal with a square box.
        ink = c.split()[-1].getbbox()
        if ink:
            self.add((px + ink[0], py + ink[1], px + ink[2], py + ink[3]), "stamp")
        else:
            self.add((px, py, px + c.width, py + c.height), "stamp")

    def calligraphy(self, path, x0, x1, y, fs, lead, max_y, col=None):
        """Paste a calligraphic panel into the text flow as `calligraphy`.

        A class of its own rather than `heading`: the panel is artwork, and
        typing it as a heading would label a framed piece as text. Two kinds of
        source are handled: pieces that already carry an alpha channel (clean
        line art, such as Calliar's) are pasted as they are, and opaque plates
        (such as DuwatBench's) have their background knocked out below.

        The background is knocked out to transparent so the piece sits on the
        paper rather than in a white box -- a printed calligraphic panel has no
        frame, and a visible rectangle would be a synthetic-data giveaway a
        detector could learn instead of the calligraphy.
        """
        try:
            src = Image.open(path)
            pre_alpha = src.mode == "RGBA"      # already cut out: paste as is
            src = src.convert("RGBA") if pre_alpha else src.convert("RGB")
        except Exception:
            return None
        avail_w = int((x1 - x0) * self.rng.uniform(0.55, 0.95))
        # A calligraphic panel is usually a small inset, not a full-column
        # plate, and its ink fills only part of its own rectangle -- so the
        # width is cut down again here and the box follows the ink, not the
        # paste rectangle.
        avail_w = int(avail_w * self.rng.uniform(0.30, 0.78))
        if avail_w < 40:
            return None
        sc = avail_w / src.width
        w, h = avail_w, max(1, int(src.height * sc))
        # A tall narrow panel can still come out far too tall after the width
        # cap, because the aspect ratio carries it. Cap against the page too.
        hcap = int(self.H * self.rng.uniform(0.05, 0.13))
        if h > hcap:
            w = max(1, int(w * hcap / h))
            h = hcap
        if h > (max_y - y) * 0.8 or h < 20:
            h = int(min(h, (max_y - y) * 0.8))
            if h < 20:
                return None
            w = max(1, int(src.width * (h / src.height)))
        src = src.resize((w, h), Image.LANCZOS)
        if pre_alpha:
            piece = src
            cx = (x0 + x1) / 2
            px, py = int(cx - w / 2), int(y)
            self.im.paste(piece, (px, py), piece)
            self.d = ImageDraw.Draw(self.im)
            ink = piece.split()[-1].getbbox()
            if ink is None:
                return None
            self.add((px + ink[0], py + ink[1], px + ink[2], py + ink[3]),
                     "calligraphy", col)
            return y + h + lead // 2
        # Knock out the piece's OWN background, not "white". Many plates sit on
        # cream or parchment; assuming white would leave a visible tan
        # rectangle on the page -- a synthetic-data giveaway a detector would
        # learn instead of the calligraphy. Estimate the ground from the border
        # and make anything at that level transparent.
        a = np.asarray(src.convert("L")).astype(np.int16)
        b = max(2, min(a.shape) // 20)
        bg = float(np.median(np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                                             a[:, :b].ravel(), a[:, -b:].ravel()])))
        alpha = np.clip((bg - a - 8) * 3.2, 0, 255).astype(np.uint8)
        rgba = np.dstack([np.asarray(src), alpha])
        piece = Image.fromarray(rgba, "RGBA")
        cx = (x0 + x1) / 2
        px, py = int(cx - w / 2), int(y)
        self.im.paste(piece, (px, py), piece)
        self.d = ImageDraw.Draw(self.im)
        ink = piece.split()[-1].getbbox()
        if ink is None:
            return None
        self.add((px + ink[0], py + ink[1], px + ink[2], py + ink[3]),
                 "calligraphy", col)
        return y + h + lead // 2

    def watermark(self, font_path, text, rng, tiled=False):
        """Digitisation watermark laid over the whole page.

        Deliberately gets NO region, unlike a stamp. A stamp is a bounded object
        on the paper and a detector should localise it; a watermark spans the
        entire page, so boxing it would produce a region covering everything and
        teach a detector nothing.
        """
        W, H = self.im.size
        layer = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        # A tiled watermark covers every square inch, so it must be far fainter
        # than a single diagonal one or the page greys out. A tiled mark at the
        # diagonal's alpha with a heavy scan tier on top leaves the text
        # unreadable, though either alone is fine.
        alpha = rng.randint(8, 18) if tiled else rng.randint(14, 30)
        grey = rng.randint(120, 180)
        try:
            f = font_bundle(font_path, int(min(W, H) * (0.055 if tiled else 0.13)))["normal"]
        except Exception:
            return
        if tiled:
            step = int(min(W, H) * 0.30)
            for yy in range(0, H * 2, step):
                for xx in range(0, W * 2, int(step * 1.6)):
                    ld.text((xx, yy), text, font=f, fill=(grey,) * 3 + (alpha,),
                            direction="rtl", anchor="ra")
        else:
            ld.text((W, H), text, font=f, fill=(grey,) * 3 + (alpha + 6,),
                    direction="rtl", anchor="mm")
        layer = layer.rotate(rng.uniform(20, 45), resample=Image.BICUBIC)
        crop = layer.crop((W // 2, H // 2, W // 2 + W, H // 2 + H))
        self.im = Image.alpha_composite(self.im.convert("RGBA"), crop).convert("RGB")
        self.d = ImageDraw.Draw(self.im)

    def spot(self, w, h):
        """Unbordered, non-rectangular spot illustration on transparent ground.

        Painted rather than sourced because the point is the SHAPE: an arabesque
        rosette, a girih star, a floral vignette or an organic blob has no frame
        and no white box, so its region is a rectangle containing mostly paper.
        That is the case a detector trained on framed photographs gets wrong --
        it either misses the figure entirely or boxes the whitespace around it.
        """
        rng = self.rng
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im, "RGBA")
        cx, cy = w / 2, h / 2
        R = min(w, h) * 0.46
        # Illuminated rather than plain ink: manuscript ornament is polychrome,
        # and a mono-ink-only corpus would teach the detector that coloured
        # vector art is something else.
        pal = rng.sample(ILLUM, 3)
        ink, sec, ter = pal
        mono = rng.random() < 0.3          # some ornament really is plain ink
        if mono:
            sec = ter = ink
        kind = rng.choice(["rosette", "girih", "floral", "blob", "crescent"])
        if kind == "rosette":
            n = rng.choice([6, 8, 12, 16])
            for ci, ring in enumerate((1.0, 0.62, 0.3)):
                pts = []
                for k in range(n * 12):
                    a = 2 * math.pi * k / (n * 12)
                    rr = R * ring * (0.72 + 0.28 * abs(math.cos(n * a / 2)))
                    pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
                col = (ink, sec, ter)[ci]
                if ci and not mono:
                    d.polygon(pts, fill=col + (55,))
                d.line(pts + [pts[0]], fill=col + (255,), width=max(1, int(R * 0.035)))
        elif kind == "girih":
            n = rng.choice([8, 10, 12])
            outer = [(cx + R * math.cos(2 * math.pi * k / n),
                      cy + R * math.sin(2 * math.pi * k / n)) for k in range(n)]
            step = rng.choice([2, 3])
            if not mono:
                d.polygon(outer, fill=ter + (45,))
            for k in range(n):
                d.line([outer[k], outer[(k + step) % n]], fill=ink + (255,),
                       width=max(1, int(R * 0.04)))
            for k in range(n):
                d.line([outer[k], outer[(k + 1) % n]], fill=sec + (200,),
                       width=max(1, int(R * 0.02)))
        elif kind == "floral":
            for k in range(rng.randint(5, 9)):
                a = 2 * math.pi * k / rng.randint(5, 9)
                px_, py_ = cx + R * 0.5 * math.cos(a), cy + R * 0.5 * math.sin(a)
                rr = R * rng.uniform(0.28, 0.46)
                d.ellipse([px_ - rr, py_ - rr * 0.6, px_ + rr, py_ + rr * 0.6],
                          fill=sec + ((0 if mono else 70),), outline=ink + (255,),
                          width=max(1, int(R * 0.03)))
            d.ellipse([cx - R * .16, cy - R * .16, cx + R * .16, cy + R * .16],
                      fill=ter + (230,))
        elif kind == "crescent":
            d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=sec + (235,))
            o = R * rng.uniform(0.3, 0.5)
            d.ellipse([cx - R + o, cy - R * .95, cx + R + o, cy + R * .95],
                      fill=(0, 0, 0, 0))
        else:                                        # organic blob with hatching
            pts = []
            for k in range(14):
                a = 2 * math.pi * k / 14
                rr = R * rng.uniform(0.55, 1.0)
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            d.polygon(pts, fill=sec + (55,), outline=ink + (255,))
            for k in range(len(pts)):
                d.line([pts[k], pts[(k + 1) % len(pts)]], fill=ink + (255,),
                       width=max(1, int(R * 0.03)))
            for t in range(int(R // 6)):             # hatch fill
                yy = cy - R + t * 6
                d.line([(cx - R * .8, yy), (cx + R * .8, yy)], fill=ter + (80,), width=1)
        return im

    def chart(self, fonts, xr, colw, y, fs, lead, max_y, col=None, kind=None):
        """Bar / line / pie / scatter / area / stacked plot with Arabic labels.

        Typed `chart`, not `figure`. The distinction earns its place: a chart is
        synthetic vector art with axes, gridlines and readable labels, so a
        downstream OCR pass may want to transcribe those labels, whereas a
        photograph should be skipped entirely. A detector that calls both
        `figure` throws that decision away.
        """
        rng = self.rng
        kind = kind or rng.choice(["bar", "bar", "line", "pie", "scatter",
                                   "area", "stacked"])
        h = int(min(colw * rng.uniform(0.62, 0.92), max_y - y - lead * 3))
        if h < 90 or colw < 120:
            return None
        x0, x1 = xr - colw, xr
        y0 = y
        d = self.d
        title = None
        if rng.random() < 0.7:
            title = rng.choice(CHART_TITLES)
            if self.cmap is None or drawable(title, self.cmap):
                tw = d.textlength(title, font=fonts["foot"], direction="rtl")
                d.text(((x0 + x1) / 2 + tw / 2, y), title, font=fonts["foot"],
                       fill=(30, 30, 30), direction="rtl", anchor="ra")
                y += int(lead * 0.95)
                h -= int(lead * 0.95)
        if h < 70:
            return None
        # plot frame
        pad_l, pad_b = int(colw * 0.14), int(h * 0.20)
        px0, py0, px1, py1 = x0 + pad_l, y, x1 - 6, y + h - pad_b
        n = rng.randint(4, 7)
        vals = [rng.uniform(0.15, 1.0) for _ in range(n)]
        labels = rng.sample(CHART_LABELS, n)
        cols = [CHART_COLORS[(i + rng.randrange(8)) % len(CHART_COLORS)] for i in range(n)]

        if kind != "pie":
            for g in range(5):                       # gridlines + y ticks
                gy = py1 - (py1 - py0) * g / 4
                d.line([(px0, gy), (px1, gy)], fill=(200, 200, 200), width=1)
                lab = arabize(int(25 * g), rng, self.cmap)
                d.text((px0 - 5, gy - fs * 0.35), lab, font=fonts["sup"],
                       fill=(90, 90, 90), direction="rtl", anchor="ra")
            d.line([(px0, py0), (px0, py1), (px1, py1)], fill=(60, 60, 60), width=2)

        if kind in ("bar", "stacked"):
            bw = (px1 - px0) / (n * 1.5)
            for i, v in enumerate(vals):
                bx = px1 - (i + 0.8) * bw * 1.5
                if kind == "bar":
                    d.rectangle([bx, py1 - (py1 - py0) * v, bx + bw, py1],
                                fill=cols[i], outline=(50, 50, 50))
                else:
                    acc = 0.0
                    for seg in range(3):
                        sv = v / 3 * rng.uniform(0.6, 1.4)
                        d.rectangle([bx, py1 - (py1 - py0) * (acc + sv),
                                     bx + bw, py1 - (py1 - py0) * acc],
                                    fill=CHART_COLORS[seg], outline=(50, 50, 50))
                        acc += sv
        elif kind in ("line", "area"):
            pts = [(px1 - (px1 - px0) * i / (n - 1), py1 - (py1 - py0) * v)
                   for i, v in enumerate(vals)]
            if kind == "area":
                d.polygon(pts + [(px0, py1), (px1, py1)], fill=cols[0] + (110,))
            d.line(pts, fill=cols[0], width=max(2, int(h * 0.012)))
            for pt in pts:
                r_ = max(3, int(h * 0.018))
                d.ellipse([pt[0] - r_, pt[1] - r_, pt[0] + r_, pt[1] + r_],
                          fill=cols[0], outline=(255, 255, 255))
        elif kind == "scatter":
            for i in range(rng.randint(18, 40)):
                sx = rng.uniform(px0 + 6, px1 - 6); sy = rng.uniform(py0 + 6, py1 - 6)
                r_ = rng.uniform(2, max(3, h * 0.016))
                d.ellipse([sx - r_, sy - r_, sx + r_, sy + r_],
                          fill=cols[i % n], outline=(60, 60, 60))
        else:                                        # pie
            R = min(px1 - px0, py1 - py0) * 0.44
            ccx, ccy = (px0 + px1) / 2, (py0 + py1) / 2
            tot = sum(vals); a0 = rng.uniform(0, 360)
            for i, v in enumerate(vals):
                a1 = a0 + 360 * v / tot
                d.pieslice([ccx - R, ccy - R, ccx + R, ccy + R], a0, a1,
                           fill=cols[i], outline=(255, 255, 255), width=2)
                a0 = a1
        # category labels along the bottom / legend for pie
        ly = py1 + 4
        for i, lb in enumerate(labels[:n]):
            if self.cmap is not None and not drawable(lb, self.cmap):
                continue
            if kind == "pie":
                # Inset by one swatch so the first entry's text cannot run past
                # px1 and out of the region box.
                cw_ = (px1 - px0) / 3
                lx = px1 - 14 - (i % 3) * cw_
                ry = ly + (i // 3) * int(fs * 0.9)
                tw_ = d.textlength(lb, font=fonts["sup"], direction="rtl")
                if lx - 11 - tw_ < px0:
                    continue                 # would overflow the plot: drop it
                d.rectangle([lx + 2, ry + 2, lx + 8, ry + 8], fill=cols[i])
                d.text((lx - 2, ry), lb, font=fonts["sup"], fill=(50, 50, 50),
                       direction="rtl", anchor="ra")
            else:
                lx = px1 - (i + 0.55) * (px1 - px0) / n
                d.text((lx + 12, ly), lb, font=fonts["sup"], fill=(50, 50, 50),
                       direction="rtl", anchor="ra")
        self.add((x0, y0 - 4, x1, y + h + 4), "chart", col)
        return y + h + lead // 2

    def figure(self, fonts, xr, colw, y, fs, lead, max_y, caption=None, col=None,
               kind="photo"):
        h = int(min(colw * self.rng.uniform(0.32, 0.62), max_y - y - lead * 2))
        if h < 40:
            return None
        # Real figures are inset and often centred in the column rather than
        # spanning it, so the width is sampled and the block is placed inside
        # it. The caption takes the figure's width, not the column's.
        wfrac = self.rng.choices([self.rng.uniform(0.30, 0.55),
                                  self.rng.uniform(0.55, 0.80),
                                  1.0], weights=[0.45, 0.40, 0.15], k=1)[0]
        w = max(60, int(colw * wfrac))
        if w >= colw:
            x0, x1 = xr - colw, xr
        else:
            # centred in the column, or flush to either edge
            off = self.rng.choices([(colw - w) / 2, 0.0, colw - w],
                                   weights=[0.6, 0.25, 0.15], k=1)[0]
            x1 = xr - off
            x0 = x1 - w
        h = int(min(h, max_y - y - lead * 2))
        if h < 40:
            return None
        art = None
        if kind == "spot":
            piece = self.spot(w, h)
            self.im.paste(piece, (int(x0), int(y)), piece)
            self.d = ImageDraw.Draw(self.im)
            self.add((x0, y, x1, y + h), "figure", col)
            y2 = y + h + lead // 3
            if caption:
                # wrapped to the FIGURE width, not the column: a caption sits
                # under its figure and inherits its width
                ln, sp = wrap_styled(self.d, [(a, "normal") for a in caption.split()],
                                     fonts, "foot", w)
                if ln and y2 + lead * len(ln) < max_y:
                    y0c = y2
                    for l in ln[:2]:
                        draw_line_c(self.d, l, fonts, "foot", x1, y2, fs, sp, (40, 40, 40))
                        y2 += int(lead * 0.8)
                    self.add((x0, y0c - 3, x1, y2), "caption", col)
            return y2 + lead // 3
        if kind == "cutout" and assets.CUTOUTS:
            # A transparent PNG dropped straight onto the page: NO background
            # fill and NO outline, so the only cue is the art itself. The box
            # is the ink extent of the alpha channel, not the paste rectangle
            # -- a cut-out rarely fills its own bounding square, and boxing the
            # square would label bare paper as figure.
            try:
                src = Image.open(self.rng.choice(assets.CUTOUTS)).convert("RGBA")
                sc = min(w / src.width, h / src.height)
                src = src.resize((max(1, int(src.width * sc)),
                                  max(1, int(src.height * sc))), Image.LANCZOS)
                if self.rng.random() < 0.25:
                    src = src.rotate(self.rng.uniform(-8, 8), expand=True,
                                     resample=Image.BICUBIC)
                px = int(x0 + (w - src.width) / 2)
                py = int(y + (h - src.height) / 2)
                self.im.paste(src, (px, py), src)
                self.d = ImageDraw.Draw(self.im)
                ink = src.split()[-1].getbbox()
                if ink is None:
                    return None
                bx = (px + ink[0], py + ink[1], px + ink[2], py + ink[3])
                if bx[2] - bx[0] < 20 or bx[3] - bx[1] < 20:
                    return None
                self.add(bx, "figure", col)
                y2 = bx[3] + lead // 3
                if caption:
                    cw = bx[2] - bx[0]
                    ln, sp = wrap_styled(self.d, [(a, "normal") for a in caption.split()],
                                         fonts, "foot", cw)
                    if ln and y2 + lead * len(ln) < max_y:
                        y0c = y2
                        for l in ln[:2]:
                            draw_line_c(self.d, l, fonts, "foot", bx[2], y2, fs,
                                        sp, (40, 40, 40))
                            y2 += int(lead * 0.8)
                        self.add((bx[0], y0c - 3, bx[2], y2), "caption", col)
                return y2 + lead // 3
            except Exception:
                pass
        if kind == "map":
            art = self.make_map(w, h)
        elif self.figure_paths:
            try:
                src = Image.open(self.rng.choice(self.figure_paths)).convert("RGB")
                # cover-crop to the block, so the figure fills it without letterbox
                sc = max(w / src.width, h / src.height)
                src = src.resize((max(1, int(src.width * sc)), max(1, int(src.height * sc))),
                                 Image.LANCZOS)
                ox, oy = (src.width - w) // 2, (src.height - h) // 2
                art = src.crop((ox, oy, ox + w, oy + h))
                if self.rng.random() < 0.5:        # halftone-ish print reproduction
                    art = art.convert("L").convert("RGB")
            except Exception:
                art = None
        if art is None:
            g = self.rng.randint(120, 205)
            art = Image.new("RGB", (w, h), (g, g, g + 6))
        self.im.paste(art, (int(x0), int(y)))
        # Not every figure is framed. Outlining all of them would hand a
        # detector the cheapest possible cue, and not one real pages reliably
        # give.
        if self.rng.random() < 0.55:
            self.d.rectangle([x0, y, x1, y + h], outline=(70, 70, 70), width=1)
        self.add((x0, y, x1, y + h), "figure", col)
        y += h + lead // 3
        if caption:
            ln, sp = wrap_styled(self.d, [(a, "normal") for a in caption.split()],
                                 fonts, "foot", w)
            if ln and y + lead * len(ln) < max_y:
                y0 = y
                for l in ln[:2]:
                    draw_line_c(self.d, l, fonts, "foot", x1, y, fs, sp, (40, 40, 40))
                    y += int(lead * 0.8)
                self.add((x0, y0 - 3, x1, y), "caption", col)
        return y + lead // 3


# --------------------------------------------------------------------------
# page composition
# --------------------------------------------------------------------------

# Per-PAGE presence rate each block type aims for: the share of pages that
# carry at least one. A flat menu would give every type the same chance per
# slot, and over ~7 slots a page even a 1-in-12 entry lands on ~45% of pages --
# chart, table, toc and verse alike. So rates are set per page and converted to
# per-slot weights below.
BLOCK_TARGETS = {
    "para": 1.00,      # the remainder; never scarce
    "heading": 0.22,   # the `title` and `banner` paths add headings outside
                       # the menu, so the menu asks for fewer than the share
                       # of pages that should carry one
    "poem": 0.12,      # -> verse regions
    # Tables and captions are rare on real pages. Both stay over-sampled so
    # the classes are learnable, but not so far that the prior inverts and a
    # detector expects one on every page.
    "table": 0.09,
    "toc": 0.06,
    "figure": 0.15,    # drags `caption` with it
    "ornament": 0.20,
    "chart": 0.08,
    "list": 0.14,      # visual variety only; emits `paragraph` regions
    "calligraphy": 0.08,
}


def block_menu(rng, spec, slots=7):
    """Per-slot weights that hit the per-PAGE targets above.

    A type appears on a page if it wins at least one of `slots` draws, so the
    per-slot probability is 1-(1-p)**(1/slots), not p. Using p directly turns
    a nominal 8% into charts on ~45% of pages.
    """
    names, weights = [], []
    for k, target in BLOCK_TARGETS.items():
        if k == "para":
            continue
        q = 1.0 - (1.0 - min(target, 0.99)) ** (1.0 / slots)
        if k == "chart":
            q *= max(0.15, spec["chart_p"] / 0.08) if spec.get("chart_p") else 1.0
        if k == "figure":
            q *= max(0.15, spec["spot_p"] / 0.20) if spec.get("spot_p") else 1.0
        names.append(k); weights.append(q)
    names.append("para"); weights.append(max(0.05, 1.0 - sum(weights)))
    return names, weights


# ---------------------------------------------------------------------------
# Frequency profiles
# ---------------------------------------------------------------------------
# `realistic` mirrors what a shelf of printed classical Arabic books looks like.
# `train` is what you should generate FROM. They differ on purpose.
#
# A detector needs on the order of a few hundred instances of a class before it
# learns it. At realistic rates a 5,000-page corpus contains roughly 50 charts,
# 25 upside-down pages and 40 three-column pages -- not enough for any of them,
# and those are precisely the classes every off-the-shelf detector already gets
# wrong. So the rare-but-important shapes are over-sampled.
#
# The cost is a mis-calibrated prior: a model trained on `train` thinks
# three-column pages are commoner than they are. That matters far less for
# detection than it would for a generative model, and it is correctable at
# inference with a confidence threshold. Under-representation is not
# correctable at all.
PROFILES = {
    "realistic": dict(
        ncols=[0.88, 0.09, 0.03], two_articles=0.05, col_rule=0.22,
        running_head=0.60, running_foot=0.12, head_line=0.22, page_number=0.85, title=0.18, footnotes=0.55,
        flip=0.005, tashkeel=0.45, latin=0.12, degrade=0.90, color_scan=0.25,
        head_font_differs=0.30, hashiya=0.08, hashiya_rotated=0.5,
        borderless=0.20, band_tint=0.06, landscape=0.02,
        full_footnote=0.02, full_figure=0.02,
        ornaments=0.30, banner=0.10, watermark=0.04,
        stamps=[0.97, 0.03, 0.00], chart_max=0.03, spot_max=0.04, map_max=0.06,
        wide_margin=0.30, calligraphy=0.02, para_break=0.18,
        title_page=0.06, display=0.02, novel=0.03, full_toc=0.03,
        latin_digits=0.12, speckle=0.25,
        cutout=0.03, asset_dirt=0.06, tinted_page=0.03,
        artifacts=dict(gutter=.20, edge=.10, holes=.05, stain=.10, tear=.04),
    ),
    # Rare classes are over-sampled 3-8x, deliberately, and no further: what
    # makes a class learnable is the ABSOLUTE count of pages that carry it (on
    # the order of a thousand is ample when fine-tuning), while what keeps a
    # detector calibrated is the ratio. Sizing by count instead of ratio gives
    # moderate over-sampling rather than an inverted prior -- a stamp on a
    # fifth of the pages, when real pages almost never carry one, teaches a
    # detector that a page ought to have a stamp, and it finds one on blank
    # paper. `wide_margin` is the other half: hard negatives, pages
    # that look like they should carry a hashiya or a stamp and do not.
    "train": dict(
        # Multi-column pages are over-sampled about 2x against `realistic`,
        # because they exist and need to be learnable -- but no further. Too
        # many columns teaches a detector that a paragraph is column-shaped, and
        # it then over-segments full-width text, the commonest block on a page.
        ncols=[0.78, 0.16, 0.06], two_articles=0.10, col_rule=0.35,
        # A running foot is rare in real books. This keeps the class learnable
        # without teaching a detector to invent a footer on every page.
        running_head=0.42, running_foot=0.09, head_line=0.30, page_number=0.70, title=0.20, footnotes=0.42,
        # flip renders the page upside down (with the boxes rotated to match)
        # to cover scans fed in the wrong way round. Such scans are rare, so it
        # stays at the `realistic` rate rather than spending capacity on them;
        # set it to 0 if they never occur in the intake at all.
        flip=0.005, tashkeel=0.50, latin=0.20, degrade=0.82, color_scan=0.45,
        # hashiya and stamps are rare on real pages, and over-sampling them far
        # teaches a detector to see them where there are none. At these rates
        # the absolute count stays learnable in a run of several thousand
        # pages, while the prior stops saying that a page ought to have one.
        head_font_differs=0.50, hashiya=0.05, hashiya_rotated=0.6,
        borderless=0.33, band_tint=0.22, landscape=0.12,
        full_footnote=0.06, full_figure=0.06,
        ornaments=0.30, banner=0.18, watermark=0.10,
        stamps=[0.965, 0.03, 0.005], chart_max=0.08, spot_max=0.20, map_max=0.20,
        wide_margin=0.26, calligraphy=0.03, para_break=0.18,
        # full_toc: a real table of contents usually fills the page, while an
        # in-page TOC block is short, so whole TOC pages are what give the class
        # its real height. Kept moderate, since real TOC pages are not common
        # either.
        # display: the large-print excerpt is a real page type but makes the
        # emptiest pages in the set, so it stays rare.
        title_page=0.10, display=0.02, novel=0.06, full_toc=0.06,
        # The page types that carry `text`: imprints and URLs (back_cover), a
        # story line under a picture (picture_book), an ownership inscription
        # (inscription). Running text alone never produces them.
        back_cover=0.05, picture_book=0.04, inscription=0.03,
        latin_digits=0.18, speckle=0.35,
        # real photographic/decorative furniture, see assets.py
        cutout=0.10, asset_dirt=0.14, tinted_page=0.10,
        artifacts=dict(gutter=.18, edge=.12, holes=.07, stain=.10, tear=.05),
    ),
}


def sample_spec(rng, force=None, profile="train"):
    """One page's layout recipe, drawn from a named frequency profile."""
    P = PROFILES[profile]
    s = {
        "ncols": rng.choices([1, 2, 3], weights=P["ncols"], k=1)[0],
        "col_rule": rng.random() < P["col_rule"],
        "articles": 2 if rng.random() < P["two_articles"] else 1,
        "running_head": rng.random() < P["running_head"],
        # Kept rarer than the running head: real books use a running foot far
        # less often.
        "running_foot": rng.random() < P.get("running_foot", 0.12),
        "head_line": rng.random() < P.get("head_line", 0.0),
        "page_number": rng.random() < P["page_number"],
        "title": rng.random() < P["title"],
        "footnotes": rng.random() < P["footnotes"],
        "flip": rng.random() < P["flip"],
        "tashkeel": rng.random() < P["tashkeel"],
        "latin": rng.random() < P["latin"],
        "poem_style": rng.choice(["inline", "inline", "offset"]),
        "degrade": rng.random() < P["degrade"],
        "color_scan": rng.random() < P["color_scan"],
        "title_color": rng.choice(TITLE_COLORS),
        "head_font_differs": rng.random() < P["head_font_differs"],
        "hashiya": (None if rng.random() >= P["hashiya"] else
                    ("rotated" if rng.random() < P["hashiya_rotated"] else "plain")),
        "hashiya_side": rng.choice(["right", "left"]),
        # HARD NEGATIVE. A page with margins as wide as a glossed page but no
        # gloss in them. Without these a detector never sees an empty wide
        # margin, so "wide margin" becomes the feature that predicts hashiya,
        # and it boxes a gloss on blank paper.
        "wide_margin": rng.random() < P["wide_margin"],
        "calligraphy": rng.random() < P["calligraphy"],
        "latin_digits": rng.random() < P["latin_digits"],
        "speckle": rng.random() < P["speckle"],
        # Probability that a gap between prose blocks is a BLOCK BREAK rather
        # than ordinary paragraph leading. A reader sees prose that runs on as
        # one block and splits it only at a visibly larger gap, so the ground
        # truth has to carry that cue instead of one region per paragraph
        # regardless.
        "para_break": P["para_break"],
        "table_style": "borderless" if rng.random() < P["borderless"] else "ruled",
        "band_tint": rng.random() < P["band_tint"],
        "tinted_page": rng.random() < P.get("tinted_page", 0.0),
        "asset_dirt": rng.random() < P.get("asset_dirt", 0.0),
        "orientation": "landscape" if rng.random() < P["landscape"] else "portrait",
        "map_p": rng.uniform(0.0, P["map_max"]),
        "spot_p": rng.uniform(0.0, P["spot_max"]),
        # transparent cut-out art, borderless by construction
        "cutout_p": rng.uniform(0.0, P.get("cutout", 0.0) * 4),
        "chart_p": rng.uniform(0.0, P["chart_max"]),
        "page_size": rng.choice(PAGE_SIZES),
        # Archetypes are whole PAGE TYPES, not block choices. A title page, a
        # cover, a large-print display page or a novel's dialogue looks
        # nothing like running text, and no mix of blocks assembles one by
        # chance. back_cover / picture_book / inscription are where `text`
        # lives -- see PUBLISHERS above.
        "archetype": rng.choices(
            ["normal", "full_footnote", "full_figure", "title_page",
             "display", "novel", "full_toc", "back_cover", "picture_book",
             "inscription"],
            weights=[1 - P["full_footnote"] - P["full_figure"] - P["title_page"]
                     - P["display"] - P["novel"] - P["full_toc"]
                     - P.get("back_cover", 0) - P.get("picture_book", 0)
                     - P.get("inscription", 0),
                     P["full_footnote"], P["full_figure"], P["title_page"],
                     P["display"], P["novel"], P["full_toc"],
                     P.get("back_cover", 0), P.get("picture_book", 0),
                     P.get("inscription", 0)], k=1)[0],
        "ornaments": rng.random() < P["ornaments"],
        "banner": rng.random() < P["banner"],
        "watermark": rng.random() < P["watermark"],
        "watermark_tiled": rng.random() < 0.4,
        "stamps": rng.choices([0, 1, 2], weights=P["stamps"], k=1)[0],
        "artifacts": [k for k, pr in P["artifacts"].items() if rng.random() < pr],
        "profile": profile,
    }
    if force:
        s.update(force)
    # A cover or a large-print display page has no running head, no footnote
    # apparatus and usually no folio. Drawing them anyway would make title
    # pages with footnotes, which real books do not have, and teach a detector
    # that a cover carries apparatus.
    if s["archetype"] in ("title_page", "display", "back_cover", "picture_book",
                          "inscription"):
        s["running_head"] = False
        s["footnotes"] = False
        s["page_number"] = s["page_number"] and s["archetype"] not in (
            "display", "back_cover", "inscription")
        s["hashiya"] = None
        s["title"] = s["title"] and s["archetype"] == "title_page"
        s["banner"] = s["banner"] and s["archetype"] == "inscription"
    elif s["archetype"] == "novel":
        s["footnotes"] = rng.random() < 0.15
        s["hashiya"] = None
    # A cover's ground is full-bleed colour, so it must be laid down BEFORE any
    # text: painted after the title it would wipe it and leave empty heading
    # boxes behind. Deciding it here lets build paint it first.
    s["cover"] = (s["archetype"] == "back_cover"
                  or (s["archetype"] == "title_page" and rng.random() < 0.45))
    if s["cover"]:
        s["tinted_page"] = False
    return s


def inject_latin(text, rng):
    ws = text.split()
    if len(ws) < 6:
        return text
    i = rng.randrange(2, len(ws) - 1)
    return " ".join(ws[:i] + [rng.choice(LATIN_SNIPPETS)] + ws[i:])




def merge_paragraphs(pg, lead, gap_frac=1.05, overlap_frac=0.75):
    """Join `paragraph` regions in the same column separated by normal leading.

    Only vertically adjacent regions in the same column, overlapping in x, with
    a gap under `gap_frac * lead`, are merged -- a block break is ~0.9-1.7*lead
    and survives. Anything that is not a paragraph (a heading, a verse block, a
    table) lying between them ends the run, so a merge never spans another
    region and the partition is preserved.

    Regions are Region dataclasses, not dicts: mutate `bbox` in place and drop
    the absorbed objects by identity.
    """
    regs = pg.regions
    paras = [r for r in regs if r.type == "paragraph"]
    if len(paras) < 2:
        return
    others = [r for r in regs if r.type != "paragraph"]

    by_col = {}
    for r in paras:
        by_col.setdefault(r.column, []).append(r)

    absorbed = set()
    for col, rs in by_col.items():
        rs.sort(key=lambda r: r.bbox[1])
        cur = None
        for r in rs:
            if cur is None:
                cur = r
                continue
            a, b = cur.bbox, r.bbox
            ox = min(a[2], b[2]) - max(a[0], b[0])
            width = min(a[2] - a[0], b[2] - b[0])
            gap = b[1] - a[3]
            between = any(o.column == col and a[3] - 2 < o.bbox[1] < b[1] + 2
                          for o in others)
            if (not between and width > 0 and ox > overlap_frac * width
                    and -lead * 0.5 <= gap < gap_frac * lead):
                cur.bbox = [min(a[0], b[0]), min(a[1], b[1]),
                            max(a[2], b[2]), max(a[3], b[3])]
                absorbed.add(id(r))
            else:
                cur = r

    if not absorbed:
        return
    out = [r for r in regs if id(r) not in absorbed]
    # renumber id and reading order after the merge, or order has holes
    out.sort(key=lambda r: r.order)
    for i, r in enumerate(out):
        r.id = i
        r.order = i
    regs[:] = out

def build(pool, rng, spec, size=None, figure_paths=None):
    W, H = size or spec.get("page_size") or (1024, 1536)
    if spec.get("orientation") == "landscape":
        W, H = H, W
        # a wide page is almost never set in one column
        spec["ncols"] = max(2, spec["ncols"])
    pg = Page(W, H, rng)
    # Coloured stock. Text and every box are drawn on top of this, so it must
    # be laid down before anything else. Real books use tinted paper and
    # colour-washed grounds, and a detector trained on white paper alone
    # treats the wash itself as an object.
    # A cover ground is full-bleed and goes down FIRST, before a single glyph;
    # `pg.ink` is then the colour every text path on this page must use, because
    # the rubrication colours are invisible on a dark board.
    pg.ink = (0, 0, 0)
    if spec.get("cover"):
        ground = rng.choice(COVER_GROUNDS)
        pg.im = Image.new("RGB", (W, H), ground)
        pg.d = ImageDraw.Draw(pg.im)
        pg.ink = (250, 250, 245) if sum(ground) < 330 else (20, 20, 20)
        spec["title_color"] = pg.ink
    if spec.get("tinted_page") and not spec.get("cover"):
        base = rng.choice(PAGE_TINTS)
        if rng.random() < 0.35:
            # soft vertical gradient rather than a flat fill
            g = Image.new("RGB", (1, H))
            gd = ImageDraw.Draw(g)
            other = tuple(max(0, min(255, c + rng.randint(-22, 22))) for c in base)
            for yy in range(H):
                t = yy / max(1, H - 1)
                gd.point((0, yy), tuple(int(base[i] * (1 - t) + other[i] * t)
                                        for i in range(3)))
            pg.im = g.resize((W, H))
        else:
            pg.im = Image.new("RGB", (W, H), base)
        pg.d = ImageDraw.Draw(pg.im)
    mx, my = int(W * rng.uniform(0.06, 0.10)), int(H * rng.uniform(0.05, 0.08))
    fs = rng.randint(20, 27)
    lead = int(fs * rng.uniform(1.55, 1.9))

    calli_pool = assets.CALLIGRAPHY
    pg.figure_paths = list(assets.FIGURES if figure_paths is None else figure_paths)
    faces = assets.FONTS
    if not faces:
        raise RuntimeError("No fonts configured: see assets.configure() or --fonts.")
    body_font = rng.choice(faces)
    head_font = rng.choice(faces) if spec["head_font_differs"] else body_font
    fonts = font_bundle(body_font, fs)
    cmap = font_cmap(body_font)
    pg.cmap = cmap
    pool = pool_for_font(pool, cmap, body_font)
    try:
        pg.map_font = ImageFont.truetype(body_font, max(11, int(fs * 0.55)),
                                         layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        pg.map_font = None
    hfonts = font_bundle(head_font, fs)

    # A title is a heading; a citation line is `text` however large it is set.
    # The scraped `heads` pool mixes the two, so it is split once per page.
    titles, cites = split_heads(pool["heads"])

    def T(s):
        s = s if spec["tashkeel"] else strip_tashkeel(s)
        if spec["latin"] and rng.random() < 0.3:
            s = inject_latin(s, rng)
        if not drawable(s, cmap):
            # last resort: remove only the offending characters. Reached rarely,
            # since pool_for_font already excluded whole passages.
            s = "".join(c for c in s if c.isspace() or ord(c) in cmap)
        return s

    # The gloss eats into the outer margin, so the text block must be narrowed
    # BEFORE anything is laid out -- a hashiya added afterwards would overprint.
    hash_w = int(W * rng.uniform(0.13, 0.19)) if spec["hashiya"] else 0
    hx0, hx1 = (W - mx - hash_w, W - mx) if spec["hashiya_side"] == "right" else (mx, mx + hash_w)
    # HARD NEGATIVE: reserve the same margin a gloss would occupy and leave it
    # EMPTY. Geometrically identical to a glossed page, no hashiya region. This
    # is the only thing in the generator that teaches absence.
    if not spec["hashiya"] and spec.get("wide_margin"):
        empty_w = int(W * rng.uniform(0.12, 0.20))
        if spec["hashiya_side"] == "right":
            mx_r = mx + empty_w
            mx_l = mx
        else:
            mx_l = mx + empty_w
            mx_r = mx
    elif spec["hashiya"]:
        if spec["hashiya_side"] == "right":
            mx_r = mx + hash_w + int(W * 0.02)
            mx_l = mx
        else:
            mx_r = mx
            mx_l = mx + hash_w + int(W * 0.02)
    else:
        mx_r = mx_l = mx
    tint = rng.choice(TABLE_TINTS)

    y = my
    # ---- running head -----------------------------------------------------
    if spec["running_head"] and spec.get("head_line"):
        # A running head that spans the measure and carries the folio on the
        # SAME line -- title flush right, number flush left, optionally joined
        # by dot leaders or a rule. Common in printed books, and the layout in
        # which a folio is most easily mistaken for part of the header, so the
        # two are boxed as separate regions below.
        # `textlength` refuses multiline text and a few pool entries contain a
        # newline; collapse whitespace before measuring.
        ht = " ".join(T(rng.choice(pool["items"] if rng.random() < 0.6
                                   else pool["heads"])).split())[:rng.randint(30, 90)]
        num = arabize(rng.randint(3, 640), rng, cmap)
        pg.d.text((W - mx_r, y), ht, font=fonts["foot"], fill=(0, 0, 0),
                  direction="rtl", anchor="ra")
        hb = pg.d.textbbox((W - mx_r, y), ht, font=fonts["foot"],
                           direction="rtl", anchor="ra")
        nw = pg.d.textlength(num, font=fonts["foot"], direction="rtl")
        pg.d.text((mx_l + nw, y), num, font=fonts["foot"], fill=(0, 0, 0),
                  direction="rtl", anchor="ra")
        nb = pg.d.textbbox((mx_l + nw, y), num, font=fonts["foot"],
                           direction="rtl", anchor="ra")
        join = rng.choices(["none", "dots", "rule"], weights=[0.5, 0.3, 0.2], k=1)[0]
        gap0, gap1 = nb[2] + 6, hb[0] - 6
        if join != "none" and gap1 - gap0 > 24:
            mid = (hb[1] + hb[3]) / 2
            if join == "rule":
                pg.d.line([(gap0, mid), (gap1, mid)], fill=(120, 120, 120), width=1)
            else:
                dw = pg.d.textlength(".", font=fonts["foot"], direction="rtl")
                xx = gap0
                while xx < gap1:
                    pg.d.text((xx + dw, mid - fs * 0.35), ".", font=fonts["foot"],
                              fill=(110, 110, 110), direction="rtl", anchor="ra")
                    xx += dw * 2
        # header and folio are SEPARATE regions even on a shared line
        pg.add((hb[0] - 4, hb[1] - 4, hb[2] + 4, hb[3] + 4), "header")
        pg.add((nb[0] - 5, nb[1] - 5, nb[2] + 5, nb[3] + 5), "page_number")
        spec["page_number"] = False        # already placed; do not repeat below
        y += int(lead * 1.1)
        y = pg.hr(mx_l, W - mx_r, y, 1)
        y += lead // 3
    elif spec["running_head"]:
        h = T(rng.choice(pool["heads"]))[:40]
        for xr, anchor in ((W - mx_r, "r"),
                           (mx_l + pg.d.textlength(h, font=fonts["foot"],
                                                   direction="rtl"), "l")):
            pg.d.text((xr, y), h, font=fonts["foot"], fill=(0, 0, 0),
                      direction="rtl", anchor="ra")
            hb = pg.d.textbbox((xr, y), h, font=fonts["foot"],
                               direction="rtl", anchor="ra")
            pg.add((hb[0] - 4, hb[1] - 4, hb[2] + 4, hb[3] + 4), "header")
            if rng.random() < 0.5:
                break
        y += int(lead * 1.1)
        y = pg.hr(mx_l, W - mx_r, y, 1)
        y += lead // 3

    # ---- title / subtitle -------------------------------------------------
    if spec["title"]:
        t = T(rng.choice(titles))[:48]
        tf = font_bundle(head_font, int(fs * 1.6))
        ln, sp = wrap_styled(pg.d, [(w, "normal") for w in t.split()], tf, "normal",
                             W - mx_l - mx_r)
        if ln:
            y0, wmax, bottom = y, 0.0, y
            for l in ln[:2]:
                w = sum(pg.d.textlength(a, font=tf["normal"], direction="rtl") for a, _ in l) \
                    + sp * (len(l) - 1)
                wmax = max(wmax, w)
                draw_line_c(pg.d, l, tf, "normal", W / 2 + w / 2, y, fs, sp, spec["title_color"])
                # Advance past the INK, not a nominal 2.1 em. A nominal step is
                # shorter than the drawn line on a 1.6x face, so the title box
                # would reach past it and the subtitle box start inside it.
                bottom = max(bottom, y + tf["normal"].size * 1.25)
                y = int(bottom)
            # centred at W/2, so the ink spans W/2 +- wmax/2 -- not mx_l..W-mx_r
            pg.add((W / 2 - wmax / 2 - 6, y0 - 6,
                    W / 2 + wmax / 2 + 6, y), "heading")
            y += lead // 2
        if rng.random() < 0.6:
            st = T(rng.choice(titles))[:40]
            sw = pg.d.textlength(st, font=hfonts["head"], direction="rtl")
            sh = hfonts["head"].size
            pg.d.text((W / 2 + sw / 2, y), st, font=hfonts["head"],
                      fill=(pg.ink if spec.get("cover") else
                            tuple(min(255, c + 60) for c in spec["title_color"])),
                      direction="rtl", anchor="ra")
            pg.add((W / 2 - sw / 2 - 4, y - 4, W / 2 + sw / 2 + 4, y + sh * 1.25), "heading")
            # Arabic descenders on a 1.35x face reach past the advance, so the
            # step follows the face size rather than fs.
            y += int(sh * 1.25) + lead // 2
        cwid = W - mx_l - mx_r
        # Clear the descenders before the divider. Arabic descenders reach well
        # below the line advance, and without this gap the rule strikes through
        # the title whenever there is no subtitle to push it down -- a fault
        # that shows on the page but not on a box overlay.
        y += int(fs * 0.7)
        y = pg.hr(mx_l + cwid * 0.25, W - mx_r - cwid * 0.25, y, 2) + lead // 2

    # ---- calligraphic banner ----------------------------------------------
    if spec["banner"]:
        cal = [f for f in faces if any(k in os.path.basename(f) for k in CALLIGRAPHIC)]
        ny = pg.banner(rng.choice(cal) if cal else body_font,
                       rng.choice(BANNERS), mx_l, W - mx_r, y, fs, H // 2,
                       spec["title_color"])
        if ny:
            y = ny
            if rng.random() < 0.6:
                y = pg.ornament(fonts, mx_l, W - mx_r, y, fs, lead, H // 2) or y

    # ---- bottom furniture bands are reserved from the bottom up -----------
    # The running foot is drawn at H - my - lead*1.35. Unless that band is
    # reserved, the text block and the footnote apparatus both run through it
    # and the footer box overlaps them. Reserved FIRST, because the footnote
    # cap below is computed against what is left.
    rfoot_h = int(lead * 2.1) if spec.get("running_foot") else 0
    foot_h = 0
    if spec["footnotes"]:
        # A short apparatus is real, but if it is the ONLY kind drawn a
        # detector learns "footnote = two lines" and calls anything longer a
        # paragraph. Heights are sampled across the real range, including the
        # half-page apparatus of a critical edition.
        foot_h = int(lead * rng.choices(
            [rng.uniform(1.2, 2.5), rng.uniform(3.0, 6.0),
             rng.uniform(6.0, 11.0), rng.uniform(11.0, 18.0)],
            weights=[0.10, 0.26, 0.36, 0.28], k=1)[0])
        # An 18*lead apparatus does not fit a short page: it would drive the
        # footnote separator to a negative y. Cap it at 42% of the text block --
        # a critical edition can be footnote-heavy, but the matn never vanishes
        # entirely.
        foot_h = max(0, min(foot_h, int((H - 2 * my - rfoot_h) * 0.42)))
    num_h = int(lead * 1.4) if spec["page_number"] else 0
    content_bottom = H - my - foot_h - num_h - rfoot_h

    # ---- article bands ----------------------------------------------------
    bands, top = [], y
    if spec["articles"] == 2:
        split = top + (content_bottom - top) * rng.uniform(0.42, 0.58)
        bands = [(top, split - lead), (split + lead // 2, content_bottom)]
    else:
        bands = [(top, content_bottom)]

    content_y0, content_y1 = bands[0][0], bands[-1][1]
    if spec["archetype"] == "full_figure":
        # One plate filling the text block, with a caption. No body text at all.
        by0, by1 = bands[0][0], bands[-1][1]
        pg.figure(fonts, W - mx_r, W - mx_l - mx_r, by0, fs, lead, by1,
                  T(rng.choice(pool["heads"]))[:70],
                  kind="map" if rng.random() < 0.5 else "photo")
        bands = []
    elif spec["archetype"] == "full_footnote":
        # A page that is entirely apparatus -- small type, numbered, edge to
        # edge. Real in critical editions where one matn line carries a page of
        # commentary, and it looks nothing like a normal page.
        by0, by1 = bands[0][0], bands[-1][1]
        yy, n = by0, 1
        while yy < by1 - lead:
            t = f"({arabize(n, rng, cmap)}) " + T(rng.choice(pool["foots"]))
            ln, sp = wrap_styled(pg.d, [(w, "normal") for w in t.split()],
                                 fonts, "foot", W - mx_l - mx_r)
            if not ln or yy + lead * 0.85 * len(ln) > by1:
                break
            for l in ln:
                draw_line_c(pg.d, l, fonts, "foot", W - mx_r, yy, fs, sp)
                yy += int(lead * 0.85)
            yy += lead // 4
            n += 1
        if yy > by0:
            pg.add((mx_l, by0 - 4, W - mx_r, yy), "footnote")
        bands = []

    elif spec["archetype"] == "title_page":
        # Cover / title page: no body text, a large display title, and
        # furniture (stamps, accession scribbles) that appears nowhere else.
        # Covers get a full-bleed ground.
        by0, by1 = bands[0][0], bands[-1][1]
        # The cover ground is already down (see spec["cover"] at the top of
        # build) -- painting it here would wipe the title just drawn.
        fill = pg.ink
        yy = by0 + int((by1 - by0) * rng.uniform(0.10, 0.22))
        big = font_bundle(head_font, int(fs * rng.uniform(2.2, 3.6)))
        # The last slot is a body line on a cover - an author line, a series
        # note, a citation. It is `text`, not a heading: it titles nothing.
        slots = [(titles, big, 2.1, "heading"), (titles, hfonts, 1.5, "heading"),
                 (cites or pool["paras"], fonts, 1.2, "text")]
        for src, face, gap, rtype in slots:
            if yy > by1 - lead * 2:
                break
            txt = T(rng.choice(src))[:60 if rtype == "heading" else 90]
            ln, sp = wrap_styled(pg.d, [(w, "normal") for w in txt.split()],
                                 face, "head" if face is not fonts else "normal",
                                 W - mx_l - mx_r)
            if not ln:
                continue
            y0c = yy
            key = "head" if face is not fonts else "normal"
            for l in ln[:2]:
                draw_line_c(pg.d, l, face, key, W - mx_r, yy, fs, sp, fill)
                yy += int(face[key].size * gap)
            pg.add(hbox(pg.d, ln[:2], face, key, sp, W - mx_r, y0c, yy), rtype)
            yy += int(lead * rng.uniform(1.4, 3.0))
        # accession scribbles and seals: the furniture that marks a title page
        for _ in range(rng.randint(0, 2)):
            pg.stamp(body_font, rng.uniform(0.15, 0.85) * W,
                     rng.uniform(0.25, 0.85) * H,
                     int(min(W, H) * rng.uniform(0.09, 0.17)), rng)
        bands = []
    elif spec["archetype"] == "back_cover":
        # Back cover: a blurb panel, then the imprint - publisher, city, phone,
        # website - and a logo. A real back cover typically carries three or
        # four `text` boxes like these.
        by0, by1 = bands[0][0], bands[-1][1]
        fill = pg.ink
        yy = by0 + int((by1 - by0) * rng.uniform(0.05, 0.18))
        if rng.random() < 0.75:
            blurb = (rng.choice(BLURB_OPENERS) + " " +
                     " ".join(T(rng.choice(pool["paras"])).split()[:rng.randint(25, 60)]))
            panel_w = int((W - mx_l - mx_r) * rng.uniform(0.62, 0.92))
            px1 = int(W / 2 + panel_w / 2)
            if rng.random() < 0.5:                      # printed on a panel
                pad = int(lead * 0.8)
                pg.d.rectangle([px1 - panel_w - pad, yy - pad, px1 + pad,
                                min(by1, yy + int((by1 - yy) * 0.55))],
                               fill=tuple(max(0, min(255, c + (30 if sum(pg.ink) < 330 else -18)))
                                          for c in pg.im.getpixel((5, 5))))
            ny = pg.para(blurb, fonts, px1, panel_w, yy, fs, lead,
                         min(by1, yy + int((by1 - yy) * 0.55)), fill=fill)
            yy = (ny or yy) + lead * 2
        # The imprint sits at the FOOT of a back cover, under open space - the
        # shape that makes it recognisable. Drawn straight after the blurb it
        # would cram the content into the top third.
        yy = max(yy, by0 + int((by1 - by0) * rng.uniform(0.58, 0.76)))
        for block in imprint_block(rng, cmap):
            if yy > by1 - lead:
                break
            box = None
            for line in block:
                t = T(line)
                lw = pg.d.textlength(t, font=fonts["normal"], direction="rtl")
                pg.d.text((W / 2 + lw / 2, yy), t, font=fonts["normal"],
                          fill=fill, direction="rtl", anchor="ra")
                bb = pg.d.textbbox((W / 2 + lw / 2, yy), t, font=fonts["normal"],
                                   direction="rtl", anchor="ra")
                box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]),
                                              max(box[2], bb[2]), max(box[3], bb[3]))
                yy += lead
            if box:
                pg.add((box[0] - 6, box[1] - 5, box[2] + 6, box[3] + 5), "text")
            yy += lead
        if rng.random() < 0.55:
            pg.stamp(head_font, W * rng.uniform(0.2, 0.8), by1 - lead * 3,
                     int(min(W, H) * rng.uniform(0.08, 0.14)), rng)
        bands = []
    elif spec["archetype"] == "picture_book":
        # Children's book: one illustration filling the page and a single line
        # of story under it. That line is one of the commonest forms of `text`.
        by0, by1 = bands[0][0], bands[-1][1]
        # The illustration FILLS the page - that is what makes the line under it
        # read as `text` rather than a caption. pg.figure insets its art to a
        # third of the column, which is right in running text and wrong here.
        art_bottom = by0 + int((by1 - by0) * rng.uniform(0.62, 0.82))
        bleed = rng.random() < 0.5                     # to the paper edge, or inset
        ax0, ax1 = (0, W) if bleed else (mx_l, W - mx_r)
        ay0 = 0 if bleed else by0
        art = None
        if pg.figure_paths:
            try:
                art = Image.open(rng.choice(pg.figure_paths)).convert("RGB")
            except Exception:
                art = None
        if art is not None:
            sc = max((ax1 - ax0) / art.width, (art_bottom - ay0) / art.height)
            art = art.resize((max(1, int(art.width * sc)), max(1, int(art.height * sc))),
                             Image.LANCZOS).crop((0, 0, ax1 - ax0, art_bottom - ay0))
            pg.im.paste(art, (int(ax0), int(ay0)))
        else:
            piece = pg.spot(ax1 - ax0, art_bottom - ay0)
            pg.im.paste(piece, (int(ax0), int(ay0)), piece)
        pg.d = ImageDraw.Draw(pg.im)
        pg.add((ax0, ay0, ax1, art_bottom), "figure")
        yy = art_bottom + lead
        for _ in range(rng.randint(1, 2)):
            if yy > by1 - lead:
                break
            # A story line, not a reference line.
            cand = [p for p in rng.sample(pool["paras"], 8) if not is_citation(p)]
            line = " ".join(T(cand[0] if cand else rng.choice(pool["paras"])).split()
                            [:rng.randint(4, 12)])
            lw = pg.d.textlength(line, font=fonts["normal"], direction="rtl")
            if lw > W - mx_l - mx_r:
                break
            pg.d.text((W / 2 + lw / 2, yy), line, font=fonts["normal"],
                      fill=pg.ink, direction="rtl", anchor="ra")
            bb = pg.d.textbbox((W / 2 + lw / 2, yy), line, font=fonts["normal"],
                               direction="rtl", anchor="ra")
            pg.add((bb[0] - 6, bb[1] - 5, bb[2] + 6, bb[3] + 5), "text")
            yy = bb[3] + lead
        bands = []
    elif spec["archetype"] == "inscription":
        # A flyleaf: an ownership inscription or two in a calligraphic hand,
        # seals, and nothing else. `text` again, and the one case where the
        # class really is calligraphic.
        by0, by1 = bands[0][0], bands[-1][1]
        cal = [f for f in faces if any(k in os.path.basename(f) for k in CALLIGRAPHIC)]
        face = font_bundle(rng.choice(cal) if cal else head_font,
                           int(fs * rng.uniform(1.3, 2.0)))
        yy = by0 + int((by1 - by0) * rng.uniform(0.10, 0.35))
        # Sampled without replacement: no flyleaf repeats the same inscription
        # down the page.
        for line in rng.sample(INSCRIPTIONS, rng.randint(1, 3)):
            if yy > by1 - lead * 2:
                break
            line = T(line)
            lw = pg.d.textlength(line, font=face["normal"], direction="rtl")
            if lw > W - mx_l - mx_r:
                continue
            xr = rng.choice([W / 2 + lw / 2, W - mx_r, mx_l + lw])
            pg.d.text((xr, yy), line, font=face["normal"], fill=pg.ink,
                      direction="rtl", anchor="ra")
            bb = pg.d.textbbox((xr, yy), line, font=face["normal"],
                               direction="rtl", anchor="ra")
            pg.add((bb[0] - 8, bb[1] - 6, bb[2] + 8, bb[3] + 6), "text")
            yy = bb[3] + lead * rng.randint(1, 3)
        for _ in range(rng.randint(1, 2)):
            pg.stamp(head_font, rng.uniform(0.2, 0.8) * W, rng.uniform(0.3, 0.85) * H,
                     int(min(W, H) * rng.uniform(0.09, 0.16)), rng)
        bands = []
    elif spec["archetype"] == "display":
        # Very large centred text, a handful of lines, no furniture -- the
        # large-print excerpt / quote-card shape.
        by0, by1 = bands[0][0], bands[-1][1]
        dfs = int(fs * rng.uniform(2.4, 4.0))
        df = font_bundle(head_font if rng.random() < 0.5 else body_font, dfs)
        txt = T(rng.choice(pool["paras"]))
        ln, sp = wrap_styled(pg.d, [(w, "normal") for w in txt.split()],
                             df, "normal", W - mx_l - mx_r)
        yy = by0 + int((by1 - by0) * 0.10)
        y0c = yy
        drew = 0
        for l in ln:
            if yy + dfs * 1.7 > by1:
                break
            draw_line_c(pg.d, l, df, "normal", W - mx_r, yy, dfs, sp)
            yy += int(dfs * 1.7); drew += 1
        if drew:
            pg.add((mx_l, y0c - 8, W - mx_r, yy), "paragraph")
        bands = []
    elif spec["archetype"] == "novel":
        # Modern novel: one-line dialogue paragraphs, double leading, and the
        # page deliberately NOT filled to the foot.
        by0, by1 = bands[0][0], bands[-1][1]
        stop = by0 + (by1 - by0) * rng.uniform(0.55, 0.92)
        yy = by0
        lead_n = int(lead * rng.uniform(1.8, 2.6))
        while yy < stop - lead_n:
            t = T(rng.choice(pool["paras"]))
            if rng.random() < 0.55:
                t = '"' + " ".join(t.split()[:rng.randint(2, 9)]) + '"'
            ln, sp = wrap_styled(pg.d, [(w, "normal") for w in t.split()],
                                 fonts, "normal", W - mx_l - mx_r)
            if not ln or yy + lead * len(ln) > stop:
                break
            y0c = yy
            for l in ln:
                draw_line_c(pg.d, l, fonts, "normal", W - mx_r, yy, fs, sp)
                yy += lead
            pg.add((mx_l, y0c - lead // 5, W - mx_r, yy), "paragraph")
            yy += lead_n - lead
        bands = []
    elif spec["archetype"] == "full_toc":
        by0, by1 = bands[0][0], bands[-1][1]
        pg.toc([r for r in rng.sample(pool["rows"], min(len(pool["rows"]), 40))],
               fonts, W - mx_r, W - mx_l - mx_r, by0, fs, lead, by1)
        bands = []

    for bi, (by0, by1) in enumerate(bands):
        if spec["band_tint"]:
            # A wash behind the whole article, magazine style. Painted first so
            # the text sits ON it; no region of its own -- it is not content.
            bt = BAND_TINTS[(bi + rng.randrange(len(BAND_TINTS))) % len(BAND_TINTS)]
            # Start exactly at the band, never above it: any higher and the wash
            # reaches up into whatever was drawn before the band -- the title,
            # its rule, a banner, an ornament -- and paints over finished
            # content.
            pg.d.rectangle([mx_l - 6, by0, W - mx_r + 6, by1 + 4], fill=bt)
        if bi > 0:
            pg.hr(mx_l, W - mx_r, by0 - lead // 2, 2)
        ncols = spec["ncols"] if bi == 0 else rng.choice([spec["ncols"], 1, 2])
        gutter = int((W - mx_l - mx_r) * (0.05 if ncols > 1 else 0))
        colw = ((W - mx_l - mx_r) - gutter * (ncols - 1)) / ncols
        # RIGHT column first: this is the reading order, and the signal a
        # detector built for left-to-right pages gets backwards.
        rights = [W - mx_r - c * (colw + gutter) for c in range(ncols)]
        if ncols > 1 and spec["col_rule"]:
            for c in range(ncols - 1):
                pg.vrule(rights[c] - colw - gutter / 2, by0, by1)
        # The block menu is weighted per page. chart_p / spot_p have to bias
        # SELECTION, not just gate a block after it is picked: a page places
        # only about seven blocks, so a 1-in-12 menu entry appears on barely
        # half of pages however high its own probability is.
        menu_names, menu_w = block_menu(rng, spec)
        for c, xr in enumerate(rights):
            yy = by0
            guard = 0
            while yy < by1 - lead * 2 and guard < 40:
                guard += 1
                kind = rng.choices(menu_names, weights=menu_w, k=1)[0]
                if kind == "heading":
                    # Real headings vary in length, alignment, face and line
                    # count, so all four are sampled. `heads` entries are short,
                    # so longer headings are built by joining entries rather
                    # than by raising the truncation.
                    h = T(rng.choice(pool["heads"]))
                    style = rng.choices(["short", "long", "multi"],
                                        weights=[0.50, 0.30, 0.20], k=1)[0]
                    if style != "short":
                        h = h + " " + T(rng.choice(pool["items"]))
                        if style == "multi":
                            h = h + " " + T(rng.choice(pool["items"]))
                    h = h[:rng.randint(20, 130)]
                    # a heading may be set larger and/or in another face
                    hf = hfonts
                    if rng.random() < 0.45:
                        hf = font_bundle(rng.choice(faces),
                                         int(fs * rng.uniform(1.15, 1.75)))
                    key = "bold" if (rng.random() < 0.40 and hf.get("bold")) else "head"
                    if key not in hf:
                        key = "head"
                    hlead = int(hf[key].size * rng.uniform(1.35, 1.75))
                    maxln = {"short": 1, "long": 2, "multi": 4}[style]
                    ln, sp = wrap_styled(pg.d, [(w, "normal") for w in h.split()],
                                         hf, key, colw)
                    ln = ln[:maxln]
                    if ln and yy + hlead * (len(ln) + 0.3) < by1:
                        y0 = yy
                        fill = spec["title_color"] if rng.random() < .4 else (0, 0, 0)
                        align = rng.choices(["right", "center"],
                                            weights=[0.55, 0.45], k=1)[0]
                        widths = [line_w(pg.d, l, hf, key, sp) for l in ln]
                        for l, lw in zip(ln, widths):
                            x = xr if align == "right" else xr - (colw - lw) / 2
                            draw_line_c(pg.d, l, hf, key, x, yy, fs, sp, fill)
                            yy += hlead
                        wmax = max(widths) if widths else 0.0
                        if align == "right":
                            bx = (xr - wmax - 4, y0 - 4, xr + 4, yy)
                        else:
                            cx = xr - colw / 2
                            bx = (cx - wmax / 2 - 4, y0 - 4, cx + wmax / 2 + 4, yy)
                        pg.add(bx, "heading", c)
                        yy += lead // 4
                    continue
                # A single pool entry is often shorter than a real body
                # paragraph, which can run much longer, so up to three entries
                # are joined.
                fn = {"para": lambda: pg.para(
                          " ".join(T(x) for x in rng.sample(
                              pool["paras"],
                              rng.choices([1, 2, 3], weights=[0.45, 0.35, 0.20],
                                          k=1)[0])),
                          fonts, xr, colw, yy, fs, lead, by1, col=c),
                      # 1-20 baits: a single quoted bayt inside prose and a
                      # full-page qasida are both ordinary, so both ends of the
                      # range are drawn, not only the middle.
                      "poem": lambda: pg.poem([[T(a), T(b)] for a, b in
                                               rng.sample(pool["verses"],
                                                          min(len(pool["verses"]),
                                                              rng.choices(
                                                                  [rng.randint(1, 2),
                                                                   rng.randint(3, 7),
                                                                   rng.randint(8, 20)],
                                                                  # A poem is one block, so the
                                                                  # bait COUNT sets the block
                                                                  # height. Most real verse is a
                                                                  # 1-2 bayt quotation inside
                                                                  # prose, not a qasida.
                                                                  weights=[0.74, 0.19, 0.07],
                                                                  k=1)[0]))],
                                              fonts, xr, colw, yy, fs, lead, by1,
                                              spec["poem_style"], c),
                      "list": lambda: pg.listing([T(i) for i in
                                                  rng.sample(pool["items"], rng.randint(2, 5))],
                                                 fonts, xr, colw, yy, fs, lead, by1,
                                                 rng.random() < 0.5, c),
                      # 5-26 rows: real tables run from a short block to a page.
                      "table": lambda: pg.table(rng.sample(pool["rows"],
                                                           min(len(pool["rows"]),
                                                               rng.randint(5, 26))),
                                                fonts, xr, colw, yy, fs, lead, by1, c,
                                                spec["table_style"], tint),
                      # 4-22 rows: a handful of rows alone would make every
                      # in-page TOC a short stub, and real ones run long.
                      "toc": lambda: pg.toc(rng.sample(pool["rows"],
                                                       min(len(pool["rows"]),
                                                           rng.randint(4, 22))),
                                            fonts, xr, colw, yy, fs, lead, by1, c),
                      "chart": lambda: pg.chart(fonts, xr, colw, yy, fs, lead, by1, c),
                      "calligraphy": lambda: (pg.calligraphy(rng.choice(calli_pool),
                                                             xr - colw, xr, yy, fs,
                                                             lead, by1, c)
                                              if calli_pool else None),
                      "ornament": lambda: (pg.ornament(fonts, xr - colw, xr, yy,
                                                       fs, lead, by1, col=c)
                                           if spec["ornaments"] else None),
                      "figure": lambda: pg.figure(fonts, xr, colw, yy, fs, lead, by1,
                                                  T(rng.choice(pool["heads"]))[:60]
                                                  if rng.random() < 0.7 else None, c,
                                                  rng.choices(
                                                      ["cutout", "spot", "map", "photo"],
                                                      weights=[spec["cutout_p"],
                                                               spec["spot_p"],
                                                               spec["map_p"], 1.0],
                                                      k=1)[0]),
                      }[kind]
                ny = fn()
                if ny is None:
                    ny = pg.para(T(rng.choice(pool["paras"])), fonts, xr, colw,
                                 yy, fs, lead, by1, col=c)
                    if ny is None:
                        break
                # A BLOCK BREAK is a visibly larger gap. A reader groups prose
                # that runs on and splits it only where the gap widens, so the
                # gap has to be bimodal for that cue to exist at all: with one
                # constant gap everywhere, nothing on the page says where one
                # block ends and the next begins.
                if kind == "para" and rng.random() < spec["para_break"]:
                    ny += int(lead * rng.uniform(0.9, 1.7))
                yy = ny

    # Merge prose blocks that are NOT separated by a block break, so the ground
    # truth matches how the pages are actually read and labelled: one region per
    # run of continuous prose, split only at the wider gap drawn above.
    merge_paragraphs(pg, lead)

    # ---- hashiya ----------------------------------------------------------
    if spec["hashiya"]:
        gy0, gy1 = content_y0, content_y1
        pg.hashiya([T(t) for t in rng.sample(pool["foots"], 10)],
                   body_font, hx0, hx1, gy0, gy1, fs,
                   rotated=(spec["hashiya"] == "rotated"))

    # ---- footnotes --------------------------------------------------------
    if spec["footnotes"]:
        # rfoot_h too: the apparatus is what would otherwise collide with the footer
        fy = H - my - num_h - rfoot_h - foot_h
        # In an RTL page the separator starts at the RIGHT margin and runs
        # leftwards, like the text.
        sep_w = (W - mx_l - mx_r) * rng.uniform(0.28, 0.55)
        fy = pg.hr(W - mx_r - sep_w, W - mx_r, fy, 2)
        y0, n = fy, 1
        # The apparatus is one box, always. Split into chunks it would break
        # on an arbitrary note boundary with no visual gap, and chunks each
        # boxed from top-4 to bottom would overlap one another by 4px. A reader
        # sees the apparatus as one block, so it is boxed as one.
        spans = []
        while fy < H - my - num_h - rfoot_h - lead:
            t = T(rng.choice(pool["foots"]))
            # a few-word note is common in real editions and was never produced
            if rng.random() < 0.22:
                t = " ".join(t.split()[:rng.randint(2, 6)])
            t = f"({arabize(n, rng, cmap)}) " + t
            ln, sp = wrap_styled(pg.d, [(w, "normal") for w in t.split()],
                                 fonts, "foot", W - mx_l - mx_r)
            if not ln or fy + lead * 0.85 * len(ln) > H - my - num_h - rfoot_h:
                break
            ns, wmax = fy, 0.0
            for l in ln:
                wmax = max(wmax, line_w(pg.d, l, fonts, "foot", sp))
                draw_line_c(pg.d, l, fonts, "foot", W - mx_r, fy, fs, sp)
                fy += int(lead * 0.85)
            # Full measure even for a one-liner: the apparatus spans the text
            # block however short its notes are.
            spans.append((mx_l, ns, fy))
            n += 1
        if fy > y0:
            pg.add((mx_l, y0 - 4, W - mx_r, fy), "footnote")

    # ---- page number ------------------------------------------------------
    if spec["page_number"]:
        # Six positions, and the decorated forms real editions use. A folio
        # that only ever sits at the foot teaches a detector its POSITION
        # rather than its appearance, and it then takes a folio anywhere else
        # for a header or an ornament.
        num = arabize(rng.randint(3, 640), rng, cmap)
        deco = rng.choices(["plain", "dashes", "ellipse", "rect"],
                           weights=[0.62, 0.20, 0.09, 0.09], k=1)[0]
        s = f"- {num} -" if deco == "dashes" else num
        w = pg.d.textlength(s, font=fonts["normal"], direction="rtl")
        # At the top the folio shares the band with the running head. Putting
        # them on opposite sides is not enough -- a running head that spans the
        # full width still collides. Only go to the top when the band is free.
        top = rng.random() < 0.28 and not spec["running_head"]
        side = rng.randint(0, 2)
        px = {0: W / 2 + w / 2, 1: W - mx_r, 2: mx_l + w}[side]
        py = int(my * 0.62) if top else H - my
        pg.d.text((px, py), s, font=fonts["normal"], fill=(0, 0, 0),
                  direction="rtl", anchor="ra")
        if deco in ("ellipse", "rect"):
            eb = pg.d.textbbox((px, py), s, font=fonts["normal"],
                               direction="rtl", anchor="ra")
            pad = int(fs * 0.45)
            box = [eb[0] - pad, eb[1] - pad // 2, eb[2] + pad, eb[3] + pad // 2]
            if deco == "ellipse":
                pg.d.ellipse(box, outline=(0, 0, 0), width=max(1, fs // 22))
            else:
                pg.d.rectangle(box, outline=(0, 0, 0), width=max(1, fs // 22))
        # Ask the font where the ink actually is instead of guessing fs+6,
        # which clips Arabic-Indic digit descenders at large point sizes.
        bb = pg.d.textbbox((px, py), s, font=fonts["normal"],
                           direction="rtl", anchor="ra")
        if deco in ("ellipse", "rect"):
            # the drawn frame is part of the folio: box it, or the label sits
            # inside a shape a detector has to guess the extent of
            bb = (min(bb[0], box[0]), min(bb[1], box[1]),
                  max(bb[2], box[2]), max(bb[3], box[3]))
        pg.add((bb[0] - 5, bb[1] - 5, bb[2] + 5, bb[3] + 5), "page_number")

    # ---- running foot -----------------------------------------------------
    # Boxed to the ink like the running head. Real running feet are short, so
    # a full-width box would be mostly empty paper.
    if spec.get("running_foot") and not spec.get("cover"):
        # Drawn mostly from `items`, which read like book and chapter titles --
        # what a running foot carries. `heads` entries are too short to give a
        # running foot its real width.
        ft = " ".join(T(rng.choice(pool["items"] if rng.random() < 0.75
                                   else pool["heads"])).split())[:rng.randint(24, 80)]
        fy = H - my - int(lead * 1.35)
        fside = rng.choices(["center", "right", "left"],
                            weights=[0.45, 0.35, 0.20], k=1)[0]
        fw = pg.d.textlength(ft, font=fonts["foot"], direction="rtl")
        fx = {"center": W / 2 + fw / 2, "right": W - mx_r,
              "left": mx_l + fw}[fside]
        pg.d.text((fx, fy), ft, font=fonts["foot"], fill=(0, 0, 0),
                  direction="rtl", anchor="ra")
        fb = pg.d.textbbox((fx, fy), ft, font=fonts["foot"],
                           direction="rtl", anchor="ra")
        pg.add((fb[0] - 4, fb[1] - 4, fb[2] + 4, fb[3] + 4), "footer")
        if rng.random() < 0.45:
            pg.hr(mx_l, W - mx_r, fy - lead // 3, 1)

    # ---- stamps -----------------------------------------------------------
    for _ in range(spec["stamps"]):
        pg.stamp(body_font,
                 rng.uniform(0.18, 0.82) * W, rng.uniform(0.12, 0.88) * H,
                 int(min(W, H) * rng.uniform(0.085, 0.16)), rng)   # smaller reads as a dot, not a seal

    if spec["watermark"]:
        pg.watermark(body_font, rng.choice(WATERMARKS), rng, spec["watermark_tiled"])

    # ---- real grunge overlay ---------------------------------------------
    if spec.get("asset_dirt") and assets.DIRT:
        try:
            dt = Image.open(rng.choice(assets.DIRT)).convert("RGBA")
            sc = max(W / dt.width, H / dt.height) * rng.uniform(1.0, 1.6)
            dt = dt.resize((max(1, int(dt.width * sc)), max(1, int(dt.height * sc))),
                           Image.LANCZOS)
            if rng.random() < 0.5:
                dt = dt.transpose(Image.FLIP_LEFT_RIGHT)
            ox = rng.randint(0, max(0, dt.width - W))
            oy = rng.randint(0, max(0, dt.height - H))
            dt = dt.crop((ox, oy, ox + W, oy + H))
            a = dt.split()[-1].point(lambda v: int(v * rng.uniform(0.10, 0.40)))
            dt.putalpha(a)
            pg.im = pg.im.convert("RGB")
            pg.im.paste(dt, (0, 0), dt)
            pg.d = ImageDraw.Draw(pg.im)
        except Exception:
            pass

    if spec["artifacts"]:
        pg.im = scan_artifacts(pg.im, rng, spec["artifacts"])
    if spec.get("speckle"):
        pg.im = speckle(pg.im, rng)

    im = pg.im
    skew = 0.0
    if spec["degrade"]:
        try:
            # A watermark is already a global contrast loss; stacking a
            # medium/heavy scan tier on it leaves a page a human could not
            # read, whose boxes then mark text that is not legibly there --
            # the reason the heavy tier itself is bounded.
            cap = "light" if spec["watermark"] else None
            src = im if spec["color_scan"] else im.convert("L")
            for attempt, this_cap in enumerate((cap, "light", "clean")):
                im2, tier, skew = degrade_tracked(src, rng,
                                                  os.path.basename(body_font),
                                                  this_cap,
                                                  color=spec["color_scan"])
                con = text_contrast(np.asarray(im2.convert("L")), pg.regions)
                if con >= 110 or tier == "clean":
                    break
            spec["degrade_tier"] = tier
            spec["skew_deg"] = round(skew, 3)
            spec["text_contrast"] = round(con, 1)
            spec["degrade_retries"] = attempt
            im = im2.convert("RGB")
        except Exception:
            skew = 0.0

    # ---- pad boxes to the ink they contain --------------------------------
    # Boxes were computed from the layout cursor, which knows the line advance
    # but not how far a glyph actually reaches. Arabic overshoots it in both
    # directions: tashkeel stacks above the line, and jim/ain/ya descenders drop
    # well below it, so tight cursor boxes clipped letters. Pad generously, then
    # snap to the ink actually inside the padded window -- that both pads and
    # re-tightens, and it cannot run away because the pad bounds it.
    clean = np.asarray(pg.im.convert("L"))
    # Horizontal pad is 0.38 of the font size: an Arabic initial/final form
    # reaches past the cursor advance the box was derived from, and a smaller
    # pad clips letters at the left and right edges.
    PADDABLE = {"paragraph", "footnote", "hashiya", "header", "footer",
                "page_number", "heading", "verse", "toc", "caption",
                "calligraphy", "table"}
    for r in pg.regions:
        if r.type not in PADDABLE:
            continue
        x0, y0, x1, y1 = r.bbox
        # The window scales with the REGION, not with the body font size. A
        # fixed fs*0.38 could not reach the ink of a display block drawn at
        # 2.2-3.6x fs, so those boxes would keep their cursor geometry and sit
        # offset from the letters.
        px_pad = max(3, int(fs * 0.38), int((x1 - x0) * 0.06))
        py_pad = max(3, int(fs * 0.46), int((y1 - y0) * 0.06))
        wx0, wy0 = max(0, x0 - px_pad), max(0, y0 - py_pad)
        wx1, wy1 = min(W, x1 + px_pad), min(H, y1 + py_pad)
        win = clean[wy0:wy1, wx0:wx1]
        if win.size == 0:
            continue
        ys, xs = np.where(win < 200)
        if len(xs) == 0:
            continue
        # SNAP, not merely pad. Taking min()/max() against the original edges
        # would let a box only ever grow: a full-measure box around a six-word
        # footnote would stay full measure, and a block given more height than
        # its lines need would keep it. Following the ink in both directions is
        # what a human does when they draw a box.
        r.bbox = [wx0 + int(xs.min()), wy0 + int(ys.min()),
                  wx0 + int(xs.max()) + 1, wy0 + int(ys.max()) + 1]

    # ---- self-check: every text region must actually contain ink ----------
    # A drawer that advances its cursor for a block that failed to render
    # leaves a box over blank paper. Nothing in the JSON looks wrong; only the
    # PNG shows it. This makes the drawing prove itself.
    INKY = {"paragraph", "footnote", "hashiya", "header", "footer", "page_number",
            "heading", "verse", "table", "list", "text", "caption", "toc"}
    a = clean          # pre-degradation: scan noise would mask an empty region
    kept, dropped = [], []
    for r in pg.regions:
        # Contrast against the box's OWN background, not against white: on a
        # dark cover every pixel passes a plain `< 200` test, so boxes whose
        # text the cover fill had painted over would ship as perfectly good
        # headings.
        frac = ink_frac(a, r.bbox) if r.type in INKY else 1.0
        if r.type in INKY and frac < 0.004:
            dropped.append((r.type, round(frac, 5), tuple(r.bbox), (W, H)))
            continue
        kept.append(r)
    for i, r in enumerate(sorted(kept, key=lambda z: z.order)):
        r.id = i
        r.order = i
    regions = kept
    # Last gate: no two text boxes may claim the same ink. Trims what it can,
    # rejects the page otherwise (BadPage -> _one draws a different one).
    resolve_overlaps(regions)
    if skew:
        # Carry every box through the same rotation the page just took. Each
        # becomes a quad; bbox stays its axis-aligned hull so nothing downstream
        # breaks. This is what makes skewed ground truth honest instead of
        # approximately right.
        for r in regions:
            x0, y0, x1, y1 = r.bbox
            q = rotate_pts([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                           W / 2, H / 2, skew)
            r.quad = q
            xs = [p[0] for p in q]; ys = [p[1] for p in q]
            r.bbox = [int(round(min(xs))), int(round(min(ys))),
                      int(round(max(xs))), int(round(max(ys)))]
    if dropped:
        print(f"    dropped {len(dropped)} empty region(s): {dropped}")

    if spec["flip"]:
        # Upside-down scans are real and most detectors assume upright.
        # Boxes must be rotated with the page or the ground truth becomes a lie.
        im = im.rotate(180)
        regions = [Region(id=r.id, bbox=[W - r.bbox[2], H - r.bbox[3],
                                         W - r.bbox[0], H - r.bbox[1]],
                          type=r.type, score=1.0, native=r.native,
                          order=r.order, column=r.column,
                          quad=([[W - x, H - y] for x, y in r.quad]
                                if r.quad else None)) for r in regions]
    return im, regions, spec
