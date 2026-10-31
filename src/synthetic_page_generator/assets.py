"""The fonts and images a page is made from.

Nothing is bundled: fonts, figures and the rest come from wherever you point
`configure` (or the command line). Only fonts are required. Every other kind is
optional, and a page that would have used one draws something else instead.

Each argument is a list of folders, glob patterns or files. A folder is read in
sorted order, not recursively; a font that appears twice by file name is kept
once, first come first served.
"""

from __future__ import annotations

import glob
import os

FONT_EXTS = (".ttf", ".otf")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

#: Typefaces to set text in. Faces whose file names contain one of the words in
#: `generator.CALLIGRAPHIC` are also used as display and calligraphic faces.
FONTS: list[str] = []
#: Pictures to place as figures.
FIGURES: list[str] = []
#: Transparent PNG art, pasted with no frame.
CUTOUTS: list[str] = []
#: Grime and dust overlays (transparent PNG), laid over a finished page.
DIRT: list[str] = []
#: Calligraphy pieces, set into the text as display text.
CALLIGRAPHY: list[str] = []


def _files(specs, exts) -> list[str]:
    out = []
    for spec in specs or ():
        spec = os.path.expanduser(str(spec))
        if os.path.isdir(spec):
            out += [os.path.join(spec, f) for f in sorted(os.listdir(spec))
                    if f.lower().endswith(exts)]
        elif any(c in spec for c in "*?["):
            out += [f for f in sorted(glob.glob(spec)) if f.lower().endswith(exts)]
        elif os.path.isfile(spec):
            out.append(spec)
    return out


def _unique_by_name(paths: list[str]) -> list[str]:
    seen, out = set(), []
    for path in paths:
        name = os.path.basename(path)
        if name not in seen:
            seen.add(name)
            out.append(path)
    return out


def configure(fonts=(), figures=(), cutouts=(), dirt=(), calligraphy=()) -> None:
    """Point the generator at its inputs. Call before generating, in every process."""
    global FONTS, FIGURES, CUTOUTS, DIRT, CALLIGRAPHY
    FONTS = _unique_by_name(_files(fonts, FONT_EXTS))
    FIGURES = _files(figures, IMAGE_EXTS)
    CUTOUTS = _files(cutouts, (".png",))
    DIRT = _files(dirt, (".png",))
    CALLIGRAPHY = _files(calligraphy, IMAGE_EXTS)


def summary() -> dict[str, int]:
    return {"fonts": len(FONTS), "figures": len(FIGURES), "cutouts": len(CUTOUTS),
            "dirt": len(DIRT), "calligraphy": len(CALLIGRAPHY)}
