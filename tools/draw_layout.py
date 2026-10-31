"""Draw each page's regions on it, to look at what was generated.

    python tools/draw_layout.py out/ drawn/            # every page
    python tools/draw_layout.py out/ drawn/ --n 12     # the first 12

Each region is outlined in its type's colour (its quad when the page is
skewed) and labelled with its type and reading order.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TYPES = ["paragraph", "text", "footnote", "hashiya", "header", "footer", "page_number",
         "heading", "verse", "table", "toc", "figure", "chart", "rule", "ornament", "stamp",
         "caption", "calligraphy", "non_text", "other"]


def colour(kind: str) -> tuple[int, int, int]:
    hue = (TYPES.index(kind) if kind in TYPES else len(TYPES)) * 0.618 % 1
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.85))


def draw(image: Image.Image, regions: list[dict]) -> Image.Image:
    page = image.convert("RGB")
    ink = ImageDraw.Draw(page)
    width = max(2, round(max(page.size) / 450))
    font = ImageFont.load_default(max(12, round(max(page.size) / 80)))
    for region in regions:
        c = colour(region["type"])
        if region.get("quad"):
            ink.polygon([tuple(p) for p in region["quad"]], outline=c, width=width)
        else:
            ink.rectangle(region["bbox"], outline=c, width=width)
        label = f"{region['order']} {region['type']}" if region.get("order") is not None else region["type"]
        x, y = region["bbox"][0], max(0, region["bbox"][1] - font.size - 4)
        box = ink.textbbox((x, y), label, font=font)
        ink.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=c)
        ink.text((x, y), label, fill="white", font=font)
    return page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="a folder synthetic-pages wrote")
    ap.add_argument("drawn", type=Path, help="where to write the drawn pages")
    ap.add_argument("--n", type=int, default=0, help="only the first N pages")
    a = ap.parse_args()
    a.drawn.mkdir(parents=True, exist_ok=True)
    pages = json.loads((a.out / "manifest.json").read_text(encoding="utf-8"))["pages"]
    for page in pages[: a.n or None]:
        layout = json.loads((a.out / page["layout"]).read_text(encoding="utf-8"))
        with Image.open(a.out / page["image"]) as image:
            draw(image, layout["regions"]).save(a.drawn / f"{page['page_id']}.png")
    print(len(pages[: a.n or None]), "pages drawn in", a.drawn)


if __name__ == "__main__":
    main()
