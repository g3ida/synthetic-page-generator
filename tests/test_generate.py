"""End to end: pages rendered from a text pool and fonts, with labels that fit them.

Skipped without an Arabic font (tools/fetch_fonts.py, or SPG_FONTS).
"""

from __future__ import annotations

import json

from conftest import arabic_text
from make_text_pool import build
from PIL import Image, features

from synthetic_page_generator.cli import main
from synthetic_page_generator.regions import TYPES


def test_pillow_shapes_arabic() -> None:
    """Without Raqm every letter is drawn in its isolated form."""
    assert features.check("raqm")


def test_pages_come_with_labels_that_fit_them(tmp_path, fonts) -> None:
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(build([arabic_text()]), ensure_ascii=False))

    main(["--n", "4", "--seed", "3", "--workers", "1", "--text-pool", str(pool),
          "--fonts", *fonts, "--out", str(tmp_path / "out")])

    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["n_pages"] == 4 and manifest["n_errors"] == 0
    for page in manifest["pages"]:
        layout = json.loads((tmp_path / "out" / page["layout"]).read_text())
        with Image.open(tmp_path / "out" / page["image"]) as image:
            assert image.size == (layout["width"], layout["height"])
        assert layout["regions"]
        for region in layout["regions"]:
            x0, y0, x1, y1 = region["bbox"]
            assert region["type"] in TYPES
            assert 0 <= x0 < x1 <= layout["width"] and 0 <= y0 < y1 <= layout["height"]


def test_the_same_seed_draws_the_same_pages(tmp_path, fonts) -> None:
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(build([arabic_text()]), ensure_ascii=False))
    for run in ("a", "b"):
        main(["--n", "2", "--seed", "5", "--workers", "1", "--text-pool", str(pool),
              "--fonts", *fonts, "--out", str(tmp_path / run)])

    for name in ("s000000.png", "s000001.png"):
        assert (tmp_path / "a/images/0000" / name).read_bytes() == (tmp_path / "b/images/0000" / name).read_bytes()
