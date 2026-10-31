from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

WORDS = ("الكتاب", "العلم", "المدينة", "النهر", "الطريق", "الصباح", "القلم", "الدار",
         "البحر", "السماء", "الشجرة", "الحديقة", "الناس", "الأرض", "الليل", "الكلام")
LINKS = ("في", "من", "على", "إلى", "عن", "مع", "و", "ثم")


def arabic_text(seed: int = 0, paragraphs: int = 40) -> str:
    """Plain Arabic prose made up for the tests: titles, then paragraphs of sentences."""
    rng = random.Random(seed)
    sentence = lambda n: " ".join(rng.choice(WORDS if i % 2 == 0 else LINKS)  # noqa: E731
                                  for i in range(n)) + "."
    blocks = []
    for i in range(paragraphs):
        blocks.append(" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 6))))
        blocks.append(" ".join(sentence(rng.randint(5, 14)) for _ in range(rng.randint(3, 7))))
    return "\n\n".join(blocks)


def _fonts() -> list[str]:
    candidates = [os.environ.get("SPG_FONTS", ""), str(ROOT / "fonts"),
                  "/usr/share/fonts/truetype/noto/NotoNaskhArabic*.ttf",
                  "/usr/share/fonts/opentype/noto/NotoNaskhArabic*.otf"]
    for spec in candidates:
        if spec and (glob.glob(os.path.join(spec, "*.ttf")) or glob.glob(spec)):
            return [spec]
    return []


@pytest.fixture
def fonts() -> list[str]:
    found = _fonts()
    if not found:
        pytest.skip("no Arabic font: run tools/fetch_fonts.py fonts/, or set SPG_FONTS")
    return found
