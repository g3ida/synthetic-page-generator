"""Build a text pool — the JSON every word on a generated page is drawn from — out of plain text.

    python tools/make_text_pool.py book.txt more.txt --verses poems.txt -o pool.json

The generator never invents text: it samples real passages, so word lengths,
diacritic density and punctuation look like the source. Give it text in the
register you want the pages to be in. Paragraphs are blocks of text separated
by a blank line. See docs/TEXT_POOL.md for the format this writes.

    paras   body text            paragraphs, gathered by sentence into
                                 passages of 25-90 words
    heads   titles               lines of 2-9 words that do not end a sentence
    foots   footnote text        sentences of 8-45 words
    items   list items           sentences of 3-16 words
    rows    table and index rows [title, page number], from the titles
    verses  [first half, second half] of a verse, from --verses: one verse per
            line, its halves split by a tab or " *** ". Without it, pairs of
            short clauses stand in, which reads as verse only by its layout.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
SENTENCE_END = re.compile(r"(?<=[.!?؟؛])\s+")
#: The generator falls back to a category's whole list when a typeface can draw
#: fewer than this many of its passages, so every category needs at least this.
MINIMUM = 8


def _paragraphs(texts: list[str]) -> list[str]:
    out = []
    for text in texts:
        for block in re.split(r"\n\s*\n", text):
            block = " ".join(block.split())
            if block:
                out.append(block)
    return out


def _lines(texts: list[str]) -> list[str]:
    return [" ".join(line.split()) for text in texts for line in text.splitlines() if line.strip()]


#: Body passages are kept to what a printed paragraph runs to. A long one does
#: not fit a narrow column, and a page that cannot place its paragraphs is left
#: with no body text at all.
PARA_WORDS = (25, 90)


def _passages(paragraphs: list[str], rng: random.Random) -> list[str]:
    """Paragraphs as passages of PARA_WORDS words, each its own length.

    Whole sentences are gathered until a passage reaches the length drawn for
    it; a sentence that alone runs past the longest passage is cut by words,
    since some texts barely punctuate at all.
    """
    low, high = PARA_WORDS
    out = []
    for paragraph in paragraphs:
        current: list[str] = []
        target = rng.randint(low, high)
        for sentence in SENTENCE_END.split(paragraph):
            current += sentence.split()
            while len(current) >= target:
                out.append(" ".join(current[:target]))
                current = current[target:]
                target = rng.randint(low, high)
        if len(current) >= low:
            out.append(" ".join(current))
    return out


def _sentences(paragraphs: list[str]) -> list[str]:
    return [s.strip() for p in paragraphs for s in SENTENCE_END.split(p) if s.strip()]


def _verses(lines: list[str]) -> list[list[str]]:
    out = []
    for line in lines:
        halves = re.split(r"\t+|\s+\*\*\*\s+", line.strip())
        if len(halves) == 2 and all(h.strip() for h in halves):
            out.append([h.strip() for h in halves])
    return out


def build(texts: list[str], verses: list[str] | None = None, seed: int = 0) -> dict:
    rng = random.Random(seed)
    paragraphs = _paragraphs(texts)
    sentences = _sentences(paragraphs)
    words = lambda s: len(s.split())  # noqa: E731

    paras = _passages(paragraphs, rng)
    heads = [line for line in _lines(texts) if 2 <= words(line) <= 9 and not line.endswith((".", ":", "،"))]
    foots = [s for s in sentences if 8 <= words(s) <= 45]
    items = [s for s in sentences if 3 <= words(s) <= 16]
    rows = [[h, str(rng.randint(3, 480)).translate(ARABIC_DIGITS)] for h in heads]
    if verses:
        verse_pairs = _verses(verses)
    else:
        clauses = [c.strip() for s in sentences for c in re.split(r"[،,]", s) if 4 <= words(c) <= 7]
        verse_pairs = [[clauses[i], clauses[i + 1]] for i in range(0, len(clauses) - 1, 2)]
    pool = {"paras": paras, "heads": heads, "verses": verse_pairs, "items": items,
            "rows": rows, "foots": foots}
    short = {k: len(v) for k, v in pool.items() if len(v) < MINIMUM}
    if short:
        raise SystemExit(f"Too little text for {short}: each kind needs at least {MINIMUM}.")
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+", type=Path, help="UTF-8 text files")
    ap.add_argument("--verses", type=Path, help="one verse per line, halves split by a tab or ' *** '")
    ap.add_argument("-o", "--out", type=Path, default=Path("pool.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    texts = [p.read_text(encoding="utf-8") for p in a.texts]
    verses = a.verses.read_text(encoding="utf-8").splitlines() if a.verses else None
    pool = build(texts, verses, a.seed)
    a.out.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in pool.items()}, "->", a.out)


if __name__ == "__main__":
    main()
