"""Building a text pool from plain text."""

from __future__ import annotations

import pytest
from conftest import arabic_text
from make_text_pool import MINIMUM, build


def test_plain_text_fills_every_kind_of_text_a_page_needs() -> None:
    pool = build([arabic_text()])

    assert set(pool) == {"paras", "heads", "foots", "items", "rows", "verses"}
    assert all(len(v) >= MINIMUM for v in pool.values())
    assert all(len(row) == 2 for row in pool["rows"])
    assert all(len(verse) == 2 for verse in pool["verses"])


def test_verse_is_taken_from_its_own_file_when_there_is_one() -> None:
    verses = [f"الشطر الأول {i}\tالشطر الثاني {i}" for i in range(10)]

    pool = build([arabic_text()], verses)

    assert pool["verses"][0] == ["الشطر الأول 0", "الشطر الثاني 0"]


def test_too_little_text_says_which_kind_ran_short() -> None:
    with pytest.raises(SystemExit, match="verses"):
        build([arabic_text(paragraphs=40)], verses=["only one half"])


def test_body_passages_vary_in_length_and_none_outruns_a_narrow_column() -> None:
    """Passages of 160 words did not fit a column, and two-column pages came
    out with no body text at all."""
    pool = build([arabic_text()])

    lengths = [len(p.split()) for p in pool["paras"]]
    assert min(lengths) >= 25 and max(lengths) <= 90
    assert len(set(lengths)) > 5
