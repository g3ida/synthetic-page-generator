"""Where the generator finds its fonts and images: folders, glob patterns and files."""

from __future__ import annotations

from synthetic_page_generator import assets


def test_a_folder_glob_and_file_are_all_read_in_sorted_order(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    for name in ("Zeta.ttf", "Alpha.ttf", "notes.txt"):
        (tmp_path / "a" / name).write_bytes(b"")
    (tmp_path / "Extra.otf").write_bytes(b"")

    assets.configure(fonts=[tmp_path / "a", str(tmp_path / "Ext*.otf")])

    assert [p.rsplit("/", 1)[1] for p in assets.FONTS] == ["Alpha.ttf", "Zeta.ttf", "Extra.otf"]


def test_a_font_found_twice_by_name_is_kept_once_first_come_first_served(tmp_path) -> None:
    for folder in ("first", "second"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "Amiri.ttf").write_bytes(b"")

    assets.configure(fonts=[tmp_path / "first", tmp_path / "second"])

    assert assets.FONTS == [str(tmp_path / "first" / "Amiri.ttf")]


def test_every_kind_but_fonts_may_be_left_empty(tmp_path) -> None:
    (tmp_path / "Amiri.ttf").write_bytes(b"")

    assets.configure(fonts=[tmp_path])

    assert assets.summary() == {"fonts": 1, "figures": 0, "cutouts": 0, "dirt": 0, "calligraphy": 0}
