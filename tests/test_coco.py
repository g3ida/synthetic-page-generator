"""Exporting a generated folder to COCO, with and without a class map."""

from __future__ import annotations

import json

from synthetic_page_generator.coco import to_coco


def _folder(tmp_path):
    (tmp_path / "layout").mkdir()
    layout = {"page_id": "s000000", "width": 100, "height": 200, "regions": [
        {"bbox": [10, 20, 60, 80], "type": "paragraph"},
        {"bbox": [10, 90, 60, 100], "type": "footnote"},
        {"bbox": [5, 5, 15, 12], "type": "page_number"},
    ]}
    (tmp_path / "layout/s000000.json").write_text(json.dumps(layout))
    manifest = {"pages": [{"page_id": "s000000", "image": "images/s000000.png",
                           "layout": "layout/s000000.json"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_boxes_become_coco_x_y_width_height(tmp_path) -> None:
    coco = to_coco(_folder(tmp_path))

    paragraph = next(a for a in coco["annotations"] if a["bbox"][0] == 10 and a["bbox"][1] == 20)
    assert paragraph["bbox"] == [10, 20, 50, 60]
    assert coco["images"][0]["file_name"] == "images/s000000.png"


def test_a_class_map_merges_types_and_drops_the_ones_it_leaves_out(tmp_path) -> None:
    coco = to_coco(_folder(tmp_path), {"paragraph": "text", "footnote": "text", "page_number": None})

    assert coco["categories"] == [{"id": 1, "name": "text"}]
    assert len(coco["annotations"]) == 2
