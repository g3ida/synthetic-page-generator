"""`synthetic-pages-coco OUT`: a generated folder's layout as one COCO file.

The generator labels regions with its own 20 types (`regions.TYPES`). A
detector is usually trained on fewer, so a class map can merge and drop them:
a JSON object from generator type to class name, where `null` drops the type.
Types the map does not mention are dropped too. Without a map every type is its
own class.

    synthetic-pages-coco out/ --map classes.json -o out/coco.json

Category ids start at 1, in the order classes first appear in the map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .regions import TYPES


def to_coco(out: Path, class_map: dict[str, str | None] | None = None) -> dict:
    out = Path(out)
    class_map = class_map if class_map is not None else {t: t for t in TYPES}
    classes = list(dict.fromkeys(c for c in class_map.values() if c))
    ids = {name: i + 1 for i, name in enumerate(classes)}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    images, annotations = [], []
    for page_id, page in enumerate(manifest["pages"], 1):
        layout = json.loads((out / page["layout"]).read_text(encoding="utf-8"))
        images.append({"id": page_id, "file_name": page["image"],
                       "width": layout["width"], "height": layout["height"]})
        for region in layout["regions"]:
            name = class_map.get(region["type"])
            if not name:
                continue
            x0, y0, x1, y1 = region["bbox"]
            annotations.append({
                "id": len(annotations) + 1, "image_id": page_id, "category_id": ids[name],
                "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            })
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": name} for name, i in ids.items()]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="synthetic-pages-coco")
    ap.add_argument("out", type=Path, help="a folder synthetic-pages wrote")
    ap.add_argument("--map", type=Path, help="JSON: generator type -> class name, or null to drop")
    ap.add_argument("-o", "--output", type=Path, help="default: OUT/coco.json")
    a = ap.parse_args(argv)
    class_map = json.loads(a.map.read_text(encoding="utf-8")) if a.map else None
    if class_map:
        unknown = sorted(set(class_map) - set(TYPES))
        if unknown:
            ap.error(f"not generator types: {unknown}")
    coco = to_coco(a.out, class_map)
    target = a.output or a.out / "coco.json"
    target.write_text(json.dumps(coco), encoding="utf-8")
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} regions, "
          f"{len(coco['categories'])} classes -> {target}")


if __name__ == "__main__":
    main()
