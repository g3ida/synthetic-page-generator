"""`synthetic-pages`: render N pages with their layout, into OUT/.

    OUT/images/0000/s000000.png      the page
    OUT/layout/0000/s000000.json     its regions, reading order and the spec it was drawn from
    OUT/manifest.json                every page, the region totals, and any page that failed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from . import assets
from .generator import PROFILES, BadPage, build, sample_spec
from .regions import PageLayout


def _one(args):
    """Render one page. Top-level so multiprocessing can pickle it."""
    (i, seed, prefix, out, pool, force, profile, shard) = args
    rng = random.Random(seed)
    pid = f"{prefix}{i:06d}"
    # A page rejected for contradictory boxes is redrawn, not lost: the corpus
    # keeps its size and its class mix, and only the bad draw is thrown away.
    rejects = 0
    try:
        for _attempt in range(6):
            spec = sample_spec(rng, force, profile)
            try:
                im, regions, spec = build(pool, rng, spec)
                break
            except BadPage:
                rejects += 1
        else:
            return {"page_id": pid, "error": "6 rejected draws (overlapping boxes)"}
    except Exception as e:
        return {"page_id": pid, "error": f"{type(e).__name__}: {e}"}
    image = os.path.join("images", shard, f"{pid}.png")
    layout = os.path.join("layout", shard, f"{pid}.json")
    os.makedirs(os.path.join(out, "images", shard), exist_ok=True)
    os.makedirs(os.path.join(out, "layout", shard), exist_ok=True)
    im.save(os.path.join(out, image))
    lay = PageLayout(page_id=pid, image=image, width=im.width, height=im.height, regions=regions)
    d = lay.to_dict()
    d["spec"] = spec
    with open(os.path.join(out, layout), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    kinds = {}
    for r in regions:
        kinds[r.type] = kinds.get(r.type, 0) + 1
    return {"page_id": pid, "image": image, "layout": layout,
            "w": im.width, "h": im.height, "n_regions": len(regions),
            "kinds": kinds, "profile": profile,
            "orientation": spec["orientation"], "ncols": spec["ncols"],
            "archetype": spec["archetype"], "flip": spec["flip"], "rejects": rejects,
            "tier": spec.get("degrade_tier", "clean")}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="synthetic-pages",
        description="Generate Arabic document pages with layout ground truth.")
    ap.add_argument("--n", type=int, default=12, help="pages to render")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-pool", "--pool", dest="pool", required=True,
                    help="JSON text pool (see docs/TEXT_POOL.md)")
    ap.add_argument("--fonts", nargs="+", required=True,
                    help="font folders, glob patterns or files")
    ap.add_argument("--figures", nargs="*", default=[], help="pictures to place as figures")
    ap.add_argument("--cutouts", nargs="*", default=[], help="transparent PNG art")
    ap.add_argument("--dirt", nargs="*", default=[], help="transparent grime overlays")
    ap.add_argument("--calligraphy", nargs="*", default=[], help="calligraphy pieces")
    ap.add_argument("--profile", default="train", choices=sorted(PROFILES))
    ap.add_argument("--prefix", default="s", help="page id prefix")
    ap.add_argument("--force", default=None, help="JSON spec overrides")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--shard-size", type=int, default=1000,
                    help="files per subfolder; 0 for none. Keeps folders under a "
                         "few thousand entries once there are tens of thousands of pages.")
    a = ap.parse_args(argv)

    sources = {"fonts": a.fonts, "figures": a.figures, "cutouts": a.cutouts,
               "dirt": a.dirt, "calligraphy": a.calligraphy}
    assets.configure(**sources)
    if not assets.FONTS:
        ap.error(f"no .ttf or .otf fonts found in {a.fonts}")
    with open(a.pool, encoding="utf-8") as f:
        pool = json.load(f)
    force = json.loads(a.force) if a.force else None
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    for i in range(a.n):
        shard = f"{i // a.shard_size:04d}" if a.shard_size else ""
        jobs.append((i, a.seed * 1000003 + i, a.prefix, a.out, pool,
                     force, a.profile, shard))

    t0 = time.time()
    rows, errors = [], []
    if a.workers > 1:
        import multiprocessing as mp

        with mp.Pool(a.workers, initializer=assets.configure, initargs=tuple(sources.values())) as p:
            for k, r in enumerate(p.imap_unordered(_one, jobs, chunksize=8), 1):
                (errors if "error" in r else rows).append(r)
                if k % 250 == 0 or k == a.n:
                    el = time.time() - t0
                    print(f"  {k}/{a.n}  {el:6.1f}s  {k / el:5.1f} pages/s  "
                          f"errors {len(errors)}", flush=True)
    else:
        for j in jobs:
            r = _one(j)
            (errors if "error" in r else rows).append(r)

    agg = {}
    for r in rows:
        for t, n in r["kinds"].items():
            agg[t] = agg.get(t, 0) + n
    manifest = {"n_pages": len(rows), "n_errors": len(errors),
                "profile": a.profile, "seed": a.seed,
                "seconds": round(time.time() - t0, 1),
                "assets": assets.summary(),
                "region_totals": dict(sorted(agg.items(), key=lambda kv: -kv[1])),
                "pages": sorted(rows, key=lambda r: r["page_id"]), "errors": errors[:50]}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"\n{len(rows)} pages, {len(errors)} errors, {time.time() - t0:.1f}s -> {a.out}")
    print("region totals:", manifest["region_totals"])


if __name__ == "__main__":
    main()
