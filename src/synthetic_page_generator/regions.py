"""What a generated page is made of: each region's box, type and place in the reading order."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: Every region type the generator draws. `toc` (an index of titles, dot leaders
#: and page numbers) is kept apart from `table`; `text` is short standalone text
#: that titles nothing and runs into nothing — an imprint, a URL, an ownership
#: inscription, a citation line.
TYPES = [
    "paragraph", "text", "footnote", "hashiya", "header", "footer",
    "page_number", "heading", "verse", "table", "toc", "figure", "chart",
    "rule", "ornament", "stamp", "caption", "calligraphy", "non_text",
    "other",
]


@dataclass
class Region:
    id: int
    bbox: list          # [x0, y0, x1, y1] absolute px, origin top-left
    type: str           # one of TYPES
    score: float
    native: str         # the type as drawn, before any mapping
    order: int = None   # reading order; None where a region is not read
    column: int = None  # 0 = first read = RIGHTMOST on an RTL page
    #: 4 corner points [[x,y]*4] when the region is not axis-aligned -- a page
    #: with scan skew. `bbox` stays the axis-aligned hull of these, so every
    #: consumer keeps working and tight-fitting consumers can use the quad.
    quad: list = None

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"{self.type!r} not in the closed vocabulary")
        x0, y0, x1, y1 = self.bbox
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"degenerate bbox {self.bbox} type={self.type}")
        self.bbox = [int(round(v)) for v in self.bbox]
        if self.quad is not None:
            self.quad = [[int(round(a)), int(round(b))] for a, b in self.quad]


@dataclass
class PageLayout:
    page_id: str
    image: str
    width: int
    height: int
    source: str = "synthetic"
    regions: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["regions"] = [asdict(r) for r in self.regions]
        return d
