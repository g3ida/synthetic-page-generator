"""Synthetic Arabic document pages with exact layout ground truth."""

from __future__ import annotations

from . import assets
from .generator import PROFILES, BadPage, build, sample_spec
from .regions import TYPES, PageLayout, Region

__version__ = "0.1.0"

__all__ = ["PROFILES", "TYPES", "BadPage", "PageLayout", "Region", "assets", "build", "sample_spec"]
