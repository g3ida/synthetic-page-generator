"""What the scan simulation shares with the rest of the generator."""

from __future__ import annotations

import numpy as np

# Fine-stroked cursive faces. Stroke thickening closes their counters, so
# `generator.degrade_tracked` never applies it to them; with that exemption they
# stay legible at every tier, so no tier cap is needed.
FINE_STROKE = ("Nastaliq", "Gulzar", "ArefRuqaa", "Qahiri", "Alkalami", "Amiri")


def rng_normal(shape, rng, sigma):
    """Gaussian noise from the seeded rng, so a run is reproducible."""
    g = np.random.default_rng(rng.getrandbits(32))
    return g.normal(0, sigma, shape)
