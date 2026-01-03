"""Measures of dispersion."""

__all__ = ("stderr",)

import numpy as np

from statmeasures.utils import Vector, ensure_1d


def stderr(vec: Vector, /, *, validate: bool = False) -> float:
    """Return the standard error."""
    v = ensure_1d(vec) if validate else vec
    res = v.std(ddof=1) / np.sqrt(len(v))
    return float(res)
