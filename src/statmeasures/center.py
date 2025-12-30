"""Measures of central tendency."""

__all__ = ("trimmed_mean", "winsorized_mean")

import numpy as np
from scipy.stats import mstats


def trimmed_mean(
    data: np.ndarray,
    limits: tuple[float, float] = (0.05, 0.05),
) -> float:
    """Return the trimmed mean."""
    return mstats.trimmed_mean(data, limits=limits)


def winsorized_mean(
    data: np.ndarray,
    limits: tuple[float, float] = (0.05, 0.05),
) -> float:
    """Return the winsorized mean."""
    winsorized_data = mstats.winsorize(data, limits=limits)
    return winsorized_data.mean()
