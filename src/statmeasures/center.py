"""Measures of central tendency."""

__all__ = ("harmonic_mean", "geometric_mean", "trimmed_mean", "winsorized_mean")

import numpy as np
from scipy.stats import mstats


def harmonic_mean(data: np.ndarray) -> float:
    """Return the harmonic mean."""
    return mstats.hmean(data)


def geometric_mean(data: np.ndarray) -> float:
    """Return the geometric mean."""
    return mstats.gmean(data)


def trimmed_mean(data: np.ndarray, alpha: float) -> float:
    """Return the trimmed mean."""
    return mstats.trimmed_mean(data, limits=(alpha, alpha))


def winsorized_mean(data: np.ndarray, alpha: float) -> float:
    """Return the winsorized mean."""
    winsorized_data = mstats.winsorize(data, limits=(alpha, alpha))
    return winsorized_data.mean()
