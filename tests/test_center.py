import numpy as np
import numpy.testing as npt
import pytest

import statmeasures as sm


class TestMeasuresOfCentralTendency:
    def test_harmonic_mean(self, data: np.array):
        result = sm.center.harmonic_mean(data)
        npt.assert_almost_equal(result, 3.87, decimal=2)

    def test_trimmed_mean(self, data: np.array):
        """Test the 20% trimmed mean."""
        alpha = 0.2
        result = sm.center.trimmed_mean(data, limits=(alpha, alpha))
        npt.assert_almost_equal(result, 5.67, decimal=2)

    def test_winsorized_mean(self, data: np.array):
        """Test the 20% winsorized mean."""
        alpha = 0.2
        result = sm.center.winsorized_mean(data, limits=(alpha, alpha))
        npt.assert_almost_equal(result, 5.6)

    @pytest.fixture
    def data(self) -> np.array:
        return np.array([1, 3, 4, 4, 5, 7, 7, 7, 9, 100])
