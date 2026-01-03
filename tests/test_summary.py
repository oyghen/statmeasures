import numpy as np
import numpy.testing as npt
import pytest

import statmeasures as sm
from statmeasures.utils import Vector


class TestMeasureSummary:
    def test_basic(self, vec: Vector):
        result = sm.MeasureSummary(vec)
        npt.assert_almost_equal(result.mean(), 14.7, decimal=2)
        npt.assert_almost_equal(result.median(), 6.0, decimal=2)
        npt.assert_almost_equal(result.harmonic_mean(), 3.87, decimal=2)
        npt.assert_almost_equal(result.geometric_mean(), 6.12, decimal=2)
        npt.assert_almost_equal(result.trimmed_mean(alpha=0.2), 5.67, decimal=2)
        npt.assert_almost_equal(result.winsorized_mean(alpha=0.2), 5.6)
        npt.assert_almost_equal(result.dwe(), 6.72, decimal=2)
        npt.assert_almost_equal(result.spwe(), 6.01, decimal=2)
        npt.assert_almost_equal(result.stderr(), 9.51, decimal=2)

    def test_summary_method(self, vec: Vector):
        result = sm.MeasureSummary(vec).summary()
        assert isinstance(result, dict)
        assert len(result) == 9

    @pytest.fixture
    def vec(self) -> Vector:
        return np.array([1, 3, 4, 4, 5, 7, 7, 7, 9, 100])
