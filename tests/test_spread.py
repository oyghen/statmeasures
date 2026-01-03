import numpy as np
import numpy.testing as npt
import pytest

import statmeasures as sm
from statmeasures.utils import Vector


class TestMeasuresOfDispersion:
    def test_stderr(self, vec: Vector):
        result = sm.spread.stderr(vec)
        npt.assert_almost_equal(result, 5.9632, decimal=4)

    @pytest.fixture
    def vec(self) -> Vector:
        return np.array([98, 127, 82, 67, 121, 119, 92, 110, 113, 107])
