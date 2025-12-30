import numpy as np
import numpy.testing as npt
import pytest

import statmeasures as sm


class TestMeasuresOfDispersion:
    def test_stderr(self, data: np.array):
        result = sm.spread.stderr(data)
        npt.assert_almost_equal(result, 5.9632, decimal=4)

    @pytest.fixture
    def data(self) -> np.array:
        return np.array([98, 127, 82, 67, 121, 119, 92, 110, 113, 107])
