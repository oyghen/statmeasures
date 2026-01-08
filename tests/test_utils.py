from contextlib import nullcontext
from typing import Any, TypeAlias

import numpy as np
import pytest
from numpy.typing import ArrayLike

import statmeasures as sm

ContextManager: TypeAlias = (
    nullcontext[None] | pytest.RaisesExc[ValueError] | pytest.RaisesExc[TypeError]
)


class TestEnsure1D:
    @pytest.mark.parametrize(
        "a",
        [
            [1, 2, 3],
            np.array([1, 2, 3]),
            np.array([[1, 2, 3]]),
            np.array([[1], [2], [3]]),
            np.array([[42]]),
        ],
        ids=[
            "python list -> (3,)",
            "already (n,)",
            "(1, n) -> (n,)",
            "(n, 1) -> (n,)",
            "(1, 1) -> (1,)",
        ],
    )
    def test_normalizes_vector_shapes(self, a: ArrayLike):
        v = sm.utils.ensure_1d(a)
        assert isinstance(v, np.ndarray)
        assert v.ndim == 1

        # value-preserving normalization
        expected = np.asarray(a).reshape(-1)
        assert np.array_equal(v, expected)

        # default dtype is floating
        assert np.issubdtype(v.dtype, np.floating)

    @pytest.mark.parametrize(
        "bad_value, ctx",
        [
            (np.array([[1, 2], [3, 4]]), pytest.raises(ValueError)),
            (5, pytest.raises(ValueError)),
            ("abc", pytest.raises(TypeError)),
        ],
        ids=[
            "true matrix -> rejected due to incorrect shape",
            "scalar -> rejected due to incorrect shape",
            "string scalar -> rejected as not array-like",
        ],
    )
    def test_rejects_invalid_shapes_or_scalars(
        self,
        bad_value: Any,
        ctx: ContextManager,
    ):
        with ctx:
            sm.utils.ensure_1d(bad_value)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            sm.utils.ensure_1d([])

    @pytest.mark.parametrize(
        "bad_arr",
        [
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ],
    )
    def test_nonfinite_raises_by_default(self, bad_arr: np.ndarray):
        with pytest.raises(ValueError):
            sm.utils.ensure_1d(bad_arr)

    def test_nonfinite_allowed_when_disabled(self):
        arr = np.array([1.0, np.nan])
        v = sm.utils.ensure_1d(arr, finite=False)
        assert np.isnan(v[1])

    def test_dtype_respected_when_passed(self):
        v = sm.utils.ensure_1d([1, 2, 3], dtype=np.int64)
        assert v.dtype == np.int64
