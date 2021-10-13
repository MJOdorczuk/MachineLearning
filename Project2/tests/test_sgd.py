import random

import numpy as np
import pytest

from csnet.optim import sgd

np.random.seed(0)
random.seed(0)


def test_sgd_working() -> None:
    m = 100
    x = 2 * np.random.rand(m, 1)
    y = 4 + 3 * x + np.random.randn(m, 1)
    result = sgd(x, y, lr=0.01, epochs=50)
    assert pytest.approx(result[0], 0.1) == 4
    assert pytest.approx(result[1], 0.1) == 3


def test_sgd_wrong_shape() -> None:
    x = np.random.rand(10, 1)
    y = np.random.randn(11, 1)
    with pytest.raises(AttributeError):
        _ = sgd(x, y, lr=0.01, epochs=50)
