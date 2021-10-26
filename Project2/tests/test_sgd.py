import random

import autograd.numpy as np
import numpy
import pytest

from csnet.optim import sgd
from csnet.utils import FrankeFunction, cost_mse, create_X

numpy.random.seed(10)
random.seed(10)


@pytest.fixture
def make_data() -> np.ndarray:
    num_points = 100
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)
    z = FrankeFunction(x, y, 0)
    X = create_X(x, y, n=2)
    weights = numpy.random.randn(X.shape[1], 1)
    return X, z, weights


def test_sgd_working(make_data) -> None:
    X, z, weights = make_data
    result = sgd(
        X, z, lr=0.01, epochs=200, cost_function=cost_mse, weights=weights
    )
    assert result is not None
    assert result.shape == (6, 1)


def test_sgd_momnetum_working(make_data) -> None:
    X, z, weights = make_data
    result = sgd(
        X,
        z,
        lr=0.01,
        epochs=200,
        momentum=True,
        alpha=0.3,
        cost_function=cost_mse,
        weights=weights
    )
    assert result is not None
    assert result.shape == (6, 1)

    # test alpha to high
    with pytest.raises(AttributeError):
        _ = sgd(
            X,
            z,
            lr=0.01,
            epochs=500,
            momentum=True,
            alpha=1.0,
            cost_function=cost_mse,
            weights=weights,
        )


def test_sgd_wrong_shape() -> None:
    x = numpy.random.rand(10, 1)
    y = numpy.random.randn(11, 1)
    weights = numpy.random.randn(x.shape[1] + 1, 1)
    with pytest.raises(AttributeError):
        _ = sgd(
            x, y, lr=0.01, epochs=50, cost_function=cost_mse, weights=weights
        )
