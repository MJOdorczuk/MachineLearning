import numpy as np
import pytest
from autograd import grad

from csnet.optim import (
    momentum_sgd_step,
    random_mini_batch_generator,
    sgd_step
)
from csnet.utils import FrankeFunction, cost_mse, create_X


@pytest.fixture
def make_data() -> np.ndarray:
    num_points = 100
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)
    z = FrankeFunction(x, y, 0)
    X = create_X(x, y, n=2)
    weights = np.random.randn(X.shape[1], 1)
    return X, z, weights


def test_sdg_momentum_step(make_data) -> None:
    X, z, weights = make_data

    expected_weights_shape = weights.shape

    w, m = momentum_sgd_step(
        X,
        z,
        weights,
        lr=0.01,
        alpha=0.5,
        m=0,
        grad_cost_function=grad(cost_mse),
        lamb=None,
    )

    assert w.shape == expected_weights_shape


def test_sdg_step(make_data) -> None:
    X, z, weights = make_data

    expected_weights_shape = weights.shape

    w = sgd_step(
        X,
        z,
        weights,
        lr=0.01,
        grad_cost_function=grad(cost_mse),
        lamb=None,
    )

    assert w.shape == expected_weights_shape


def test_generate_mini_batch_working(make_data) -> None:
    X, z, weights = make_data

    n_matches = 0
    for x, y in random_mini_batch_generator(X, z, 2):
        n_matches += 1

    assert n_matches == X.shape[0] // 2
