import numpy
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from csnet.regression import Regression, sgd_regression
from csnet.utils import FrankeFunction, cost_mse, create_X

np.random.seed(10)


def test_regression_works() -> None:
    num_points = 100
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)
    z = FrankeFunction(x, y, 0)
    X = create_X(x, y, n=10)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train = X_train - np.mean(X_train)
    X_test = X_test - np.mean(X_test)
    z_train = z_train - np.mean(z_train)
    z_test = z_test - np.mean(z_test)

    reg = Regression(method="ridge")
    reg.fit(X_train.copy(), z_train.copy(), lamb=1e-2)
    z_pred = reg.predict(X_test.copy())
    assert z_pred is not None

    # reg = Regression(method="ridge_sgd")
    # reg.fit(X_train, z_train, lamb=1e-2)
    # z_pred = reg.predict(X_test)
    # assert z_pred is not None

    reg = Regression(method="ols_sgd")
    reg.fit(X_train, z_train, lamb=1e-2)
    z_pred = reg.predict(X_test)
    assert z_pred is not None

    reg = Regression(method="ols")
    reg.fit(X_train, z_train, lamb=1e-2)
    z_pred = reg.predict(X_test)
    assert z_pred is not None


@pytest.fixture
def make_data() -> np.ndarray:
    num_points = 100
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)
    z = FrankeFunction(x, y, 0)
    X = create_X(x, y, n=2)
    weights = numpy.random.randn(X.shape[1], 1)
    return X, z, weights


def test_sgd_regression_working(make_data) -> None:
    X, z, weights = make_data
    result, mse = sgd_regression(
        X, z, lr=0.01, epochs=200, cost_function=cost_mse, weights=weights
    )
    assert result is not None
    assert isinstance(mse, list)
    assert result.shape == (6, 1)


def test_sgd_regression_momnetum_working(make_data) -> None:
    X, z, weights = make_data
    result, _ = sgd_regression(
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


def test_sgd_regression_wrong_shape() -> None:
    x = numpy.random.rand(10, 1)
    y = numpy.random.randn(11, 1)
    weights = numpy.random.randn(x.shape[1] + 1, 1)
    with pytest.raises(AttributeError):
        _ = sgd_regression(
            x, y, lr=0.01, epochs=50, cost_function=cost_mse, weights=weights
        )
