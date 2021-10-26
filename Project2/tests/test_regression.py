import numpy as np
from sklearn.model_selection import train_test_split

from csnet.regression import Regression
from csnet.utils import FrankeFunction, create_X


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
    reg.fit(X_train, z_train, lamb=1e-2)
    z_pred = reg.predict(X_test)
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
