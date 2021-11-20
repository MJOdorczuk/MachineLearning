from __future__ import annotations

from typing import NamedTuple

import autograd.numpy as np
import numpy
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def FrankeFunction(x, y, sigma=0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    noise = numpy.random.normal(0, 1, x.shape[0])
    noise = noise.reshape(x.shape[0], 1)

    return (term1 + term2 + term3 + term4).reshape(-1, 1) + sigma * noise


def create_X(x, y, n):
    """Create a polynomial designe matrix."""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l_beta = int((n + 1) * (n + 2) / 2)   # Number of elements in beta
    X = numpy.ones((N, l_beta))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)

    return X


def cost_mse(beta, X, lamb, y_true):
    """Cost function for OLS."""
    inner = (y_true - np.dot(X, beta))
    ret = 1 / len(y_true) * np.dot(inner.T, inner)
    assert ret.shape == (1, 1), ret.shape
    return ret[0, 0]


def cost_mse_ridge(beta, X, lamb, y_true):
    """Cost function for ridge."""
    assert lamb is not None
    inner = y_true - np.dot(X, beta)
    outer = 1 / len(y_true) * np.dot(inner.T, inner)

    ret = outer + lamb * np.dot(beta.T, beta)
    assert ret.shape == (1, 1), ret.shape
    return ret[0, 0]


class Result(NamedTuple):
    lr: float
    lamb: float
    batchsize: int
    momentum: bool
    alpha: float
    val_mse: float
    val_r2: float
    all_mse: list[float]


def regression_grid_search(
    model,
    X,
    y,
    *,
    n_epochs: int,
    lrs: list[float],
    batchsize: list[int],
    lambdas: list[float],
    momentum: bool,
    alpha: list[float],
) -> list[Result]:
    """Perform a grid search for all given parameters on model."""

    all_results: list[Result] = []

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=41,
    )

    number_of_params = len(lrs) * len(batchsize) * len(lambdas) * len(alpha)

    if momentum:
        momentum_perm = [True]
    else:
        momentum_perm = [False]

    if alpha is None:
        alpha = [0]

    it = 1
    for mom in momentum_perm:
        for a in alpha:
            for lam in lambdas:
                for b in batchsize:
                    for lr in lrs:
                        print(f"Iteration {it}/{number_of_params}", end="\r")
                        model.fit(
                            X_train,
                            y_train,
                            lamb=lam,
                            lr=lr,
                            n_epochs=n_epochs,
                            display=False,
                            batch_size=b,
                            momentum=mom,
                            alpha=a,
                        )

                        pred = model.predict(X_val)

                        all_results.append(
                            Result(
                                lr=lr,
                                lamb=lam,
                                batchsize=b,
                                momentum=mom,
                                alpha=a,
                                val_mse=mean_squared_error(y_val, pred),
                                val_r2=r2_score(y_val, pred),
                                all_mse=model._all_mse,
                            )
                        )
                        it += 1

    # sort results by mse, best results in element 0
    all_results = sorted(all_results, key=lambda r: r.val_mse)
    assert len(all_results) > 0 and isinstance(all_results[0], Result)

    return all_results
