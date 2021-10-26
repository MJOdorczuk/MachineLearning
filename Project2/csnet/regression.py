"""Code from project 1 adapted to be used in project 2."""
from __future__ import annotations

from typing import Literal  # noqa: TYP001

import autograd.numpy as np
import numpy

from csnet.optim import sgd
from csnet.utils import cost_mse, cost_mse_ridge


def least_square(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: np.ndarray,
) -> np.ndarray:
    return np.linalg.pinv(
        x_value.transpose().dot(x_value)
    ).dot(x_value.transpose().dot(y_value))


def least_square_sgd(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: np.ndarray,
) -> np.ndarray:
    return sgd(
        x_value,
        y_value,
        weights=weights,
        batch_size=2,
        epochs=n_epochs,
        lr=0.001,
        momentum=True,
        alpha=0.5,
        cost_function=cost_mse,
        silent=not display,
    )


def ridge_sgd(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: np.ndarray,
) -> np.ndarray:
    # TODO grid search for lambda
    result = sgd(
        # x_value.T.dot(x_value) + lamb*np.identity(x_value.shape[1]),
        # TODO: Need to add regularisation term to X some how
        x_value,
        y_value,
        weights=weights,
        batch_size=2,
        epochs=n_epochs,
        lr=0.001,
        momentum=True,
        alpha=0.5,
        cost_function=cost_mse_ridge,
        lamb=lamb,
        silent=not display,
    )
    return result.dot(x_value.T).dot(y_value)


def ridge(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: np.ndarray,
) -> np.ndarray:
    return np.linalg.pinv(
        x_value.T.dot(x_value) + lamb * np.identity(x_value.shape[1])
    ).dot(x_value.T).dot(y_value)


METHODS = Literal["ols", "ols_sgd", "ridge", "ridge_sgd"]


class Regression:
    def __init__(
        self,
        method: METHODS = "ols",
    ) -> None:
        self.parameters: np.ndarray | None = None

        if method == "ols":
            self.func = least_square
        elif method == "ols_sgd":
            self.func = least_square_sgd
        elif method == "ridge":
            self.func = ridge
        elif method == "ridge_sgd":
            self.func = ridge_sgd
        else:
            raise NotImplementedError

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        lamb: float | None = None,
        n_epochs: int = 100,
        display: bool = False,
    ) -> None:
        if self.parameters is None:
            self.parameters = numpy.random.randn(x.shape[1], 1)

        if lamb is None and self.func.__name__ in ["ridge", "ridge_sgd"]:
            raise AttributeError

        self.parameters = self.func(
            x,
            y,
            weights=self.parameters,
            n_epochs=n_epochs,
            display=display,
            lamb=lamb,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:

        if self.parameters is None:
            raise RuntimeError("Model is not fitted.")

        y = np.dot(x, self.parameters)
        return y
