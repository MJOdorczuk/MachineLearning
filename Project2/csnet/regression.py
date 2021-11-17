"""Code from project 1 adapted to be used in project 2."""
from __future__ import annotations

from typing import Callable, Literal  # noqa: TYP001

import numpy as np
from autograd import grad

from csnet.optim import sgd_minibatch, SGD
from csnet.utils import cost_mse, cost_mse_ridge


def sgd_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    cost_function: Callable = cost_mse,
    lamb: np.ndarray | None = None,
    batch_size: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    silent: bool = True,
    momentum: bool = False,
    alpha: float = 0.5,
    tol: float = 1e-8,
    lr_scaling_func: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Stochastic gradient descent implementation for optimize `beta`
    in regression.


    Parameters
    ----------
    x   :
        Input
    y   :
        Targets
    batch_size  :
        Number of samples to use in each mini-batch.
    epochs  :
        Number of iteration over the mini-batches.
    lr  :
        Learning rate/step lenght used in gradient decent.
    silent  :
        Print output or not
    momentum :
        Use momentum in calculations
    alpha   :
        Gradient decay constant used in momentum.

    Returns
    -------
    np.ndarray, list[float]

    """
    if x.shape[0] != y.shape[0]:
        raise AttributeError(
            f"Wrong shape of x and y. Shape {x.shape=} != {y.shape=}"
        )

    if lr_scaling_func is None:
        lr_scaling_func = lambda lr: lr

    # To store momentum between epochs if SGD with momentum is used.
    m: np.ndarray | None = None

    # Derivitive of cost function
    grad_cost = grad(cost_function, 0)

    # Variables to store results
    best_cost = None
    best_weights = None
    prev_cost = cost_function(weights, x, lamb, y)
    all_cost = [prev_cost]
    all_weights = [weights.copy()]

    optim = SGD(
        lr=lr,
        batch_size=batch_size,
        use_momentum=momentum,
        alpha=alpha,
        decay=lr / epochs,
    )

    for epoch in range(epochs):
        weights = sgd_minibatch(
            x=x,
            y=y,
            weights=weights,
            batch_size=batch_size,
            grad_cost=grad_cost,
            lamb=lamb,
            optimizer=optim,
        )

        cost = cost_function(weights, x, lamb, y)

        if best_cost is None or best_weights is None or cost < best_cost:
            best_cost = cost
            best_weights = weights.copy()

        # early stop if diff in cost is less the `tol`
        if np.abs(prev_cost - cost) <= tol:
             break

        all_cost.append(cost)
        prev_cost = cost
        all_weights.append(weights.copy())

    assert best_weights is not None
    return best_weights, all_cost


def least_square(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    *,
    n_epochs: int,
    display: bool,
    lamb: float,
    lr: float,
    batch_size: int = 2,
    momentum: bool = True,
    alpha: float = 0.5,
) -> np.ndarray:
    """Calculating the beta from project 1.

    Note some of parameter is not used in this function to get same function
    signature across methods.
    """
    return (
        np.linalg.pinv(
            x_value.transpose().dot(x_value)
        ).dot(x_value.transpose().dot(y_value)),
        [cost_mse(beta=weights, X=x_value, y_true=y_value, lamb=None)],
    )


def ridge(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    *,
    n_epochs: int,
    display: bool,
    lamb: float,
    lr: float,
    batch_size: int = 2,
    momentum: bool = True,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[float]]:
    """Calculating the beta from project 1.

    Note some of parameter is not used in this function to get same function
    signature across methods.
    """
    weights = np.linalg.pinv(
        x_value.T.dot(x_value) + lamb * np.identity(x_value.shape[1])
    ).dot(x_value.T).dot(y_value)

    mse = [cost_mse_ridge(beta=weights, X=x_value, y_true=y_value, lamb=lamb)]
    return weights, mse


def least_square_sgd(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: float,
    lr: float,
    batch_size: int = 2,
    momentum: bool = True,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[float]]:
    """Calculating the beta by using the sgd_regression optimization function.

    Note some of parameter is not used in this function to get same function
    signature across methods.
    """
    return sgd_regression(
        x_value,
        y_value,
        weights=weights,
        batch_size=batch_size,
        epochs=n_epochs,
        lr=lr,
        momentum=momentum,
        alpha=alpha,
        cost_function=cost_mse,
        silent=not display,
    )


def ridge_sgd(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    display: bool,
    lamb: float,
    lr: float,
    batch_size: int = 2,
    momentum: bool = True,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[float]]:
    """Calculating the beta by using the sgd_regression optimization function.
    """
    result, mse = sgd_regression(
        x_value,
        y_value,
        weights=weights,
        batch_size=batch_size,
        epochs=n_epochs,
        lr=lr,
        momentum=momentum,
        alpha=alpha,
        cost_function=cost_mse_ridge,
        lamb=lamb,
        silent=not display,
    )
    return result, mse


METHODS = Literal["ols", "ols_sgd", "ridge", "ridge_sgd"]


class Regression:
    """Regression model.

    Example
    -------
    >>> model = Regression(method="ols")
    >>> model.fit(X, y)
    >>> preds = model.pred(X_2)

    """

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
        lr: float = 0.01,
        lamb: float | None = None,
        n_epochs: int = 100,
        display: bool = False,
        batch_size: int = 2,
        momentum: bool = True,
        alpha: float = 0.5,
        reset_parameters: bool = True,
    ) -> None:
        """Perform a fit of the model with given parameters."""
        if reset_parameters or self.parameters is None:
            self.parameters = np.random.randn(x.shape[1], 1)

        if lamb is None and self.func.__name__ in ["ridge", "ridge_sgd"]:
            raise AttributeError

        if lamb is None:
            lamb = 0

        self.parameters, self._all_mse = self.func(
            x,
            y,
            weights=self.parameters,
            n_epochs=n_epochs,
            display=display,
            lamb=lamb,
            lr=lr,
            batch_size=batch_size,
            momentum=momentum,
            alpha=alpha,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make a prediction on the fitted model."""
        if self.parameters is None:
            raise RuntimeError("Model is not fitted.")

        y = np.dot(x, self.parameters)
        return y
