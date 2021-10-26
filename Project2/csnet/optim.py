from __future__ import annotations

import random
from typing import Callable, Generator

import autograd.numpy as np
from autograd import grad

from csnet.utils import cost_mse


def sgd(
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
) -> np.ndarray:
    """Simple stochastic gradient descent implementation.


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
    np.ndarray

    """
    if x.shape[0] != y.shape[0]:
        raise AttributeError(
            f"Wrong shape of x and y. Shape {x.shape=} != {y.shape=}"
        )
    if momentum and alpha >= 1:
        raise AttributeError("alpha must be less the 1")

    m = 0
    grad_cost = grad(cost_function, 0)
    prev_cost = 2**32

    best_cost = None
    best_weights = None

    for epoch in range(epochs):
        if not silent:
            print(f"Epoch {epoch}/{epochs}")

        for x_sub, y_sub in _random_mini_batch_generator(x, y, batch_size):
            if momentum:
                weights, m = _momentum_sgd_step(
                    x_sub,
                    y_sub,
                    weights,
                    lr,
                    alpha,
                    m,
                    grad_cost,
                    lamb,
                )
            else:
                weights = _sgd_step(x_sub, y_sub, weights, lr, grad_cost, lamb)

        cost = cost_function(weights, x, lamb, y)

        if best_cost is None or best_weights is None or cost < best_cost:
            best_cost = cost
            best_weights = weights.copy()

        # early stop if diff in cost is less the `tol`
        if abs(prev_cost - cost) <= tol:
            break

        prev_cost = cost

    if not silent:
        print(f"{best_cost=}")

    assert best_weights is not None
    return best_weights


def _momentum_sgd_step(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lr: float,
    alpha: float,
    m: float,
    grad_cost_function: Callable,
    lamb: np.ndarray | None = None,
) -> np.ndarray:
    """One optimization step using momentum SGT."""
    gradient = grad_cost_function(weights, x, lamb, y)
    m = alpha * m + (1 - alpha) * gradient
    weights = weights - lr * m
    return weights, m


def _sgd_step(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lr: float,
    grad_cost_function: Callable,
    lamb: np.ndarray | None = None,
) -> np.ndarray:
    """One optimization step using SGT."""
    gradient = grad_cost_function(weights, x, lamb, y)
    return weights - lr * gradient


def _random_mini_batch_generator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    """Generates mini-batches from `x` and `y`."""
    mini_batches = x.shape[0] // batch_size
    m = x.shape[0]
    for _ in range(mini_batches):
        k = random.randint(0, m)
        subset_x = x[k:k + batch_size]
        subset_y = y[k:k + batch_size]
        yield subset_x, subset_y
