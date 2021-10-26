from __future__ import annotations

import random
from typing import Generator

import numpy as np


def sgd(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    silent: bool = True,
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


    Returns
    -------
    np.ndarray

    """
    if x.shape != y.shape:
        raise AttributeError(
            f"Wrong shape of x and y. Shape {x.shape=} != {y.shape=}"
        )

    theta = np.random.randn(x.shape[1] + 1, 1)

    X = np.column_stack((np.ones((x.shape[0], 1)), x))

    for epoch in range(epochs):
        if not silent:
            print(f"Epoch {epoch}/{epochs}")

        for x_sub, y_sub in _random_mini_batch_generator(X, y, batch_size):
            theta = (
                theta - lr * 2 * x_sub.T @ ((x_sub @ theta) - y_sub)
            )

    return theta


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
