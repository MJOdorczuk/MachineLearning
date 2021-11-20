from __future__ import annotations

from typing import Generator

import numpy as np


class SGD():
    """Stochastic gradient decent optimizer.

    Example
    -------
    >>> EPOCHS = 100
    >>> LR = 0.1
    >>> otim = SDG(lr=LR, batch_size=4, use_momentum=True, decay=LR/EPOCHS)
    >>> weights = optim.step(weights, gradient)
    """
    def __init__(self,
            lr: float = 0.01,
            silent: bool = True,
            use_momentum: bool = False,
            alpha: float = 0.9,
            use_lr_decay: bool = True,
            decay: float = 0.001,
        ) -> None:

        self.lr = lr
        self.silent = silent
        self.use_momentum = use_momentum
        self.alpha = alpha
        self.decay = decay
        self.use_lr_decay = use_lr_decay
        self.steps_done = 0
        self.momentum_weights = 0
        self.momentum_bias = 0

    def _step_lr(self) -> float:
        """Get learning rate for next step."""
        if self.use_lr_decay is False:
            return self.lr

        self.lr = self.lr * (1.0 / (1 + self.decay * self.steps_done))

        if self.lr < 0:
            print(
                f"Learning rate decayed to zero. lr: {self.lr}, "
                f"decay: {self.decay}, sets_done: {self.steps_done}"
            )
        return self.lr

    def step(
        self,
        weights: np.ndarray,
        grad: np.ndarray,
        bias = False
    ) -> np.ndarray:
        if self.use_momentum:
            return self.momentum_step(weights,grad, bias)
        else:
            return self.sgd_step(weights,grad)

    def sgd_step(self,
            weights: np.ndarray,
            grad: np.ndarray,
        ) -> np.ndarray:

        updated_weights = weights - self._step_lr() * grad
        self.steps_done += 1

        return np.nan_to_num(updated_weights)

    def momentum_step(self,
            weights: np.ndarray,
            grad: np.ndarray,
            bias: bool = False
        ) -> np.ndarray:
        """One optimization step using momentum SGD."""
        if bias:
            self.momentum_bias = (
                self.alpha * self.momentum_bias + (1 - self.alpha) * grad
            )
            updated_weights = (
                weights - self._step_lr() * self.momentum_bias
            )
        else:
            self.momentum_weights = (
                self.alpha * self.momentum_weights + (1 - self.alpha) * grad
            )
            updated_weights = (
                weights - self._step_lr() * self.momentum_weights
            )

        self.steps_done += 1

        return np.nan_to_num(updated_weights)


def random_mini_batch_generator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    """Generates mini-batches from `x` and `y`."""
    mini_batches = x.shape[0] // batch_size
    m = x.shape[0]
    for _ in range(mini_batches):
        # With replacement as done in the lectures, but also with a constant batch size.
        k = np.random.randint(m, size = batch_size)
        subset_x = x[k]
        subset_y = y[k]
        if subset_x.shape[0] != 0:
            yield subset_x, subset_y


def sgd_minibatch(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    optimizer: SGD,
    grad_cost: Callable,
    lamb: float | None,
) -> np.ndarray:
    """Run SGD over all minibatches of the input."""
    for x_sub, y_sub in random_mini_batch_generator(x, y, batch_size):
        gradient = grad_cost(weights, x_sub, lamb, y_sub)
        weights = optimizer.step(weights=weights, grad=gradient)
    return weights
