from autograd import numpy as np
from typing import Tuple, List, Callable


class Activation:
    """
    Class containing a set of chosen activation functions and their derivatives.
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_grad(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0, 1)

    def leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, self.alpha * x, x)

    def leaky_relu_grad(self, x: np.ndarray)  -> np.ndarray:
        return np.where(x < 0, self.alpha, 1)

    def get_all_activations(self) -> Tuple[Callable, ...]:
        """
        Returns a tuple of all activation functions

        returns:
            A tuple of all activation functions
        """

        return (self.sigmoid, self.relu, self.leaky_relu)

    def get_all_grads(self) -> Tuple[Callable, ...]:
        """
        Returns gradients for all activation functions

        returns:
            A tuple of all gradients for all the activation functions
        """
        return (self.sigmoid_grad, self.relu_grad, self.leaky_relu_grad)

    def get_all_activations_and_grads(self) -> List[Tuple[Callable, Callable]]:
        """
        Returns all activations and their gradients

        args:

        returns:
            A list of tuples containing (activation, grad)
        """
        activations = self.get_all_activations()
        grads = self.get_all_grads()

        return list(zip(activations, grads))
