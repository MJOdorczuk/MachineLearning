from autograd import numpy as np
from typing import Callable
from typing import List
from autograd import grad, elementwise_grad
from sklearn.metrics import r2_score


def mean_squared_error(x):
    return np.mean(np.square(x))

'''Example, to be removed when real optimizer developed.
some_counter is just an example argument showing, how the evolving of the optimizer should proceed'''
def SGD(w,b,delta,a,eta,some_counter):
    new_weights = w - eta * np.dot(delta, a).T / delta.shape[1]
    new_biases = b - eta * np.mean(delta, 1)
    new_opt = lambda w,b,delta,a,eta: SGD(w,b,delta,a,eta,some_counter+1)
    return new_weights, new_biases, new_opt

def init_SGD(w,b,delta,a,eta):
    return SGD(w,b,delta,a,eta,0)


class layer:
    def __init__(self, activ: Callable[[float],float], input_size: int, output_size: int, opt = init_SGD) -> None:
        """Feed Forward Neural Network layer implementation, representing the net of connections between two layers.

        Parameters
        ----------
        activ   :
            Activation function for this layer.
        input_size   :
            Size of the previous layer/input neurons.
        output_size  :
            Size of the layer/output neurons.
        opt  :
            Optimizer function updating weights and biases based on the error.

        """
        self.activation = activ
        self.d_activation = np.vectorize(grad(activ))
        self.weights = np.random.normal(size=(input_size,output_size))
        self.bias = np.zeros((output_size,))
        self.opt = opt

    def pre_activation(self, input: np.ndarray) -> np.ndarray:
        """Layer output before feeding to the activation function.

        Parameters
        ----------
        input   :
            Data given from the input neurons

        Returns
        -------
        np.ndarray

        """
        self.z = np.dot(input,self.weights) + self.bias
        return self.z

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Layer output after feeding to the activation function.

        Parameters
        ----------
        input   :
            Data given from the input neurons

        Returns
        -------
        np.ndarray

        """
        self.a = self.activation(self.pre_activation(input))
        return self.a

    def back_error(self, z: np.ndarray, delta: np.ndarray, f_prime: Callable[[float], float]) -> np.ndarray:
        """Backwards propagated error of the previous layer based on the error of the current output, based on the formula:
        delta_j^{l-1}=sum_k delta_k^l w_{kj}^lf'(z_j^{l-1})

        Parameters
        ----------
        z   :
            Outputs from previous layer before activation
        delta   :
            Backwards propagated error for the current layer (delta_k^l).
        f_prime  :
            Gradient of the activation function of the previous layer (f')

        Returns
        -------
        np.ndarray

        """
        return np.multiply(f_prime(z).T, np.dot(self.weights, delta))

    def update_weights(self, eta: float, delta: np.ndarray, a: np.ndarray) -> None:
        """Update weights using the optimizer based on the learning rate and the backwards propagated error of this layer.

        Parameters
        ----------
        eta   :
            Learning rate.
        delta   :
            Backwards propagated error for the current layer.
        a  :
            Output of the current layer with the error delta

        """
        self.weights, self.bias, self.opt = self.opt(self.weights, self.bias, delta, a, eta)

class nn:
    def __init__(self, layers: List[layer], cost: Callable[[np.ndarray],float] = mean_squared_error) -> None:
        """Feed Forward Neural Network implementation.

        Parameters
        ----------
        layers   :
            List of network layers.
        cost   :
            Cost function in regards to which the optimization is done.

        """
        self.layers = layers
        self.cost = cost
        self.d_cost = elementwise_grad(self.cost)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Neural network output after feeding to the activation function.

        Parameters
        ----------
        input   :
            Data fed to the input neurons

        Returns
        -------
        np.ndarray

        """
        self.x = x
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def back(self, y: np.ndarray, eta: float) -> None:
        """Back propagation implementation based on the last forward feed and the expected values:

        Parameters
        ----------
        y   :
            Expected output for last forward feed.
        eta   :
            Learning rate.
        """
        a = [layer.a for layer in self.layers]
        z = [layer.z for layer in self.layers]

        error = np.apply_along_axis(self.d_cost, 0, a[-1].T - y)
        delta = np.multiply(self.layers[-1].d_activation(z[-1]).T, error)
        '''L-1,L-3,...,1'''
        #print(delta)
        for l in np.arange(len(self.layers)-1,0,-1):
            layer = self.layers[l]
            '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
            new_delta = layer.back_error(z[l-1],delta, self.layers[l-1].d_activation)
            layer.update_weights(eta, delta, a[l-1])
            delta = new_delta
        self.layers[0].update_weights(eta, delta, self.x)

    def error(self, x: np.ndarray, y: np.ndarray):
        return self.cost(y - self.forward(x))
