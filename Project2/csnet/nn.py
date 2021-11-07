from autograd import numpy as np
from typing import Callable
from typing import List
from autograd import grad


'''Example, to be removed when real optimizer developed.
some_counter is just an example argument showing, how the evolving of the optimizer should proceed'''
def SGD(w,b,delta,a,eta,some_counter):
    return (w-np.dot(np.asmatrix(delta), a).T * eta / delta.shape[1],
        b-np.dot(delta, np.ones(delta.shape[1]) / delta.shape[1])*eta,
        lambda w,b,delta,a,eta: SGD(w,b,delta,a,eta,some_counter+1))

def init_SGD(w,b,delta,a,eta):
    return SGD(w,b,delta,a,eta,0)

def mean_squares(x, y):
    return np.mean(np.square(x-y))

def mean_squares_grad(x, y):
    return 2*(y-x)

# type SF =
#     | A of (float -> float * SF)
#
# let rec f (x: int) (c: float) : float * SF=
#     c, A (f (x+1))
#
# That's how you type things like opt in OCaml, do you know, how to type it here?


class layer:
    def __init__(self, activ: Callable[[float],float], input_size: int, output_size: int, opt = init_SGD) -> None:
        """Feed Forward Neural Network layer implementation

        Parameters
        ----------
        activ   :
            Activation function for this layer
        input_size   :
            Size of the previous layer/input neurons
        output_size  :
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

        Yields
        -------
        np.ndarray

        Returns
        -------
        np.ndarray

        """
        self.activation = activ
        self.d_activation = np.vectorize(grad(activ))
        #print(self.d_activation(5.1))
        self.weights = np.random.normal(size=(input_size,output_size))
        self.bias = np.zeros(output_size)
        self.input_size = input_size
        self.outsize = output_size
        self.opt = opt

    def preactivation(self, input: np.ndarray) -> np.ndarray:
        self.z = np.dot(input,self.weights) + self.bias
        return self.z

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.a = self.activation(self.preactivation(input))
        return self.a

    '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
    def backerror(self, z: np.ndarray, nxterr: np.ndarray, d_activation: Callable[[float], float]) -> np.ndarray:
        return np.multiply(d_activation(z).T, np.dot(self.weights, nxterr))

    def update_weights(self, eta: float, delta: np.ndarray, a: np.ndarray):
        self.weights, self.bias, self.opt = self.opt(self.weights, self.bias, delta, a, eta)

class nn:
    def __init__(self,
    layers: List[layer],
    # I know that it can depend on different traits but let us limit ourselves to just the difference for now
    cost: Callable[[np.ndarray],float],
    opt= init_SGD) -> None:
        self.layers = layers
        self.opt = opt
        self.cost = mean_squares
        self.i = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def back(self, x: np.ndarray, y: np.ndarray, eta: float) -> None:
        self.forward(x)
        a = [layer.a for layer in self.layers]
        a.insert(0, x)
        z = [layer.z for layer in self.layers]
        '''
        The cost function can depend on both y and y_tilde separately and not only on their difference
        thence we have to compute grad for each iteration :(
        '''
        dC = mean_squares_grad
        #print(self.layers[-1].d_activation(z[-1]))
        #print(self.layers[-1].d_activation(z[-1]),dC(y,a[-1]),y-a[-1],"\n\n")
        delta = np.multiply(np.asmatrix(self.layers[-1].d_activation(z[-1])), dC(y,a[-1])).T
        self.i += 1
        '''L-1,L-3,...,1'''
        #print(delta)
        for l in np.arange(len(self.layers)-1,0,-1):
            layer = self.layers[l]
            '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
            new_delta = layer.backerror(z[l-1],delta, self.layers[l-1].d_activation)
            layer.update_weights(eta, delta, a[l])
            delta = new_delta
        self.layers[0].update_weights(eta, delta, x)

    def error(self, x: np.ndarray, y: np.ndarray):
        return self.cost(y, self.forward(x))
