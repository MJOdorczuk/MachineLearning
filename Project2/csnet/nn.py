import numpy as np
from typing import Callable
from typing import List
from jax import grad


'''Example, to be removed when real optimizer developed.
some_counter is just an example argument showing, how the evolving of the optimizer should proceed'''
def SGD(w,b,delta,a,eta,some_counter):
    return (w-delta.reshape(-1,1).T.dot(a.reshape(1,-1))*eta,
        b-delta*eta,
        lambda w,b,delta,a,eta: SGD(w,b,delta,a,eta,some_counter+1))

def initSGD(w,b,delta,a,eta):
    return SGD(w,b,delta,a,eta,0)

# type SF =
#     | A of (float -> float * SF)
#
# let rec f (x: int) (c: float) : float * SF=
#     c, A (f (x+1))
#
# That's how you type things like opt in OCaml, do you know, how to type it here?


class layer:
    def __init__(self, activ: Callable[[float],float], input_size: int, output_size: int, opt = initSGD) -> None:
        self.activation = np.vectorize(activ)
        self.d_activation = np.vectorize(grad(activ))
        self.weights = np.random.normal((input_size,output_size))
        self.bias = np.random.normal((input_size))
        self.input_size = input_size
        self.outsize = output_size
        self.opt = opt

    def preactivation(self, input: np.ndarray) -> np.ndarray:
        return input.dot(self.weights) + self.bias

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activation(self.preactivation(input))

    '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
    def backerror(self, x: np.ndarray, nxterr: np.ndarray) -> np.ndarray:
        z = self.preactivation(x)
        return self.d_activation(z) * nxterr.dot(self.weights)

    def update_weights(self, eta: float, delta: np.ndarray, a: np.ndarray):
        self.weights, self.bias, self.opt = self.opt(self.weights, self.bias, delta, a, eta)

class nn:
    def __init__(self, layers: List[layer], output_activation: Callable[[float],float], cost: Callable[[np.ndarray, np.ndarray],float],opt) -> None:
        self.layers = layers
        self.activation = np.vectorize(output_activation)
        self.d_activation = np.vectorize(grad(output_activation))
        self.opt = opt
        self.cost = cost # Right now it is not necessary, but may be useful in the future

    def preactivation(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(x)
        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(self.preactivation(x))

    def stepforward(self, x: np.ndarray) -> List[np.ndarray]:
        a = [x]
        for layer in self.layers:
            a.append(layer.forward(a[-1]))
        return a

    def back(self, x: np.ndarray, y: np.ndarray, eta: float) -> None:
        z = self.preactivation(x)
        a = self.stepforward(x)
        '''
        The cost function can depend on both y and y_tilde separately and not only on their difference
        thence we have to compute grad for each iteration :(
        '''
        dC = grad(lambda y_tilde: self.cost(y,y_tilde))
        delta = self.d_activation(z)*dC(a[-1])
        '''L-2,L-3,...,1'''
        for l in np.arange(len(self.layers)-2,0,-1):
            layer = self.layers[l]
            '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
            ndelta = layer.backerror(a[l],delta)
            layer.update_weights(eta, delta, a[l])
            delta = ndelta
        self.layers[0].update_weights(eta, delta, a[0])
