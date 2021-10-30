from autograd import numpy as np
from typing import Callable
from typing import List
from autograd import grad
from time import time


'''Example, to be removed when real optimizer developed.
some_counter is just an example argument showing, how the evolving of the optimizer should proceed'''
def SGD(w,b,delta,a,eta,some_counter):
    return (w-np.dot(a.reshape(-1,1),delta.reshape(1,-1))*eta,
        b-delta*eta,
        lambda w,b,delta,a,eta: SGD(w,b,delta,a,eta,some_counter+1))

def initSGD(w,b,delta,a,eta):
    return SGD(w,b,delta,a,eta,0)

def meanSquares(x,y):
    return np.mean((x-y)**2)

# type SF =
#     | A of (float -> float * SF)
#
# let rec f (x: int) (c: float) : float * SF=
#     c, A (f (x+1))
#
# That's how you type things like opt in OCaml, do you know, how to type it here?


class layer:
    def __init__(self, activ: Callable[[float],float], input_size: int, output_size: int, opt = initSGD) -> None:
        self.activation = activ
        self.d_activation = np.vectorize(grad(activ))
        #print(self.d_activation(5.1))
        self.weights = np.random.normal(size=(input_size,output_size))
        self.bias = np.zeros(output_size)
        self.input_size = input_size
        self.outsize = output_size
        self.opt = opt

    def preactivation(self, input: np.ndarray) -> np.ndarray:
        return np.dot(input,self.weights) + self.bias

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activation(self.preactivation(input))

    '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
    def backerror(self, z: np.ndarray, nxterr: np.ndarray, d_activation: Callable[[float], float]) -> np.ndarray:
        return d_activation(z) * np.dot(self.weights, nxterr)

    def update_weights(self, eta: float, delta: np.ndarray, a: np.ndarray):
        self.weights, self.bias, self.opt = self.opt(self.weights, self.bias, delta, a, eta)

class nn:
    def __init__(self,
    layers: List[layer],
    cost: Callable[[np.ndarray, np.ndarray],float],
    opt= initSGD) -> None:
        self.layers = layers
        self.opt = opt
        self.cost = cost
        self.i = 0
        self.deltas = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def stepforward(self, x: np.ndarray) -> List[np.ndarray]:
        a = [x]
        z = []
        for layer in self.layers:
            z.append(layer.preactivation(a[-1]))
            a.append(layer.forward(a[-1]))
        return (a, z)

    def back(self, x: np.ndarray, y: np.ndarray, eta: float) -> None:
        a, z = self.stepforward(x)
        '''
        The cost function can depend on both y and y_tilde separately and not only on their difference
        thence we have to compute grad for each iteration :(
        '''
        dC = grad(self.cost,1)
        #print(self.layers[-1].d_activation(z[-1]))
        #print(self.layers[-1].d_activation(z[-1]),dC(y,a[-1]),y-a[-1],"\n\n")
        delta = self.layers[-1].d_activation(z[-1])*dC(y,a[-1])
        self.deltas.append(delta)
        self.i += 1
        '''L-1,L-3,...,1'''
        #print(delta)
        for l in np.arange(len(self.layers)-1,0,-1):
            layer = self.layers[l]
            '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
            ndelta = layer.backerror(z[l-1],delta, self.layers[l-1].d_activation)
            layer.update_weights(eta, delta, a[l])
            delta = ndelta
        self.layers[0].update_weights(eta, delta, x)

    def error(self, x: np.ndarray, y: np.ndarray):
        return self.cost(y, self.forward(x))

    def batch_error(self, x, y):
        y_tilde = np.apply_along_axis(self.forward, 1, x)
        return self.cost(y, y_tilde)
