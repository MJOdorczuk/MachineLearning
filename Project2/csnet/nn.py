import numpy as np
from typing import Callable
from typing import List

class activation:
    def __init__(self, f: Callable[[float], float], df: Callable[[float], float]):
        self.f = f
        self.df = df

class layer:
    def __init__(self, activ: activation,size: int,outsize: int) -> None:
        self.f = np.vectorize(activ.f)
        self.df = np.vectorize(activ.df)
        self.neurons = np.zeros([size,outsize])
        self.bias = np.zeros([size])
        self.size = size
        self.outsize = outsize

    def activations(self, input: np.ndarray) -> np.ndarray:
        return input.dot(self.neurons) + self.bias

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.f(self.activations(input))

    '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
    def backerror(self, a: np.ndarray, nxterr: np.ndarray) -> np.ndarray:
        z = self.activations(a)
        return np.array([np.sum(nxterr[k]*self.neurons[j,k]*self.df(z[j])
            for k in range(self.outsize)) for j in range(self.size)])

    def improve(self, eta: float, delta: np.ndarray, a: np.ndarray):
        for j in self.size:
            for k in self.outsize:
                self.neurons[j,k] = self.neurons[j,k]-eta*delta[j]*a[k]
            self.bias[j] = self.bias[j] - eta*delta[j]

class nn:
    def __init__(self, layers: List[layer], output_activation: activation) -> None:
        self.layers = layers
        self.f = np.vectorize(output_activation.f)
        self.df = np.vectorize(output_activation.df)

    def activations(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(x)
        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.f(self.activations(x))

    def stepforward(self, x: np.ndarray) -> List[np.ndarray]:
        a = [x]
        for layer in self.layers:
            a.append(layer.forward(a[-1]))
        return a

    def back(self, x: np.ndarray, y: np.ndarray, eta: float) -> None:
        z = self.activations(x)
        a = self.stepforward(x)
        delta = self.df(z)*(y-a[-1])
        '''L-2,L-3,...,1'''
        for l in np.arange(len(self.layers)-2,0,-1):
            layer = self.layers[l]
            '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
            ndelta = layer.backerror(a[l],delta)
            layer.improve(eta, delta, a[l])
            delta = ndelta
        self.layers[0].improve(eta, delta, a[0])
