from autograd import numpy as np
from typing import Callable
from typing import List
from autograd import grad, elementwise_grad
from sklearn.metrics import mean_squared_error, r2_score



'''Example, to be removed when real optimizer developed.
some_counter is just an example argument showing, how the evolving of the optimizer should proceed'''
def SGD(w,b,delta,a,eta,some_counter):
    return (w-np.dot(np.asmatrix(delta), a).T * eta / delta.shape[1],
        b-np.dot(delta, np.ones(delta.shape[1]) / delta.shape[1])*eta,
        lambda w,b,delta,a,eta: SGD(w,b,delta,a,eta,some_counter+1))

def initSGD(w,b,delta,a,eta):
    return SGD(w,b,delta,a,eta,0)

def mse(x, y):
    return np.mean(np.square(x-y))

def meanSquaresGrad(x, y):
    return (y-x)

# type SF =
#     | A of (float -> float * SF)
#
# let rec f (x: int) (c: float) : float * SF=
#     c, A (f (x+1))
#
# That's how you type things like opt in OCaml, do you know, how to type it here?


class layer:
    def __init__(self, activation_function: Callable[[float],float], input_size: int, output_size: int, opt = initSGD) -> None:
        self.activation = activation_function
        self.d_activation = elementwise_grad(activation_function, 0)
        self.weights = np.random.normal(size=(input_size,output_size))
        self.bias = np.zeros((output_size,))
        self.opt = opt

    def pre_activation(self, input: np.ndarray) -> np.ndarray:
        return np.dot(input,self.weights) + self.bias

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.activation(self.pre_activation(input))

    '''delta_j^l=sum_k delta_k^{l+1} w_{kj}^{l+1}f'(z_j^l)'''
    def back_prop_error(self, delta_l_plus_1: np.ndarray, layer_l_plus_1: np.ndarray, df_dz: np.ndarray) -> np.ndarray:

        return df_dz * np.dot(delta_l_plus_1, layer_l_plus_1.weights)

    def update_weights(self, eta: float, delta: np.ndarray, activation: np.ndarray):
        self.weights = self.weights - eta * np.dot(delta.T, activation)
        self.bias = self.bias - eta*np.dot(delta.T, np.ones((delta.shape[0], 1)))


class nn:
    def __init__(self,
    layers: List[layer],
    # I know that it can depend on different traits but let us limit ourselves to just the difference for now
    cost: Callable[[np.ndarray],float],
    opt= initSGD) -> None:
        self.layers = layers
        self.opt = opt
        self.cost = mse
        self.i = 0

        self.activation = []
        self.pre_activation = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            self.pre_activation.append(y)
            y = layer.forward(y)
            self.activation.append(y)
        return y


    def back(self, x: np.ndarray, y: np.ndarray, eta: float) -> None:
        #a, z = self.stepforward(x)

        a = self.activation
        z = self.pre_activation
        '''
        The cost function can depend on both y and y_tilde separately and not only on their difference
        thence we have to compute grad for each iteration :(
        '''
        dc_da = elementwise_grad(self.cost, 0)

        # Make output layer instead of layer[-1]
        delta = self.layers[-1].d_activation(z[-1]) * dc_da(a[-1], y)
        '''L-1,L-3,...,1'''
        for l in range(len(self.layers)-1,0,-1):
            layer = self.layers[l]
            df_dz = elementwise_grad(layer.activation, 0)(self.pre_activation[l])
            hidden_delta = layer.back_prop_error(delta, self.layers[l], df_dz)

            layer.update_weights(eta, hidden_delta, a[l])
            delta = hidden_delta

        self.layers[0].update_weights(eta, delta, x)

    def error(self, x: np.ndarray, y: np.ndarray):
        pred = self.forward(x)
        print(r2_score(pred, y))
        return self.cost(y, pred)

