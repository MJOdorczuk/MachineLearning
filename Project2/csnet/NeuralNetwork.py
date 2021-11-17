from autograd import numpy as np
from typing import Callable
from typing import List
from autograd import elementwise_grad

from csnet.optim import SGD

class Layer:
    def __init__(self, activ: Callable[[float],float], input_size: int, output_size: int, opt: SGD) -> None:
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
        self.d_activation = elementwise_grad(activ)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(size=(input_size,output_size))
        self.bias = np.zeros((output_size,))
        self.opt = opt
        self.z = None
        self.a = None

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
        self.z = np.dot(input,self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def update_weight(self, weight_grad, bias_grad) -> None:
        """Update weights using the optimizer based on the learning rate and the backwards propagated error of this layer.

        Parameters
        ----------
        weight_grad:
            Gradients for the weights.
        bias_grad:
            Gradient for the bias

        """

        self.weights = self.opt.step(self.weights, weight_grad)
        self.bias = self.opt.step(self.bias, bias_grad, True)

class NeuralNetwork:
    def __init__(self, layers: List[Layer], cost: Callable[[np.ndarray],float]) -> None:
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
        self.d_cost = elementwise_grad(self.cost, 1)

        self.output_layer = self.layers[-1]

        self.prev_weights = [l.weights for l in self.layers]

        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Neural network output after feeding to the activation function.

        Parameters
        ----------
        x   :
            Input data

        Returns
        -------
        np.ndarray

        """
        self.x = x
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def reduce_lr(self):
        for l in self.layers:
            l.opt.lr /= 2/3

    def backward(self, y: np.ndarray, lamb: float = 0) -> None:
        """Back propagation implementation based on the last forward feed and the expected values:

        Parameters
        ----------
        y   :
            Expected output for last forward feed.
        lamb:
            Regularization factor.
        """        

        last_layer = self.layers[-1]

        # Calculate delta for last layer
        delta_output = last_layer.d_activation(last_layer.z) * self.d_cost(y,last_layer.a)
        deltas = [delta_output]

        # No hidden layers:
        if len(self.layers) == 1:
            weights_grad_output = np.matmul(self.x.T,delta_output)
            bias_grad_output = np.sum(delta_output, axis = 0)

            # Regularization
            if lamb > 0:
                weights_grad_output += lamb * self.layers[-1].weights

            # update_weights last layer:
            self.layers[-1].update_weight(weights_grad_output, bias_grad_output)

        else:
            # Gradients for last layer
            weights_grad_output = np.matmul(self.layers[-2].a.T,delta_output)
            bias_grad_output = np.sum(delta_output, axis = 0)

            # Regularization
            if lamb > 0:
                weights_grad_output += lamb * self.layers[-1].weights

            # Delta second last
            delta_second_last = np.matmul(deltas[-1], self.layers[-1].weights.T) * self.layers[-2].d_activation(self.layers[-2].z)
            deltas.append(delta_second_last)

            # update_weights last layer:
            self.layers[-1].update_weight(weights_grad_output, bias_grad_output)

            for l in np.arange(len(self.layers)-2,0,-1):
                # Gradient of the current layer
                weights_grad = np.matmul(self.layers[l-1].a.T, deltas[-1])
                bias_grad = np.sum(deltas[-1], axis = 0)
                # Regularization
                if lamb > 0:
                    weights_grad += lamb * self.layers[l].weights
                
                # Calculate next delta term
                next_delta = np.matmul(deltas[-1], self.layers[l].weights.T) * self.layers[l-1].d_activation(self.layers[l-1].z)
                deltas.append(next_delta)

                # Update weights
                self.layers[l].update_weight(weights_grad, bias_grad)

            # First layer:
            weights_grad = np.matmul(self.x.T, deltas[-1])
            if lamb > 0:
                weights_grad_output += lamb * self.layers[-1].weights
            bias_grad = np.sum(deltas[-1], axis = 0)
            self.layers[0].update_weight(weights_grad, bias_grad)

            
        
        