from typing import Callable
import numpy as np
from csnet.activation import Activation
from csnet.optim import sgd, _random_mini_batch_generator
from csnet.metrics import accuracy
from sklearn.model_selection import train_test_split
from autograd import elementwise_grad
from autograd.numpy import exp
import autograd.numpy as anp
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LogisticRegression:

    def __init__(self, dims: int, optimizer: Callable, loss: Callable) -> None:
        """
        Args:
            dims: Number of features
            optimizer: A Gradient optimizer.
        """
        self.weights = np.random.randn(dims, 1)
        self.bias = np.zeros(1)
        self.optimizer = optimizer
        self.loss_func = loss
        self.activation = Activation().sigmoid
        self.scaler= StandardScaler()

    def _sgd_step(self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        lr: float,
        grad: Callable,
        lamb: int = 0,
    ) -> np.ndarray:
        """One optimization step using SGT."""
        return weights - lr * grad


    def forward(self, x):
        """
        Forward pass of logistic regression.

        Args:
            x: Input data
        
        Return:
            pred: Output of 1 hidden layer followed by a sigmoid activation.
        """

        y = np.dot(x, self.weights) + self.bias
        out = self.activation(y)
        return out

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prediction: Forward pass into a thresholding. 

        Args:
            x: Input data
            threshold: decision threshold

        Returns:
            A prediction for a binary classification problem based on a threshold.
        """

        y = self.forward(x)
        return (y>threshold).astype('int')

    def validate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, lamb: float = 0):
        losses = []
        accuracies = []
        for sub_x, sub_y in _random_mini_batch_generator(x, y, batch_size = batch_size):
            if(sub_x.shape[0]==0):
                continue
            output = self.forward(sub_x)
            predictions = self.predict(sub_x)
            loss = self.loss_func(sub_y, output, lamb, self.weights)
            losses.append(loss)

            acc = accuracy(sub_y, predictions)
            accuracies.append(acc)
        
        return np.mean(losses), np.mean(accuracies)

    
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.01, lamb: float = 0.0, batch_size: int = 1):
        """
        #TODO
        Add aggregation of loss and acc
        """
        
        self.weights = self.optimizer(x, 
                                        y, 
                                        self.weights, 
                                        self.loss_func, 
                                        lamb,
                                        batch_size,
                                        epochs,
                                        lr,
                                        momentum = True)
        self.bias = self.optimizer(x, 
                                        y, 
                                        self.bias, 
                                        self.loss_func, 
                                        lamb,
                                        batch_size,
                                        epochs,
                                        lr,
                                        momentum = True)

    def loss(self, x, weights:np.ndarray, bias:np.ndarray, y: np.ndarray, lamb) -> int:
        """
        BCE loss for classification.

        Args:
            yhat: Predictions
            y: True labels

        Returns:
            The average binary cross entropy for each sample.
        """

        a = anp.dot(x, weights) + bias
        yhat = 1.0 / (1.0+exp(-a))
        yhat = yhat.ravel()
        n = y.shape[0]

        return - 1/n * anp.sum(y * anp.log(yhat) + (1-y) * anp.log(1 - yhat)) + (lamb / n) * np.sum(np.square(weights))

    def single_step(self, x: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.01, lamb: float = 0.0, batch_size: int = 1):
        """
        #TODO
        Add aggregation of loss and acc
        """
        if x.shape[0] == 0:
            return
        grad_cost_function_w = elementwise_grad(self.loss, 1)
        grad_cost_function_b = elementwise_grad(self.loss, 2)
        gradient_w = grad_cost_function_w(x, self.weights, self.bias, y, lamb)
        gradient_b = grad_cost_function_b(x, self.weights, self.bias, y, lamb)

        self.weights = self._sgd_step(x, 
                                        y, 
                                        self.weights, 
                                        lr,
                                        gradient_w,
                                        lamb = lamb)
        self.bias = self._sgd_step(x, 
                                        y, 
                                        self.bias, 
                                        lr,
                                        gradient_b,
                                        lamb = lamb)
        

    def train_model(self, x: np.ndarray, y: np.ndarray, epochs: int = 50, lr:float = 0.01, lamb: float = 0.0, batch_size: int = 16, eval_size:float = 0.25):
        
        # Train, eval set
        X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=eval_size)
        

        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_eval = self.scaler.transform(X_eval)

        best_epoch = 0
        best_loss = np.inf
        best_acc = 0
        best_model = (self.weights, self.bias)

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(epochs):
            
            # Single step
            self.single_step(X_train, y_train, epochs = 1, lr =lr, lamb = lamb, batch_size=batch_size)
            
            train_loss, train_acc = self.validate(X_train, y_train, batch_size,lamb)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            test_loss, test_acc = self.validate(X_eval, y_eval, batch_size, lamb)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            # If new best:
            if best_acc < test_acc:
                best_epoch = epoch
                best_loss = test_loss
                best_acc = test_acc
                best_model = (self.weights, self.bias)
        
        return best_model, best_loss, best_epoch, best_acc, train_losses, train_accuracies, test_losses, test_accuracies
        

