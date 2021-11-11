import random
import copy
from typing import Callable

from autograd import numpy as np
import matplotlib.pyplot as plt

from csnet.trainer import tune_neural_network
from csnet.loss import binary_cross_entropy
from csnet.metrics import accuracy
from csnet.data import load_breast_cancer_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_test_neural_net_classification(X: np.ndarray, Y: np.ndarray, epochs: int = 1000, batch_size: int = 16, lamb: float = 0):

    # Train and test (not evel - eval are being split in the trianing form the trianing set)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Split train set into train and eval
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.25)

    # Scale data
    scaler_input = StandardScaler()
    scaler_input.fit(X_train)

    X_train = scaler_input.transform(X_train)
    X_eval = scaler_input.transform(X_eval)
    X_test = scaler_input.transform(X_test)

    #learning_rates = [1.5, 1, 0.5, 0.1, 0.05]
    returns = tune_neural_network(X_train, y_train, X_eval, y_eval, epochs=epochs, loss_func = binary_cross_entropy, measure = accuracy, problem_type = 'Classification')

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(y_test, output))
    predictions = (output > 0.5).astype(int)
    test_acc = accuracy(y_test, predictions)

    print(f"Neural network final test on best model: BCE: {test_loss}, Accuracy: {test_acc}")

    # Plotting losses and R2
    plt.plot(returns['train_losses'], label = "Train loss")
    plt.plot(returns['eval_losses'], label = "Eval loss")
    plt.legend()
    plt.show()
    plt.plot(returns['train_measure'], label = "Train Acc")
    plt.plot(returns['eval_measure'], label = "Eval Acc")
    plt.legend()
    plt.show()

    from IPython import embed; embed()


if __name__ == '__main__':
    num_epochs = 100
    
    x, y = load_breast_cancer_data()
    
    train_and_test_neural_net_classification(x, y, epochs=num_epochs)