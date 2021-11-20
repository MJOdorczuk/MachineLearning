import random
import copy
from typing import Callable

from autograd import numpy as np
import matplotlib.pyplot as plt

from csnet.trainer import tune_neural_network, train_pytorch_net
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

    X = [X_train, X_eval, X_test]
    y = [y_train, y_eval, y_test]

    #learning_rates = [1.5, 1, 0.5, 0.1, 0.05]
    returns = tune_neural_network(X, y, epochs=epochs, loss_func = binary_cross_entropy, measure = accuracy, problem_type = 'Classification')

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(y_test, output))
    predictions = (output > 0.5).astype(int)
    test_acc = accuracy(y_test, predictions)

    print(f"Neural network final test on best model: BCE: {test_loss}, Accuracy: {test_acc}")

    model, train_losses, train_measures, eval_losses, eval_measures, test_losses, test_measure = train_pytorch_net(best_model, X, y, returns['lr'], epochs, batch_size, measure = accuracy)
    print(f"Torch Neural network final test on best model: BCE: {test_losses}, Accuracy: {test_measure}")

    # Plotting losses and R2
    plt.plot(returns['train_losses'], label = "Train loss")
    plt.plot(returns['eval_losses'], label = "Eval loss")
    plt.plot(train_losses, label = 'Torch Train loss')
    plt.plot(eval_losses, label = "Torch Eval loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/task_4_loss.pdf", dpi=100)
    plt.show()
    plt.plot(returns['train_measure'], label = "Train Acc")
    plt.plot(returns['eval_measure'], label = "Eval Acc")
    plt.plot(train_measures, label = "Torch Train Acc")
    plt.plot(eval_measures, label = 'Torch eval Acc')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figures/task_4_acc.pdf", dpi=100)
    plt.show()

    from IPython import embed; embed()


if __name__ == '__main__':
    num_epochs = 100

    x, y = load_breast_cancer_data()

    train_and_test_neural_net_classification(x, y, epochs=num_epochs)
