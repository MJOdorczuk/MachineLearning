import random
import copy
from typing import Callable

from autograd import numpy as np
import matplotlib.pyplot as plt

from csnet.trainer import tune_neural_network
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def FrankeFunction(x, y, sigma = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma*noise


def train_and_test_neural_net_regression(X: np.ndarray, Z: np.ndarray, epochs: int = 1000, batch_size: int = 16, lamb: float = 0):

    # Train and test (not evel - eval are being split in the trianing form the trianing set)
    X_train, X_test, z_train, z_test = train_test_split(X, Z, test_size=0.2)
    # Split train set into train and eval
    X_train, X_eval, z_train, z_eval = train_test_split(X_train, z_train, test_size=0.25)

    # Scale data by subtracting mean
    scaler_input = StandardScaler(with_std = False)
    scaler_input.fit(X_train)
    scaler_output = StandardScaler(with_std = False)
    scaler_output.fit(z_train)

    X_train = scaler_input.transform(X_train)
    X_eval = scaler_input.transform(X_eval)
    X_test = scaler_input.transform(X_test)

    z_train = scaler_output.transform(z_train)
    z_eval = scaler_output.transform(z_eval)
    z_test = scaler_output.transform(z_test)

    
    returns = tune_neural_network(X_train, z_train, X_eval, z_eval, epochs=epochs)

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(z_test, output))
    test_r2 = r2_score(z_test, output)

    print(f"Neural network final test on best model: MSE: {test_loss}, R2: {test_r2}")

    # Plotting losses and R2
    plt.plot(returns['train_losses'], label = "Train loss")
    plt.plot(returns['eval_losses'], label = "Eval loss")
    plt.legend()
    plt.show()
    plt.plot(returns['train_measure'], label = "Train R2")
    plt.plot(returns['eval_measure'], label = "Eval R2")
    plt.legend()
    plt.show()

    from IPython import embed; embed()


if __name__ == '__main__':
    num_points = 100
    num_epochs = 500
    noise = 0.001
    
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)

    X = np.column_stack((x,y))
    Z = FrankeFunction(x, y, noise).reshape(-1,1)
    
    train_and_test_neural_net_regression(X, Z, epochs=num_epochs)