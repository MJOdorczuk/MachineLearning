import random
import copy
from typing import Callable

from autograd import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from csnet.trainer import tune_neural_network, train_pytorch_net
from csnet.metrics import accuracy

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def FrankeFunction(x, y, sigma = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma*noise

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)   # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def least_square(x_value,y_value, *args, **kwargs):
    # Using pinv
    return np.linalg.pinv(x_value.transpose().dot(x_value)).dot(x_value.transpose().dot(y_value))

def Ridge(x_value,y_value, lamb):
    return np.linalg.pinv(x_value.T.dot(x_value) + lamb*np.identity(x_value.shape[1])).dot(x_value.T).dot(y_value)

def simple_mse_and_r2_by_complexity(
    reg_func = least_square,
    x = None,
    y = None,
    z = None,
    num_points = 1000,
    complexity = 15,
    noise = 0.1,
    scale = True,
    lamb = 0,
    return_losses = False,
    scale_with_std = False,
    ):
    """
    Computes the simples ordinary least square based on the Franke Function
    """

    if x is None:
        x = np.random.uniform(0, 1, num_points)
        y =  np.random.uniform(0, 1, num_points)
    if z is None:
        z = FrankeFunction(x, y, noise).reshape(-1,1) # Target

    MSE_train = []
    MSE_pred = []
    r2_train = []
    r2_pred = []

    all_ols_betas = []
    all_xtx_inv = []

    for complexity in range(1,complexity+1):

        #Trying not to sort the x and y's
        X = create_X(x, y, n=complexity)  # Data

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        scaler_in = StandardScaler(with_std=scale_with_std)
        scaler_in.fit(X_train)
        scale_z = StandardScaler(with_std=scale_with_std)
        scale_z.fit(z_train)

        if scale:
            X_train = scaler_in.transform(X_train)
            X_test = scaler_in.transform(X_test)

            z_train = scale_z.transform(z_train)
            z_test = scale_z.transform(z_test)


        beta_opt = reg_func(X_train, z_train, lamb)
        all_ols_betas.append(beta_opt)

        xtx = np.linalg.pinv(X_train.transpose().dot(X_train))
        all_xtx_inv.append(xtx)

        z_tilde = X_train.dot(beta_opt)
        z_pred = X_test.dot(beta_opt)


        mse_train = mean_squared_error(z_tilde, z_train)
        MSE_train.append(mse_train)
        mse_test = mean_squared_error(z_pred, z_test)
        MSE_pred.append(mse_test)

        r2_train.append(r2_score(z_tilde, z_train))
        r2_pred.append(r2_score(z_pred, z_test))


    return MSE_pred, r2_pred

def train_and_test_neural_net_regression(
        X: np.ndarray, Z: np.ndarray, epochs: int = 100, batch_size: int = 16,
    ):

    X_train, X_eval, X_test = X
    z_train, z_eval, z_test = Z

    returns = tune_neural_network(X,Z, epochs=epochs, batch_size = batch_size)

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(z_test, output))
    test_r2 = r2_score(z_test, output)

    print(f"Neural network final test on best model: MSE: {test_loss}, R2: {test_r2}")

    returns = {
        'model': best_model,
        'lr': returns['lr'],
        'lamb': returns['lamb'],
        'best_eval_mse': returns['best_Loss'],
        'best_eval_r2': returns['best_r2_score'],
        'test_loss': test_loss,
        'test_r2': test_r2,
        'train_losses': returns['train_losses'],
        'eval_losses': returns['eval_losses'],
        'train_measure': returns['train_measure'],
        'eval_measure': returns['eval_measure']
    }

    return returns

def tune_ridge(X, Z):
    best_mse = np.inf
    best_r2 = -np.inf
    best_lamb = None
    best_ridge_mse_run = []
    best_ridge_r2_run = []
    for lamb in np.logspace(-4,2, 10):
        ridge_mse_pred, ridge_r2_pred = simple_mse_and_r2_by_complexity(Ridge, x = X, y = Z, lamb = lamb)
        mse = ridge_mse_pred[np.argmin(ridge_mse_pred)]
        r2 = ridge_r2_pred[np.argmin(ridge_r2_pred)]

        if r2 > best_r2:
            best_mse = mse
            best_r2 = r2
            best_ridge_mse_run = ridge_mse_pred
            best_ridge_r2_run = ridge_r2_pred
            best_lamb = lamb

    return best_ridge_mse_run, best_ridge_r2_run, best_lamb


def compare_nn_franke():

    num_points = 100
    num_epochs = 100
    batch_size = 16
    noise = 0.001

    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)

    X = np.column_stack((x,y))
    Z = FrankeFunction(x, y, noise).reshape(-1,1)

    # From project 1
    ols_mse_pred, ols_r2_pred = simple_mse_and_r2_by_complexity(x = x, y = y)
    best_ols_complexity = np.argmax(ols_r2_pred) + 1
    ols_best_mse = ols_mse_pred[np.argmin(ols_r2_pred)]
    ols_best_r2_pred = ols_r2_pred[np.argmax(ols_mse_pred)]

    ridge_mse_pred, ridge_r2_pred, ridge_lamb = tune_ridge(x, y)
    best_ridge_complexit = np.argmax(ridge_r2_pred) + 1
    ridge_mest_mse = ridge_mse_pred[np.argmin(ridge_mse_pred)]
    ridge_best_r2 = ridge_r2_pred[np.argmax(ridge_r2_pred)]

    """
    Do something with the values above
    """

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

    X = [X_train, X_eval, X_test]
    Z = [z_train, z_eval, z_test]

    best_nn = train_and_test_neural_net_regression(X, Z, num_epochs, batch_size)

    lamb = best_nn['lamb']
    lr = best_nn['lr']

    # Testing against Pytorch
    (
        model,
        train_losses,
        train_measures,
        eval_losses,
        eval_measures,
        test_losses,
        test_measure
    ) = train_pytorch_net(
        best_nn['model'],
        X,
        Z,
        best_nn['lr'],
        num_epochs,
        batch_size,
        lamb,
    )

    # Plotting losses and R2
    plt.plot(best_nn['train_losses'], label = "Train loss")
    plt.plot(best_nn['eval_losses'], label = "Eval loss")
    plt.plot(train_losses, label = "Torch Train loss")
    plt.plot(eval_losses, label = "Torch Eval loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/task_2_loss.pdf", dpi=100)
    plt.show()
    plt.plot(best_nn['train_measure'], label = "Train R2")
    plt.plot(best_nn['eval_measure'], label = "Eval R2")
    plt.plot(train_measures, label = "Torch Train R2")
    plt.plot(eval_measures, label = "Torch Eval R2")
    plt.xlabel("Epochs")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.savefig("figures/task_2_r2.pdf", dpi=100)
    plt.show()


if __name__ == '__main__':
    compare_nn_franke()
