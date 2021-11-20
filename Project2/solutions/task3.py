import random
import copy
from typing import Callable

from autograd import numpy as np
import matplotlib.pyplot as plt

from csnet.trainer import tune_neural_network, train_pytorch_net
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

    X = [X_train, X_eval, X_test]
    Z = [z_train, z_eval, z_test]

    returns = tune_neural_network(X, Z, epochs=epochs, batch_size = batch_size)

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(z_test, output))
    test_r2 = r2_score(z_test, output)

    print(f"Neural network final test on best model: MSE: {test_loss}, R2: {test_r2}")

    # Testing against Pytorch
    model, train_losses, train_measures, eval_losses, eval_measures, test_losses, test_measure = train_pytorch_net(best_model, X, Z, returns['lr'], epochs, batch_size)
    print(f"Torch Neural network final test on best model: MSE: {test_losses}, R2: {test_measure}")


    # Plotting losses and R2
    plt.plot(returns['train_losses'], label = "Train loss")
    plt.plot(returns['eval_losses'], label = "Eval loss")
    plt.plot(train_losses, label = "Torch Train loss")
    plt.plot(eval_losses, label = "Torch Eval loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/task_3_loss.pdf", dpi=100)
    plt.show()
    plt.plot(returns['train_measure'], label = "Train R2")
    plt.plot(returns['eval_measure'], label = "Eval R2")
    plt.plot(train_measures, label = "Torch Train R2")
    plt.plot(eval_measures, label = "Torch Eval R2")
    plt.ylim(-1,1)
    plt.xlabel("Epochs")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.savefig("figures/task_3_r2.pdf", dpi=100)
    plt.show()

    fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
    #X_test_sorted = np.sort(X_test, axis=0)
    X_test_sorted = np.arange(-0.5, 0.5, 0.01/2).reshape(100,2)
    X_plot, Y_plot = np.meshgrid(X_test_sorted[:,0], X_test_sorted[:,1])

    # Create input from X and Y corresponding to expected input
    data = np.column_stack([X_plot.ravel(), Y_plot.ravel()])
    # data -= np.mean(data)
    # Predict z values
    z_plot = best_model.forward(data)

    # rearrange `z` to have a shape corresponding to `X` and `Y`
    pred_2d = z_plot.reshape(X_plot.shape[0], Y_plot.shape[1])

    ax.set_title("Predictions neural network")
    surf = ax.plot_surface(
        X_plot, Y_plot, pred_2d, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.scatter(X_test[:, 0], X_test[:, 1], z_test, marker="o")

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.view_init(elev=20, azim=60)

    plt.savefig("figures/NN_surface_plot_final.pdf", dpi=150)
    plt.show()


    from IPython import embed; embed()


if __name__ == '__main__':
    num_points = 100
    num_epochs = 100
    noise = 0.001

    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)

    X = np.column_stack((x,y))
    Z = FrankeFunction(x, y, noise).reshape(-1,1)

    train_and_test_neural_net_regression(X, Z, epochs=num_epochs, batch_size = 5)
