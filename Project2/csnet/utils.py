import autograd.numpy as np
import numpy


def FrankeFunction(x, y, sigma=0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    noise = numpy.random.normal(0, 1, x.shape[0])
    noise = noise.reshape(x.shape[0], 1)

    return (term1 + term2 + term3 + term4).reshape(-1, 1) + sigma * noise


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l_beta = int((n + 1) * (n + 2) / 2)   # Number of elements in beta
    X = numpy.ones((N, l_beta))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)

    return X


def cost_mse(beta, X, lamb, y_true):
    return np.mean((y_true - np.dot(X, beta))**2)


def cost_mse_ridge(beta, X, lamb, y_true):
    assert lamb is not None
    return np.mean((y_true - np.dot(X, beta))**2) + lamb * np.dot(beta.T, beta)
