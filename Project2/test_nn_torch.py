import random

import matplotlib
import matplotlib.pyplot as plt
from autograd import numpy as np
from sklearn.model_selection import train_test_split

from csnet.torch_nn import make_and_train_torch_nn

matplotlib.use("tkagg")
np.random.seed(0)
random.seed(0)


def FrankeFunction(x, y, sigma=0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma * noise


num_points = 1000
num_epochs = 100
noise = 0
eta = 0.01


xs = (np.random.uniform(0, 1, num_points))
ys = (np.random.uniform(0, 1, num_points))
zs = np.asmatrix(FrankeFunction(xs, ys, noise)).T  # Target

X = np.stack([xs, ys], axis=1)
X_train, X_test, z_train, z_test = train_test_split(X, zs, test_size=0.2, random_state=0)
X_train, X_val, z_train, z_val = train_test_split(X_train, z_train, test_size=0.2, random_state=0)

# Test a torch nn to compare with
model, loss, r2 = make_and_train_torch_nn(X_train, z_train, eta, num_epochs)

plt.plot(loss)
plt.show()
plt.plot(r2)
plt.show()
