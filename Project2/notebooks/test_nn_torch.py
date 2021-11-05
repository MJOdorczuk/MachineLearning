import random

import matplotlib
import matplotlib.pyplot as plt
from autograd import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch

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

num_points = 100
num_epochs = 100
noise = 0
eta = 0.01

x = np.random.uniform(0, 1, num_points)
y = np.random.uniform(0, 1, num_points)
z = np.array([FrankeFunction(x, y, noise)]).T

X = np.stack([x, y], axis=1)
X_train, X_test, z_train, z_test = train_test_split(
    X,
    z,
    test_size=0.2,
    random_state=0,
)
X_train, X_val, z_train, z_val = train_test_split(
    X_train,
    z_train,
    test_size=0.2,
    random_state=0,
)
X_train -= np.mean(X_train)
X_val -= np.mean(X_val)
X_test -= np.mean(X_test)

z_train -= np.mean(z_train)
z_val -= np.mean(z_val)
z_test -= np.mean(z_test)

# Test a torch nn to compare with
model, loss, r2 = make_and_train_torch_nn(X_train, z_train, eta, num_epochs)

plt.title("Torch train loss")
plt.plot(loss)
plt.show()
plt.title("Torch train R2 score")
plt.plot(r2)
plt.show()

pred = model(torch.as_tensor(X_test, dtype=torch.float)).detach().numpy()
test_mse = mean_squared_error(z_test, pred)
test_r2 = r2_score(z_test, pred)

print(f"Test dataset performance: MSE={test_mse:.2f}, R2={test_r2:.2f}")



