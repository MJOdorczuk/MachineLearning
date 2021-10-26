from __future__ import annotations

import os
import os.path
from typing import Any, cast

import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from csnet.regression import METHODS, Regression
from csnet.utils import FrankeFunction, create_X

matplotlib.use("tkagg")

numpy.random.seed(3190)


def part_a() -> None:
    """Part a) compare regression with use of SGD to `pinv`."""
    noise = 0.01
    num_points = 100
    x = (numpy.random.uniform(0, 1, num_points))
    y = (numpy.random.uniform(0, 1, num_points))
    z = FrankeFunction(x, y, noise)

    all_results: dict[str, Any] = {}
    all_results["mse"] = {}
    all_results["r2"] = {}

    for complexity in range(16):
        print(f" Complexity {complexity} ".center(80, "="))
        # Create design matrix
        X = create_X(x, y, n=complexity)
        # Split and scale data
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2,)
        X_train = X_train - np.mean(X_train)
        X_test = X_test - np.mean(X_test)
        z_train = z_train - np.mean(z_train)
        z_test = z_test - np.mean(z_test)

        # fit and calculate mse/r2 for each method
        list_of_methods = ["ols", "ols_sgd", "ridge"]  # , "ridge_sgd"]
        all_results["mse"][complexity] = {}
        all_results["r2"][complexity] = {}
        for method in list_of_methods:
            reg = Regression(method=cast(METHODS, method))
            reg.fit(X_train, z_train, lamb=1e-2, n_epochs=500, display=False)
            z_pred = reg.predict(X_test)
            mse = mean_squared_error(z_test, z_pred)
            r2 = r2_score(z_test, z_pred)
            print(f"----From {method} - Cost={mse} , r2={r2}")
            all_results["mse"][complexity][method] = mse
            all_results["r2"][complexity][method] = r2

    # Display data
    # import json
    # print(json.dumps(all_results, indent=4))
    # Generate plots and tables?
    path = os.path.join(os.path.dirname(__file__), "figures")
    try:
        os.mkdir(path=path)
    except FileExistsError:
        pass

    df_mse = pandas.DataFrame(all_results["mse"]).T
    df_r2 = pandas.DataFrame(all_results["r2"]).T
    print(df_mse)
    print(df_r2)
    df_mse.plot(figsize=(16, 9), xlabel="Complexity", ylabel="$MSE$")
    plt.savefig(os.path.join(path, "part-a-mse.png"), dpi=150)
    plt.show()
    df_r2.plot(
        figsize=(16, 9),
        xlabel="Complexity",
        ylabel="$R^2$",
        ylim=(-0.1, 1.0)
    )
    plt.savefig(os.path.join(path, "part-a-r2.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    part_a()
