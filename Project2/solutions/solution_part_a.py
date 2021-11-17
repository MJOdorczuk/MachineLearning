from __future__ import annotations

import json
import os
import os.path
from typing import Any, cast

import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from csnet.regression import METHODS, Regression
from csnet.utils import FrankeFunction, create_X, regression_grid_search

matplotlib.use("tkagg")

numpy.random.seed(3190)

sns.set(font_scale=2)

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
    results_pr_method: dict[Any, Any] = {}

    for complexity in range(1, 11):
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
        list_of_methods = ["ols", "ols_sgd", "ridge", "ridge_sgd"]
        all_results["mse"][complexity] = {}
        all_results["r2"][complexity] = {}
        results_pr_method[complexity] = {}
        for method in list_of_methods:
            results_pr_method[complexity][method] = []
            reg = Regression(method=cast(METHODS, method))

            # grid search is time consuming, so a small epoch and selected
            # parameters is chosen.
            results = regression_grid_search(
                reg,
                X_train,
                z_train,
                n_epochs=30,
                lrs=np.logspace(-5, -1, 5).tolist(),
                batchsize=[2, 4, 8],
                lambdas=np.logspace(-4, -1, 5).tolist(),
                momentum=True,
                alpha=np.logspace(-8, 0, 10).tolist()
            )

            # make a model from best results
            best = results[0]
            reg = Regression(method=cast(METHODS, method))
            reg.fit(
                X_train,
                z_train,
                lamb=best.lamb,
                n_epochs=100,
                display=False,
                batch_size=best.batchsize,
                momentum=best.momentum,
                alpha=best.alpha,
            )
            z_pred = reg.predict(X_test)

            mse = mean_squared_error(z_test, z_pred)
            r2 = r2_score(z_test, z_pred)
            print(f"---- Results from {method} on testdata - MSE={mse} , R2={r2}")
            # Print top 3 results
            # print(json.dumps(results[0:3], indent=4))

            # Save results for plotting later
            all_results["mse"][complexity][method] = mse
            all_results["r2"][complexity][method] = r2
            results_pr_method[complexity][method] = results

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
    df_mse.plot(
        figsize=(16, 9),
        xlabel="Complexity",
        ylabel="$MSE$",
    )
    plt.savefig(os.path.join(path, "part_a_mse.pdf"), dpi=150)
    plt.show()
    df_r2.plot(
        figsize=(16, 9),
        xlabel="Complexity",
        ylabel="$R^2$",
        ylim=(-0.1, 1.0)
    )
    plt.savefig(os.path.join(path, "part_a_r2.pdf"), dpi=150)
    plt.show()

    for c, dct in results_pr_method.items():
        results = dct["ridge_sgd"]
        lrs = {[res.lr for res in results].sort()}
        lambs = {[res.lamb for res in results].sort()}
        heatmap = pandas.DataFrame()

        for res in results:
            heatmap.at[res.lr, res.lamb] = res.mse

        heatmap = heatmap.dropna()
        print(heatmap)

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle(f"Complexity {c}")
        ax = sns.heatmap(heatmap)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.legend()
        plt.savefig(os.path.join(path, f"heatmap_ridge_comp_{c}.pdf"), dpi=150)
        plt.show()
        plt.close()


if __name__ == "__main__":
    part_a()
