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

numpy.random.seed(41)

sns.set(font_scale=2.5)

def part_a() -> None:
    """Part a) compare regression with use of SGD to `pinv`."""
    # path for saving figures
    path = os.path.join(os.path.dirname(__file__), "figures")
    try:
        os.mkdir(path=path)
    except FileExistsError:
        pass

    EPOCHS = 100
    noise = 0.00
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
            X, z, test_size=0.2, random_state=41,
        )
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
                n_epochs=EPOCHS,
                lrs=[0.01, 0.1, 0.25, 0.5, 0.75],
                batchsize=[1, 2, 4],
                # lambdas=[0.001, 0.01, 0.1, 0.5, 0.9],
                lambdas=np.logspace(-3, 1, 5).tolist(),
                momentum=True,
                alpha=np.arange(0.1, 0.9, 0.4).tolist(),
            )

            # make a model from best results
            best = results[0]
            reg = Regression(method=cast(METHODS, method))
            reg.fit(
                X_train,
                z_train,
                lamb=best.lamb,
                n_epochs=150,
                display=False,
                batch_size=best.batchsize,
                momentum=best.momentum,
                alpha=best.alpha,
            )
            z_pred = reg.predict(X_test)

            mse = mean_squared_error(z_test, z_pred)
            r2 = r2_score(z_test, z_pred)
            print(f"---- Results from {method} on testdata - MSE={mse} , R2={r2}")
            print(f"------ Best val result: {best}")

            if method in ["ols_sgd", "ridge_sgd"]:
                fig, ax = plt.subplots(1, figsize=(16, 9))
                ax.set_title(
                    f"Training loss {method}, complexity {complexity}",
                )
                ax.set_ylabel("MSE")
                ax.set_xlabel("Epochs")
                ax.plot(
                    range(len(best.all_mse)),
                    best.all_mse,
                    label="Training loss",
                )
                plt.legend()
                plt.savefig(
                    os.path.join(
                        path, f"part_a_train_loss_{method}_c{complexity}.pdf"
                    ),
                    dpi=100,
                )
                plt.close()

            # Save results for plotting later
            all_results["mse"][complexity][method] = mse
            all_results["r2"][complexity][method] = r2
            results_pr_method[complexity][method] = results

    # Display data
    print(f"Saving figures at {path}")

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
    # plt.show()
    df_r2.plot(
        figsize=(16, 9),
        xlabel="Complexity",
        ylabel="$R^2$",
        ylim=(-0.1, 1.0)
    )
    plt.savefig(os.path.join(path, "part_a_r2.pdf"), dpi=150)
    # plt.show()

    for c, dct in results_pr_method.items():
        results = dct["ridge_sgd"]
        heatmap = pandas.DataFrame()
        heatmap_r2 = pandas.DataFrame()

        for res in results:
            heatmap.at[res.lr, res.lamb] = res.val_mse
            heatmap_r2.at[res.lr, res.lamb] = res.val_r2

        heatmap = heatmap.dropna()
        heatmap_r2 = heatmap_r2.dropna()
        print(heatmap)
        print(heatmap_r2)

        fig, ax = plt.subplots(2, figsize=(16, 20))
        # fig.suptitle(f"Complexity {c}")
        ax[0] = sns.heatmap(
            heatmap,
            ax=ax[0],
            square=False,
            xticklabels=True,
            yticklabels=True,
        )
        ax[0].set_title("MSE", pad=10)
        ax[0].set_ylabel("$\eta$", rotation=0, labelpad=35)
        ax[0].set_xlabel("$\lambda$", labelpad=5)
        ax[1] = sns.heatmap(
            heatmap_r2,
            ax=ax[1],
            square=False,
            xticklabels=True,
            yticklabels=True,
        )
        ax[1].set_title("$R^2$", pad=10)
        ax[1].set_ylabel("$\eta$", rotation=0, labelpad=35)
        ax[1].set_xlabel("$\lambda$", labelpad=40)
        fig.tight_layout(pad=60)
        plt.yticks(rotation=0, fontsize=28)
        plt.xticks(rotation=0, fontsize=28)
        plt.savefig(
            os.path.join(path, f"part_a_heatmap_ridge_comp_{c}.pdf"),
            dpi=100,
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    part_a()
