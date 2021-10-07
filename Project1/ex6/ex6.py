from __future__ import annotations
from typing import Any
import json

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("tkagg")

np.random.seed(10)

### Functions from notebook:

def create_X(x, y, n ):
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
    return np.linalg.pinv(
        x_value.transpose().dot(x_value)
    ).dot(x_value.transpose().dot(y_value))


def Ridge(x_value,y_value, lamb):
    return np.linalg.pinv(
        x_value.T.dot(x_value) + lamb*np.identity(x_value.shape[1])
    ).dot(x_value.T).dot(y_value)


def Lasso(x,y, lamb):
    """
    A hybrid implementation of SKlearn and our custom code.
    We only use the coeficients from SKlearn, so that we can reuse our own code.
    """
    lasso = linear_model.Lasso(
        fit_intercept = False,
        alpha = lamb,
        max_iter = 1000,
        tol = 0.001,
        copy_X = True,
        )
    lasso.fit(x,y)
    return lasso.coef_

##### Exercise 6 specific code:

def load_data(display: bool = False, step: int = 1) -> tuple[np.ndarray, ...]:
    """Load terrain dataset from image and create x and y coordinates.

    Returns
    -------
    x, y, z
    """
    import os.path

    image_path = os.path.join(
        os.path.dirname(__file__),
        "SRTM_data_Norway_1.tif",
    )
    data = imageio.imread(image_path)

    if display:
        plt.imshow(data, plt.cm.gray)
        plt.axis("off")
        plt.show()

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    x, y = np.meshgrid(x, y)

    return x[::step, ::step], y[::step, ::step], data[::step, ::step]


def ex6(
    complexity: int = 10,
    n_folds = 10,
    step: int = 1,
    n_lambdas: int = 40,
) -> dict[str, Any]:
    """Evaluate what model is best for a given complexity and number of folds.
    """
    # Load terrain data
    x, y, z = load_data(display=False, step=step)

    # Create design matrix
    X = create_X(x, y, n=complexity)

    # Splitt data
    X_train, X_test, z_train, z_test = train_test_split(
            X,
            z.ravel(),
            test_size=0.3,
            random_state=10,
    )

    # Scaling features using standardization
    X_train = (X_train - X_train.mean()) / X_train.std()
    z_train = (z_train - z_train.mean()) / z_train.std()

    X_test = (X_test - X_test.mean()) / X_test.std()
    z_test = (z_test - z_test.mean()) / X_test.std()

    # Dict to save best model data
    model_data = {}

    # For each model fit and calculate MSE
    for reg in [least_square, Ridge, Lasso]:
        print(f"Testing {reg.__name__}...")

        best_mse: float | None = None
        best_lambda: float | None = None

        kfold = KFold(n_splits = n_folds)

        if reg.__name__ != "least_square":
            lambda_values = np.logspace(-4, 4, n_lambdas)

            # Grid search for best lambda for Ridge and Lasso
            for l in lambda_values:
                # for each lambda calculate MSE with CV
                mse_cv_list = np.zeros(n_folds)
                for i, (train_idx, test_idx) in enumerate(kfold.split(X_train)):
                    beta = reg(X_train[train_idx], z_train[train_idx], l)

                    z_pred = X_train[test_idx].dot(beta)

                    mse_cv_list[i] = mean_squared_error(z_pred, z_train[test_idx])

                mse = mse_cv_list.mean()

                if best_mse is None or mse < best_mse:
                    best_mse = mse
                    best_lambda = l
        else:
            # OLS
            mse_cv_list = np.zeros(n_folds)
            for i, (train_idx, test_idx) in enumerate(kfold.split(X_train)):
                beta = reg(X_train[train_idx], z_train[train_idx])

                z_pred = X_train[test_idx].dot(beta)

                mse_cv_list[i] = mean_squared_error(z_pred, z_train[test_idx])

            mse = mse_cv_list.mean()

            if best_mse is None or mse < best_mse:
                best_mse = mse

        model_data[reg.__name__] = {
            "best_mse": best_mse,
            "best_lambda": best_lambda,
        }
        #print(json.dumps(model_data, indent=4))

    print("\n\nTesting done. \nStats:\n")
    print(json.dumps(model_data, indent=4))

    return model_data


if __name__ == "__main__":
    import pandas as pd

    data_pr_complexity = {}
    for c in range(1, 15+1):
        print(f"Calculating for complexity {c}")
        data_pr_complexity[c] = ex6(complexity=c, n_folds=10, step=5, n_lambdas=50)
    print(json.dumps(data_pr_complexity, indent=4))

    df = pd.concat({k: pd.DataFrame(v) for k, v in data_pr_complexity.items()})
    print(df)
