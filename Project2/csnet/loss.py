import autograd.numpy as np 

def binary_cross_entropy(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    BCE loss for classification.

    Args:
        yhat: Predictions
        y: True labels

    Returns:
        The average binary cross entropy for each sample.
    """

    n = y.shape[0]
    yhat = yhat.ravel()

    return - 1/n * np.sum(y * np.log(yhat) + (1-y) * np.log(1 - yhat))