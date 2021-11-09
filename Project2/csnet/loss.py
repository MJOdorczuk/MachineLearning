import autograd.numpy as np 

def binary_cross_entropy(yhat: np.ndarray, y: np.ndarray, lamb: float = 0, weights: np.ndarray = 0) -> np.ndarray:
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

    return - 1/n * np.sum(y * np.log(yhat) + (1-y) * np.log(1 - yhat)) + (lamb / n) * np.sum(np.square(weights))

def mean_squared_error(y: np.ndarray, yhat:np.ndarray):
    return np.mean(np.square(y - yhat), axis=0) 