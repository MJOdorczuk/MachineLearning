from sklearn.datasets import load_breast_cancer


def load_breast_cancer_data():
    """
    Returns design matrix and labels of breast cancer data
    """
    data = load_breast_cancer()

    return data.data, data.target