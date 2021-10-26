import autograd.numpy as np 
import numpy
from sklearn.datasets import load_breast_cancer
from autograd import elementwise_grad as egrad
from autograd import grad
from csnet.logistic_regression import LogisticRegression
from csnet.loss import binary_cross_entropy
from csnet.optim import sgd

import matplotlib.pyplot as plt

if __name__ == "__main__":


    data = load_breast_cancer()
    x = data.data
    y = data.target
    
    log_reg = LogisticRegression(x.shape[1], sgd, binary_cross_entropy)
    best_model, best_epoch, best_acc, train_losses, train_accuracies, test_losses, test_accuracies = log_reg.train_model(x,y,100, 0.1, 0, 1, 0.2)
    plt.plot(train_losses, label = "Train loss")
    plt.plot(test_losses, label = "Test loss")
    plt.legend()
    plt.show()
    plt.plot(train_accuracies, label = "Train acc")
    plt.plot(test_accuracies, label = "Test acc")
    plt.legend()
    plt.show()
    #from IPython import embed; embed()