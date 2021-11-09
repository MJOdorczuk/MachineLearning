
from csnet.logistic_regression import LogisticRegression
from csnet.loss import binary_cross_entropy
from csnet.optim import sgd
from csnet.data import load_breast_cancer_data
from sklearn.model_selection import train_test_split


import numpy as np

import matplotlib.pyplot as plt

def tune_log_reg(x: np.ndarray, y:np.ndarray, epochs:int=100, batch_size:int = 16, lamb: float = 0):
    global_best_model = None
    global_best_acc = 0
    global_best_loss = np.inf
    global_best_lr = 0
    global_best_model_object = None
    global_best_lamb = 0

    # These are for plotting purposes
    global_best_train_losses = None
    global_best_test_losses = None
    global_best_train_acc = None
    global_best_test_acc = None
    
    for lamb in np.logspace(-4,2, 10):
        for lr in [1, 0.5, 0.1, 0.05, 0.01]:
            log_reg = LogisticRegression(x.shape[1], sgd, binary_cross_entropy)
            # Best based on validation set
            best_model, best_loss, best_epoch, best_acc, train_losses, train_accuracies, test_losses, test_accuracies = log_reg.train_model(x,y,epochs, lr, lamb, batch_size, 0.25)
            if global_best_acc <= best_acc:
                if best_loss < global_best_loss:
                    global_best_acc = best_acc
                    global_best_loss = best_loss
                    global_best_model = best_model
                    global_best_lr = lr
                    global_best_model_object = log_reg
                    global_best_lamb = lamb

                    global_best_train_losses = train_losses
                    global_best_test_losses = test_losses
                    global_best_train_acc = train_accuracies
                    global_best_test_acc = test_accuracies
            print(f"Lr: {lr}, lamb: {lamb}, current_acc: {best_acc}, best acc: {global_best_acc} with lr {global_best_lr} and lamb {global_best_lamb}")

    return global_best_model, global_best_model_object, global_best_train_losses, global_best_test_losses, global_best_train_acc, global_best_test_acc

def train_and_test_log_reg(x: np.ndarray, y: np.ndarray, epochs:int=100, batch_size:int = 16, lamb: float = 0):
    """
    """
    # Train and test (not evel - eval are being split in the trianing form the trianing set)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Tune the model
    best_log_reg_model, log_reg_object, global_best_train_losses, global_best_test_losses, global_best_train_acc, global_best_test_acc = tune_log_reg(X_train, y_train, epochs=epochs, batch_size=batch_size, lamb=lamb)
    
    # Initialize the best model, based on best weights
    log_reg_object.weights = best_log_reg_model[0]
    log_reg_object.bias = best_log_reg_model[1]

    # Final test
    X_test = log_reg_object.scaler.transform(X_test)
    final_loss, final_acc = log_reg_object.validate(X_test, y_test, batch_size=batch_size, lamb=lamb)
    print(f"Logistic Regression final test on best model: Loss: {final_loss}, Accuracy: {final_acc}")

    # Plotting losses and accuracies
    plt.plot(global_best_train_losses, label = "Train loss")
    plt.plot(global_best_test_losses, label = "Test loss")
    plt.legend()
    plt.show()
    plt.plot(global_best_train_acc, label = "Train acc")
    plt.plot(global_best_test_acc, label = "Test acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    x, y = load_breast_cancer_data()

    train_and_test_log_reg(x,y)

    