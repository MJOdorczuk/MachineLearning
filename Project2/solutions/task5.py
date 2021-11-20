from csnet.logistic_regression import LogisticRegression
from csnet.loss import binary_cross_entropy
from csnet.optim import SGD
from csnet.data import load_breast_cancer_data
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import seaborn as sns


import matplotlib.pyplot as plt

def tune_log_reg(x: np.ndarray, y:np.ndarray, epochs:int=100, batch_size:int = 16, lamb: float = 0, lambdas = np.logspace(-4,2, 10), lrs = [1, 0.5, 0.1, 0.05, 0.01], plot_heatmap=False):
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

    X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2)

    X = [X_train, X_eval]
    Y = [y_train, y_eval]

    heatmap_loss = {}
    heatmap_acc = {}
    for lamb in lambdas:
        heatmap_loss[str(lamb)] = {}
        heatmap_acc[str(lamb)] = {}
        for lr in lrs:
            sgd = SGD(lr, use_momentum=True)
            log_reg = LogisticRegression(x.shape[1], sgd, binary_cross_entropy)
            # Best based on validation set
            best_model, best_loss, best_epoch, best_acc, train_losses, train_accuracies, test_losses, test_accuracies = log_reg.train_model(X, Y,epochs, lr, lamb, batch_size, 0.25)
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
            print(f"Lr: {lr}, lamb: {lamb}, current_acc: {best_acc}, current_loss_ {best_loss}, best acc: {global_best_acc}, best loss: {global_best_loss} with lr {global_best_lr} and lamb {global_best_lamb}")
            heatmap_loss[str(lamb)][str(lr)] = best_loss
            heatmap_acc[str(lamb)][str(lr)] = best_acc
    if plot_heatmap:
        plot_lr_lamb_heatmap(heatmap_loss, heatmap_acc)

    return global_best_model, global_best_model_object, global_best_train_losses, global_best_test_losses, global_best_train_acc, global_best_test_acc

def plot_lr_lamb_heatmap(losses, accuracies):
    sns.set(font_scale=2)

    losses_df = pd.DataFrame()
    for lamb_key, lamb_dict in losses.items():
        for lr_key, lr_value in lamb_dict.items():
            losses_df.at[round(float(lamb_key),4), round(float(lr_key),4)] = lr_value
    losses_df = losses_df.dropna()

    acc_df = pd.DataFrame()
    for lamb_key, lamb_dict in accuracies.items():
        for lr_key, lr_value in lamb_dict.items():
            acc_df.at[round(float(lamb_key),4), round(float(lr_key), 4)] = lr_value
    acc_df = acc_df.dropna()

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.suptitle(f"Loss")

    ax = sns.heatmap(
            losses_df,
            ax=ax,
            square=False,
            xticklabels=True,
            yticklabels=True,
        )
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$",labelpad=0)
    plt.yticks(rotation=45, fontsize=28)
    plt.xticks(rotation=20, fontsize=28)
    plt.savefig('figures/logreg_loss_heatmap.pdf', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.suptitle(f"Accuracies")
    ax = sns.heatmap(
            acc_df,
            ax=ax,
            square=False,
            xticklabels=True,
            yticklabels=True,
        )
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$", labelpad=0)
    plt.yticks(rotation=45, fontsize=28)
    plt.xticks(rotation=20, fontsize=28)
    plt.savefig("figures/logreg_acc_heatmap.pdf", dpi=150)
    plt.show()


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
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/task_5_loss.pdf", dpi=100)
    plt.show()
    plt.plot(global_best_train_acc, label = "Train acc")
    plt.plot(global_best_test_acc, label = "Test acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figures/task_5_acc.pdf", dpi=100)
    plt.show()

if __name__ == "__main__":

    x, y = load_breast_cancer_data()

    train_and_test_log_reg(x,y)


