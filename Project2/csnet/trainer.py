from __future__ import annotations

from itertools import permutations
import copy
from typing import Any, Callable

from autograd import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from csnet.NeuralNetwork import Layer, NeuralNetwork
from csnet.activation import Activation
from csnet.loss import mean_squared_error
from csnet.optim import random_mini_batch_generator, SGD

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def tune_neural_network(
        X: np.ndarray,
        z: np.ndarray,
        epochs:int = 100,
        batch_size:int = 16,
        lamb: float = 0,
        learning_rates=np.logspace(-3, -0.5, 6),
        lambdas=np.logspace(-3, -0.5, 5),
        measure: Callable = r2_score,
        loss_func: Callable = mean_squared_error,
        problem_type: str = 'Regression',
) -> dict[str, Any]:
    """Perform grid search of hyperparameters fo NN.

    """
    X_train, X_eval, X_test = X
    Z_train, Z_eval, Z_test = z

    global_best_model = None
    global_best_loss = np.inf
    global_best_measure = -np.inf
    global_best_lr = 0
    global_best_model_object = None
    global_best_lamb = 0
    global_best_neuron_combo = None
    global_best_activation = None

    global_best_train_losses = None
    global_best_eval_losses = None
    global_best_train_measure = None
    global_best_eval_measure = None

    activations = Activation()
    for lamb in lambdas:
        print(f"New Lambda {lamb}")
        for lr in learning_rates:
            print(f"New Learning rate: {lr}")
            for num_layers in range(1, 4):
                for activation in activations.get_all_activations():
                    all_num_neurons_combinations = permutations([1,2,3,4,5], num_layers)
                    for neuron_combination in all_num_neurons_combinations:

                        # Construct Neural network
                        input_size = X_train.shape[1]
                        layers = []
                        for neurons_for_a_layer in neuron_combination:
                            opt = SGD(lr, batch_size, True, True, 0.9)
                            layer = Layer(activation, input_size, neurons_for_a_layer,opt)
                            input_size = neurons_for_a_layer
                            """
                            # TODO
                            # test different ways of implementing weights
                            Something like:
                            layer.weights = np.something_not_gaussian(input_size, neurons_for_a_layer)
                            """
                            layers.append(layer)

                        # Output layer
                        opt = SGD(lr, batch_size, True, True, 0.9)
                        if problem_type == 'Classification':
                            layers.append(Layer(activations.sigmoid, input_size, 1, opt))
                        else:
                            layers.append(Layer(activations.identity, input_size, 1, opt))

                        network = NeuralNetwork(layers, loss_func)

                        train_losses = []
                        train_measure_score = []

                        eval_losses = []
                        eval_measure_score = []

                        best_model = None
                        best_loss = np.inf
                        best_measure = -np.inf
                        prev_weights = network.layers[-1].weights
                        for epoch in range(epochs):
                            #print(f"{epoch}---------------")
                            #print(prev_weights - network.layers[-1].weights,end="\r")
                            batch_train_losses = []
                            batch_train_measure = []
                            # Training
                            for sub_x, sub_y in random_mini_batch_generator(X_train, Z_train, batch_size = batch_size):
                                # Forward Pass
                                output = network.forward(sub_x)
                                # MSE and measure
                                train_loss = np.mean(network.cost(sub_y, output))
                                batch_train_losses.append(train_loss)
                                # Backward pass
                                network.backward(sub_y, lamb = lamb)

                                if problem_type == 'Classification':
                                    # Classify
                                    output = (output > 0.5).astype(int)

                                if problem_type == 'Regression':
                                    if sub_y.shape[0] == 1:
                                        # R2 scores cannot handle batch size of 1.
                                        continue
                                train_measure = measure(sub_y, output)
                                batch_train_measure.append(train_measure)
                            train_losses.append(np.mean(batch_train_losses))
                            train_measure_score.append(np.mean(batch_train_measure))

                            # Eval
                            batch_eval_losses = []
                            batch_eval_measure = []

                            for sub_x, sub_y in random_mini_batch_generator(X_eval, Z_eval, batch_size = batch_size):
                                eval_output = network.forward(sub_x)
                                eval_loss = np.mean(network.cost(sub_y, eval_output))
                                batch_eval_losses.append(eval_loss)
                                if problem_type == 'Classification':
                                    # Classify
                                    eval_output = (eval_output >= 0.5).astype(int)
                                if problem_type == 'Regression':
                                    if sub_y.shape[0] == 1:
                                        # R2 scores cannot handle batch size of 1.
                                        continue
                                eval_measure = measure(sub_y, eval_output)
                                batch_eval_measure.append(eval_measure)
                            eval_losses.append(np.mean(batch_eval_losses))
                            eval_measure_score.append(np.mean(batch_eval_measure))

                            # Tuning hyperparameters on eval set.
                            if eval_measure_score[-1] > best_measure:
                                best_model = copy.deepcopy(network)
                                best_loss = eval_losses[-1]
                                best_measure = eval_measure_score[-1]

                            # Simple early stopping
                            if epoch % 11 == 10:
                                # Very little change the past 10 epochs or if the training has become worse.
                                if np.abs(np.mean(eval_losses[5:]) - np.mean(eval_losses[10:])) < 0.0001:
                                    print("Early stopping, no learning", epoch, np.abs(np.mean(eval_losses[5:]) - np.mean(eval_losses[10:])),  np.mean(eval_losses[5:]),  np.mean(eval_losses[10:]))
                                    break

                        # Tuning hyperparameters on eval set.
                        if best_measure > global_best_measure:
                            print(f"New best {measure.__name__}: {best_measure}, {loss_func.__name__}: {best_loss}, learning rate: {lr}, lamb: {lamb} with {neuron_combination} and activation {activation.__name__}")
                            global_best_model = best_model
                            global_best_loss = best_loss
                            global_best_measure = best_measure
                            global_best_lr = lr
                            global_best_lamb = lamb
                            global_best_neuron_combo = neuron_combination
                            global_best_activation = activation

                            global_best_train_losses = train_losses
                            global_best_eval_losses = eval_losses
                            global_best_train_measure = train_measure_score
                            global_best_eval_measure = eval_measure_score

                            """
                            plt.plot(global_best_train_losses, label = "Train loss")
                            plt.plot(global_best_eval_losses, label = "Eval loss")
                            plt.legend()
                            plt.show()
                            plt.plot(global_best_train_measure, label = "Train Acc")
                            plt.plot(global_best_eval_measure, label = "Eval Acc")
                            plt.legend()
                            plt.show()
                            """

                            # TODO Remove this, this is cuz im impatient
                            if best_measure >= 0.95:
                                returns = {
                                    'best_Loss': global_best_loss,
                                    'best_'+str(measure.__name__): global_best_measure,
                                    'model': global_best_model,
                                    'activation': global_best_activation,
                                    'lr': global_best_lr,
                                    'lamb': global_best_lamb,
                                    'layer_neurons': global_best_neuron_combo,
                                    'train_losses': global_best_train_losses,
                                    'eval_losses': global_best_eval_losses,
                                    'train_measure': global_best_train_measure,
                                    'eval_measure': global_best_eval_measure
                                }

                                return returns


                    """
                    plt.plot(global_best_train_losses, label = "Train loss")
                    plt.plot(global_best_eval_losses, label = "Eval loss")
                    plt.legend()
                    plt.show()
                    plt.plot(global_best_train_measure, label = "Train Acc")
                    plt.plot(global_best_eval_measure, label = "Eval Acc")
                    plt.legend()
                    plt.show()
                    """

    returns = {
        'best_Loss': global_best_loss,
        'best_'+str(measure.__name__): global_best_measure,
        'model': global_best_model,
        'activation': global_best_activation,
        'lr': global_best_lr,
        'lamb': global_best_lamb,
        'layer_neurons': global_best_neuron_combo,
        'train_losses': global_best_train_losses,
        'eval_losses': global_best_eval_losses,
        'train_measure': global_best_train_measure,
        'eval_measure': global_best_eval_measure
    }

    return returns


def train_pytorch_net(custom_model, X, z, lr=0.1, epochs=20, batch_size = 16, lamb = 0, measure = r2_score):
    """Make a simple pytorch network to compare with."""

    X_train, X_eval, X_test = X
    z_train, z_eval, z_test = z

    X_train = torch.as_tensor((X_train), dtype=torch.float)
    X_eval = torch.as_tensor((X_eval), dtype=torch.float)
    X_test = torch.as_tensor((X_test), dtype=torch.float)

    # Regression
    if custom_model.cost.__name__ == 'mean_squared_error':
        scaler_output = StandardScaler(with_std = False)
        scaler_output.fit(z_train)

        z_train = torch.as_tensor((z_train), dtype=torch.float)
        z_eval = torch.as_tensor((z_eval), dtype=torch.float)
        z_test = torch.as_tensor((z_test), dtype=torch.float)

    else:
        z_train = torch.reshape(torch.as_tensor(z_train, dtype=torch.float), (-1,1))
        z_eval = torch.reshape(torch.as_tensor(z_eval, dtype=torch.float), (-1,1))
        z_test = torch.reshape(torch.as_tensor(z_test, dtype=torch.float), (-1,1))


    modules = []
    for layer in custom_model.layers:
        modules.append(nn.Linear(layer.input_size, layer.output_size))
        if layer.activation.__name__ == 'identity':
            continue
        elif layer.activation.__name__ == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif layer.activation.__name__ == 'relu':
            modules.append(nn.ReLU())
        elif layer.activation.__name__ == 'leaky_relu':
            modules.append(nn.LeakyReLU())


    model = torch.nn.Sequential(*modules)


    optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=lamb)

    train_dataset = torch.utils.data.TensorDataset(X_train, z_train)
    eval_dataset = torch.utils.data.TensorDataset(X_eval, z_eval)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True) # returns an iterator
    eval_loader =torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,shuffle=False)

    if custom_model.cost.__name__ == 'mean_squared_error':
        cost_func = torch.nn.MSELoss()
    else:
        cost_func = torch.nn.BCELoss()

    train_losses = []
    train_measures = []
    eval_losses = []
    eval_measures = []

    for e in range(epochs):
        model.train(True)
        batch_train_losses = []
        batch_train_mea = []
        for i, data in enumerate(train_loader):
            sub_x = data[0]
            sub_y = data[1]
            pred = model(sub_x)
            loss = cost_func(pred, sub_y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.detach().numpy().copy())
            if custom_model.cost.__name__ == 'binary_cross_entropy':
                pred = (pred >= 0.5)

            batch_train_mea.append(
                measure(sub_y.detach().numpy(), pred.detach().numpy())
            )
        train_losses.append(np.mean(batch_train_losses))
        train_measures.append(np.mean(batch_train_mea))

        model.train(False)
        batch_eval_losses = []
        batch_eval_mea = []
        for i, data in enumerate(eval_loader):
            sub_x = data[0]
            sub_y = data[1]
            pred = model(sub_x)
            loss = cost_func(pred, sub_y)
            batch_eval_losses.append(loss.detach().numpy().copy())
            if custom_model.cost.__name__ == 'binary_cross_entropy':
                pred = (pred >= 0.5)
            batch_eval_mea.append(
                measure(
                    sub_y.detach().numpy(),
                    pred.detach().numpy(),
                )
            )
        eval_losses.append(np.mean(batch_eval_losses))
        eval_measures.append(np.mean(batch_eval_mea))

    # Final test
    model.train(False)
    pred = model(X_test)
    test_loss = cost_func(pred, z_test)
    if custom_model.cost.__name__ == 'binary_cross_entropy':
        pred = (pred >= 0.5)
    test_measure = measure(z_test.detach().numpy(), pred.detach().numpy())

    return model, train_losses, train_measures, eval_losses, eval_measures, test_loss, test_measure
