from itertools import permutations
import copy
from typing import Callable

from autograd import numpy as np

from csnet.nn import Layer, NeuralNetwork
from csnet.activation import Activation
from csnet.loss import mean_squared_error
from csnet.optim import _random_mini_batch_generator

from sklearn.metrics import r2_score


def tune_neural_network(X_train: np.ndarray, 
                        Z_train: np.ndarray, 
                        X_eval: np.ndarray, 
                        Z_eval: np.ndarray, 
                        epochs:int = 1000, 
                        batch_size:int = 16, 
                        lamb: float = 0,
                        learning_rates = [3, 2.5, 2, 1.5, 1, 0.5],
                        measure: Callable = r2_score,
                        loss_func: Callable = mean_squared_error,
                        problem_type: str = 'Regression'):
    
    # TODO
    """
    Add some raise error: Not regression or classifiation
    """

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

    for lr in learning_rates:
        print(f"New Learning rate: {lr}")
        for num_layers in range(3):
            for activation in activations.get_all_activations():
                all_num_neurons_combinations = permutations([1,2,3,4,5], num_layers)
                for neuron_combination in all_num_neurons_combinations:
                    
                    # Construct Neural network
                    input_size = X_train.shape[1]
                    layers = []
                    for neurons_for_a_layer in neuron_combination:
                        layer = Layer(activation, input_size, neurons_for_a_layer)
                        input_size = neurons_for_a_layer
                        """
                        # TODO
                        # test different ways of implementing weights
                        Something like:
                        layer.weights = np.something_not_gaussian(input_size, neurons_for_a_layer)
                        """
                        layers.append(layer)
                    
                    # Output layer
                    if problem_type == 'Classification':
                        layers.append(Layer(activations.sigmoid, input_size, 1))
                    else:
                        layers.append(Layer(activations.identity, input_size, 1))

                    network = NeuralNetwork(layers, loss_func)
                    
                    train_losses = []
                    train_measure_score = []

                    eval_losses = []
                    eval_measure_score = []

                    best_model = None
                    best_loss = np.inf
                    best_measure = -np.inf

                    for epoch in range(epochs):

                        batch_train_losses = []
                        batch_train_measure = []
                        # Training
                        for sub_x, sub_y in _random_mini_batch_generator(X_train, Z_train, batch_size = batch_size):
                            # Forward Pass
                            output = network.forward(sub_x)
                            
                            # MSE and measure
                            train_loss = np.mean(network.cost(output, sub_y))
                            batch_train_losses.append(train_loss)

                            if problem_type == 'Classification':
                                # Classify
                                output = (output > 0.5).astype(int)
                            
                            train_measure = measure(output, sub_y)
                            batch_train_measure.append(train_measure)
                            
                            # Backward pass
                            network.backward(sub_y, lr)

                        train_losses.append(np.mean(batch_train_losses))
                        train_measure_score.append(np.mean(batch_train_measure))

                        # Eval
                        batch_eval_losses = []
                        batch_eval_measure = []

                        for sub_x, sub_y in _random_mini_batch_generator(X_eval, Z_eval, batch_size = batch_size):
                            eval_output = network.forward(sub_x)
                            eval_loss = np.mean(network.cost(eval_output, sub_y))
                            batch_eval_losses.append(eval_loss)
                            if problem_type == 'Classification':
                                # Classify
                                eval_output = (eval_output > 0.5).astype(int)

                            eval_measure = measure(eval_output, sub_y)
                            batch_eval_measure.append(eval_measure)
                        
                        eval_losses.append(np.mean(batch_eval_losses))
                        eval_measure_score.append(np.mean(batch_eval_measure))

                        # Tuning hyperparameters on eval set.
                        if eval_measure_score[-1] > best_measure:
                            best_model = copy.deepcopy(network)
                            best_loss = eval_losses[-1]
                            best_measure = eval_measure_score[-1]

                    # Tuning hyperparameters on eval set.
                    if best_measure > global_best_measure:
                        print(f"New best {measure.__name__}: {best_measure}, {loss_func.__name__}: {best_loss}, learning rate: {lr} with {neuron_combination} and activation {activation.__name__}")
                        global_best_model = best_model
                        global_best_loss = best_loss
                        global_best_measure = best_measure
                        global_best_lr = lr
                        global_best_neuron_combo = neuron_combination
                        global_best_activation = activation

                        global_best_train_losses = train_losses
                        global_best_eval_losses = eval_losses
                        global_best_train_measure = train_measure_score
                        global_best_eval_measure = eval_measure_score

                        # TODO Remove this, this is cuz im impatient
                        if best_measure == 1:
                            returns = {
                                'MSE': global_best_loss,
                                str(measure.__name__): global_best_measure,
                                'model': global_best_model,
                                'activation': global_best_activation,
                                'lr': global_best_lr,
                                'layer_neurons': global_best_neuron_combo,
                                'train_losses': global_best_train_losses,
                                'eval_losses': global_best_eval_losses,
                                'train_measure': global_best_train_measure,
                                'eval_measure': global_best_eval_measure
                            }

                            return returns

    returns = {
        'MSE': global_best_loss,
        str(measure.__name__): global_best_measure,
        'model': global_best_model,
        'activation': global_best_activation,
        'lr': global_best_lr,
        'layer_neurons': global_best_neuron_combo,
        'train_losses': global_best_train_losses,
        'eval_losses': global_best_eval_losses,
        'train_measure': global_best_train_measure,
        'eval_measure': global_best_eval_measure
    }

    return returns
