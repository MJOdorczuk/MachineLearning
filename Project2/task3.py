import random
from itertools import permutations
import copy

from autograd import numpy as np
import matplotlib.pyplot as plt

from csnet.nn import Layer, NeuralNetwork
from csnet.activation import Activation
from csnet.loss import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def FrankeFunction(x, y, sigma = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma*noise

def tune_neural_network(X_train: np.ndarray, 
                        Z_train: np.ndarray, 
                        X_eval: np.ndarray, 
                        Z_eval: np.ndarray, 
                        epochs:int = 1000, 
                        batch_size:int = 16, 
                        lamb: float = 0):
    
    global_best_model = None
    global_best_loss = np.inf
    global_best_r2 = -np.inf
    global_best_lr = 0
    global_best_model_object = None
    global_best_lamb = 0
    global_best_neuron_combo = None
    global_best_activation = None

    global_best_train_losses = None
    global_best_eval_losses = None
    global_best_train_r2 = None
    global_best_eval_r2 = None

    activations = Activation()

    for lr in [3, 2.5, 2, 1.5, 1, 0.5]:
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
                    layers.append(Layer(activations.identity, input_size, 1))

                    network = NeuralNetwork(layers, mean_squared_error)
                    
                    train_losses = []
                    train_r2_scores = []

                    eval_losses = []
                    eval_r2_scores = []

                    best_model = None
                    best_loss = np.inf
                    best_r2 = -np.inf


                    for epoch in range(epochs):
                        
                        # Forward Pass
                        output = network.forward(X_train)
                        
                        # MSE and R2
                        train_loss = np.mean(network.cost(output, Z_train))
                        train_losses.append(train_loss)
                        train_r2 = r2_score(Z_train, output)
                        train_r2_scores.append(train_r2)
                        
                        # Backward pass
                        network.backward(Z_train, lr)

                        # Eval
                        eval_output = network.forward(X_eval)
                        eval_loss = np.mean(network.cost(eval_output, Z_eval))
                        eval_losses.append(eval_loss)
                        eval_r2 = r2_score(Z_eval, eval_output)
                        eval_r2_scores.append(eval_r2)

                        """
                        if epoch % 500 == 0:
                            print(f"Epoch {epoch}, Train R2: {train_r2}, Train loss: {train_loss}")
                            print(f"Epoch {epoch}, Eval R2: {eval_r2}, Eval loss: {eval_loss}")
                        """

                        # Tuning hyperparameters on eval set.
                        if eval_r2 > best_r2:
                            best_model = copy.deepcopy(network)
                            best_loss = eval_loss
                            best_r2 = eval_r2

                    # Tuning hyperparameters on eval set.
                    if best_r2 > global_best_r2:
                        print(f"New best R2: {best_r2}, MSE: {best_loss}, learning rate: {lr} with {neuron_combination} and activation {activation.__name__}")
                        global_best_model = best_model
                        global_best_loss = best_loss
                        global_best_r2 = best_r2
                        global_best_lr = lr
                        global_best_neuron_combo = neuron_combination
                        global_best_activation = activation

                        global_best_train_losses = train_losses
                        global_best_eval_losses = eval_losses
                        global_best_train_r2 = train_r2_scores
                        global_best_eval_r2 = eval_r2_scores

    returns = {
        'MSE': global_best_loss,
        'r2': global_best_r2,
        'model': global_best_model,
        'activation': global_best_activation,
        'lr': global_best_lr,
        'layer_neurons': global_best_neuron_combo,
        'train_losses': global_best_train_losses,
        'eval_losses': global_best_eval_losses,
        'train_r2': global_best_train_r2,
        'eval_r2': global_best_eval_r2
    }

    return returns


def train_and_test_neural_net(X: np.ndarray, Z: np.ndarray, epochs: int = 1000, batch_size: int = 16, lamb: float = 0):

    # Train and test (not evel - eval are being split in the trianing form the trianing set)
    X_train, X_test, z_train, z_test = train_test_split(X, Z, test_size=0.2)
    # Split train set into train and eval
    X_train, X_eval, z_train, z_eval = train_test_split(X_train, z_train, test_size=0.25)

    # Scale data by subtracting mean
    scaler_input = StandardScaler(with_std = False)
    scaler_input.fit(X_train)
    scaler_output = StandardScaler(with_std = False)
    scaler_output.fit(z_train)

    X_train = scaler_input.transform(X_train)
    X_eval = scaler_input.transform(X_eval)
    X_test = scaler_input.transform(X_test)

    z_train = scaler_output.transform(z_train)
    z_eval = scaler_output.transform(z_eval)
    z_test = scaler_output.transform(z_test)

    
    returns = tune_neural_network(X_train, z_train, X_eval, z_eval, epochs=epochs)

    # Final test
    best_model = returns['model']
    output = best_model.forward(X_test)
    test_loss = np.mean(best_model.cost(output, z_test))
    test_r2 = r2_score(z_test, output)

    print(f"Neural network final test on best model: MSE: {test_loss}, R2: {test_r2}")

    # Plotting losses and R2
    plt.plot(returns['train_losses'], label = "Train loss")
    plt.plot(returns['eval_losses'], label = "Eval loss")
    plt.legend()
    plt.show()
    plt.plot(returns['train_r2'], label = "Train R2")
    plt.plot(returns['eval_r2'], label = "Eval R2")
    plt.legend()
    plt.show()

    from IPython import embed; embed()


if __name__ == '__main__':
    num_points = 100
    num_epochs = 1000
    noise = 0.001
    
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)

    X = np.column_stack((x,y))
    Z = FrankeFunction(x, y, noise).reshape(-1,1)
    
    train_and_test_neural_net(X, Z)