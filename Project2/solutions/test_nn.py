import random

from autograd import numpy as np
from csnet.nn import Layer as layer
from csnet.nn import NeuralNetwork as nn
from csnet.activation import Activation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from csnet.loss import mean_squared_error

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    print("okok")

    def FrankeFunction(x, y, sigma = 0):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        noise = np.random.normal(0, 1, x.shape[0])

        return (term1 + term2 + term3 + term4) + sigma*noise

    num_points = 100
    num_epochs = 10000
    noise = 0.001

    a = Activation()
    def id(x):
        return x

    l1 = layer(a.relu, 2, 4)
    l2 = layer(a.sigmoid, 4, 4)
    l3 = layer(a.relu, 4, 6)
    l4 = layer(id, 6, 1)
    layers = [l1, l2, l3, l4]
    n = nn(layers, mean_squared_error)


    loss = []

    fig = plt.figure()

    x = np.linspace(0,1,30)
    y = np.linspace(0,1,30)
    X, Y = np.meshgrid(x, y)

    eta = 0.5

    xs = (np.random.uniform(0, 1, num_points))
    ys =  (np.random.uniform(0, 1, num_points))
    zs = FrankeFunction(xs, ys, noise).reshape(-1,1)

    X = np.column_stack((xs,ys))
    Z = FrankeFunction(xs, ys, noise).reshape(-1,1)
    import time
    start = time.time()
    for j in range(num_epochs):
        x = np.array([xs,ys]).T
        n.forward(x)
        n.backward(zs, eta)
    print(time.time()-start)
    plt.plot(loss[1:])
    plt.show()

    x = np.linspace(0,1,30)
    y = np.linspace(0,1,30)
    X, Y = np.meshgrid(x, y)
    Z = FrankeFunction(X,Y)

    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')

    f = np.vectorize((lambda x,y: n.forward([x,y])[0]))
    x = np.linspace(0,1,30)
    y = np.linspace(0,1,30)
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)
    ax.contour3D(X, Y, Z, 50, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()