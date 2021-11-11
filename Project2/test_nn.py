import random

from autograd import numpy as np
from csnet.nn import Layer as layer
from csnet.nn import NeuralNetwork as nn
from csnet.nn import init_SGD
from csnet.activation import Activation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import tensorflow as tf

np.random.seed(0)
random.seed(0)

def FrankeFunction(x, y, sigma = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma*noise

num_points = 100
num_epochs = 10000
noise = 0.01

a = Activation()
def id(x):
    return x

lamba = 0.0002

in_size = 2
h1_size = 4
h2_size = 8
h3_size = 16
out_size = 1

l1 = layer(a.sigmoid, in_size, h1_size, init_SGD(lamba))
l2 = layer(a.sigmoid, h1_size, h2_size, init_SGD(lamba))
l3 = layer(a.sigmoid, h2_size, h3_size, init_SGD(lamba))
l4 = layer(id, h3_size, out_size, init_SGD(lamba))
layers = [l1, l2, l3, l4]
n = nn(layers)

def tf_nn_train(X,Y):
    input = tf.placeholder("float", shape=[None, in_size])
    output = tf.placeholder("float", shape=[None, out_size])
    w1 = tf.Variable(tf.random_normal((in_size,h1_size), stddev=1.))
    w2 = tf.Variable(tf.random_normal((h1_size,h2_size), stddev=1.))
    w3 = tf.Variable(tf.random_normal((h2_size, h3_size), stddev=1.))
    w4 = tf.Variable(tf.random_normal((h3_size, out_size), stddev=1.))

    l1 = tf.nn.sigmoid(tf.matmul(input, w1))
    l2 = tf.nn.sigmoid(tf.matmul(l1, w2))
    l3 = tf.nn.sigmoid(tf.matmul(l2, w3))
    l4 = tf.matmul(l3, w4)

    predict = tf.argmax(l4, axis=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output, logits=l4))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


    model = 

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

loss = []

for j in range(num_epochs):
    x = np.array([xs,ys]).T
    z_tilde = n.forward(x)
    n.backward(zs, eta)
    loss.append(mse(zs, z_tilde))

plt.plot(loss)
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
