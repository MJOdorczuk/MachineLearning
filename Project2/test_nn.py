import random

from autograd import numpy as np
from csnet.nn import layer, nn, meanSquares
from csnet.activation import Activation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

def FrankeFunction(x, y, sigma = 0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    noise = np.random.normal(0, 1, x.shape[0])

    return (term1 + term2 + term3 + term4) + sigma*noise

def sincos(x,y,sigma = 0):
    return x * y

num_points = 10
num_epochs = 1
noise = 0



a = Activation()
def id(x):
    return x

l1 = layer(a.sigmoid, 2, 4)
l2 = layer(a.sigmoid, 4, 8)
l3 = layer(a.sigmoid, 8, 16)
l4 = layer(id, 16, 1)
layers = [l1, l2, l3, l4]
n = nn(layers, meanSquares)

merr = 0
loss = []

fig = plt.figure()

x = np.linspace(0,1,30)
y = np.linspace(0,1,30)
X, Y = np.meshgrid(x, y)


for j in range(num_epochs):
    xs = (np.random.uniform(0, 1, num_points))
    ys =  (np.random.uniform(0, 1, num_points))
    zs = FrankeFunction(xs, ys, noise) # Target
    err = n.error(np.array([xs,ys]).T,zs)
    print("epoch",j,err)
    loss.append(err)

    # f = np.vectorize((lambda x,y: n.forward([x,y])[0]))
    # x = np.linspace(0,1,30)
    # y = np.linspace(0,1,30)
    # X, Y = np.meshgrid(x, y)
    # Z = f(X,Y)

    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='viridis')

    # x = np.linspace(0,1,30)
    # y = np.linspace(0,1,30)
    # X, Y = np.meshgrid(x, y)
    # Z = FrankeFunction(X,Y)
    # ax.contour3D(X, Y, Z, 50, cmap='binary')

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
    x = np.array([xs,ys]).T
    n.back(x, zs, 0.01)
    # for i in range(len(xs)):
    #     x_ = np.array([xs[i], ys[i]])
    #     z_ = np.array([zs[i]])
    #     prerr = n.error(x_,z_)
    #     n.back(x_, z_, 0.01)
    #     merr = (merr * 0.99 + prerr * 0.01)
    #     #print(f"{i}th train error is {posterr} for test error {prerr}, pseudo-mean is {merr}")
    #     if(i%1000==0):
    #         #print(l1.weights, l4.weights)
    #         pass#print(merr)




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

# x = np.linspace(0,1,30)
# y = np.linspace(0,1,30)
# X, Y = np.meshgrid(x, y)
# Z = FrankeFunction(X,Y) - f(X,Y)
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# plt.show()
