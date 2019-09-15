import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors

def sigmoid_2d(x, w, b): 
  return 1/(1+ np.exp(-(w*x+b)))

w = 3
b = 3
X = np.linspace(-10, 10, 100)
Y = sigmoid_2d(X, w, b)
plt.plot(X, Y)
plt.show()

def sigmoid_3d(x1, x2, w1, w2, b):
  return 1/(1+ np.exp(-(w1*x1 + w2*x2 +b))) 

X1 = np.linspace(-10, 10, 100)
X2 = np.linspace(-10, 10, 100)
XX1, XX2 = np.meshgrid(X1, X2)

w1 = 0.2
w2 = 0.8
b = 0.6

Y = sigmoid_3d(XX1, XX2, w1, w2, b)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

#ax.view_init(30, 270)