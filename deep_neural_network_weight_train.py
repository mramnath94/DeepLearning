from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import copy

data,labels = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
plt.scatter(data[:,0], data[:,1], c=labels)
plt.show()
labels_orig = labels
labels = np.mod(labels_orig, 2)
plt.scatter(data[:,0], data[:,1], c=labels)
plt.show()
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)

class FFNN:

  def __init__(self):
    self.w1 = np.random.randn()
    self.w2 = np.random.randn()
    self.w3 = np.random.randn()
    self.w4 = np.random.randn()
    self.w5 = np.random.randn()
    self.w6 = np.random.randn()
    self.b1 = 0
    self.b2 = 0
    self.b3 = 0

  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))

  def forward_pass(self, x):
    self.x1, self.x2 = x
    self.a1 = self.w1*self.x1 + self.w2*self.x2 + self.b1
    self.h1 = self.sigmoid(self.a1)
    self.a2 = self.w3*self.x1 + self.w4*self.x2 + self.b2
    self.h2 = self.sigmoid(self.a2)
    self.a3 = self.w5*self.h1 + self.w6*self.h2 + self.b2
    self.h3 = self.sigmoid(self.a3)
    return self.h3

  def grad(self, x, y):
    self.forward_pass(x)
    self.dw1 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1) * self.x1

  def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):
    
    if display_loss:
      loss = {}
      w_plot = []

    for i in tqdm_notebook(range(epochs), total =epochs, unit = 'epochs'):
      dw1 = [0]
      for x,y in zip(X,Y):
        self.grad(x,y)
        dw1+= self.dw1;
      
      m = X.shape[0]
      self.w1 -= learning_rate * dw1 / m
      if display_loss:
        w_plot.append(copy.deepcopy(self.w1))
        Y_pred = self.predict(X)
        loss[i] = mean_squared_error(Y_pred, np.array(Y))
    
    if display_loss:
      plt.tight_layout()
      plt.subplot(2,1,1)
      plt.plot(w_plot)
      plt.xlabel('Epochs')
      plt.ylabel('W1')
      
      plt.subplot(2,1,2)
      plt.plot(list(loss.values()))
      plt.xlabel('Epochs')
      plt.ylabel('Mean Squared Error')

  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.forward_pass(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred)

ffn = FFNN()
ffn.fit(X_train, Y_train, epochs=2000, learning_rate=.01, display_loss=True)