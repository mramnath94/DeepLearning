import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from tqdm import tqdm_notebook

class SigmoidNeuron:
  
  def __init__(self):
    self.w = None
    self.b = None
  
  def perceptron(self, x):
    return np.dot(x, self.w.T) + self.b
  
  def sigmoid(self, x):
    return 1/(1+np.exp(-x));
  
  def grad_w_mse(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x));
    return (y_pred-y) * y_pred * (1-y_pred) * x
  
  def grad_b_mse(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x));
    return (y_pred-y) * y_pred * (1-y_pred)
  
  def grad_w_ce(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x));
    if y == 0:
      return y_pred * x
    else:
      return -1 * (1-y_pred) * x
    
  def grad_b_ce(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x));
    if y == 0:
      return y_pred
    else:
      return -1 * (1-y_pred)
    
  def fit(self, X, Y, epochs = 1, learning_rate = 1, initialize = True, loss_function = 'mse', display_loss = True):
    if initialize:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0
      
    if display_loss:
      loss = {}
      
    for i in tqdm_notebook(range(epochs), total = epochs, unit = "epoch"):
      dw = 0;
      db = 0;
      for x,y in zip(X,Y):
        if loss_function == 'mse':
          dw += self.grad_w_mse(x,y)
          db += self.grad_b_mse(x,y)
        else:
          dw += self.grad_w_ce(x,y)
          db += self.grad_b_ce(x,y)
      m = X.shape[1]
      self.w -= learning_rate * (dw/m)
      self.b -= learning_rate * (db/m)
      
      if display_loss:
        Y_pred = self.sigmoid(self.perceptron(X));
        if loss_function == 'mse':
          loss[i] = mean_squared_error(Y, Y_pred)
        elif loss_function == 'ce':
          loss[i] = log_loss(Y, Y_pred)
          
          
    if display_loss:
      plt.plot(loss.values())
      plt.xlabel("Epochs")
      if loss_function == 'mse':
        plt.ylabel("Mean Squared Error")
      elif loss_function == "ce":
        plt.ylabel("Cross Entropy Loss")
      plt.show();
      
  
  def predict(self , X):
    Y_pred = []
    for x in X:
      y_pred = self.sigmoid(self.perceptron(x));
      Y_pred.append(y_pred)
    return np.array(Y_pred)