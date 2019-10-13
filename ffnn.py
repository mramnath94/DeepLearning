import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from tqdm import tqdm_notebook

class FeedForwardNeuralNetwork:
  def __init__(self, input_size, output_size, layer_info = [2]):
    self.nx = input_size;
    self.ny = output_size || 1;
    self.nh = len(layer_info);
    self.sizes = [self.nx] + layer_info + [self.ny];
    self.W = {}
    self.B = {}
    for i in range(self.nh+1):
      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
      self.B[i+1] = np.zeros((1, self.sizes[i+1]))


  def sigmoid(self, x):
    return 1/(1+np.exp(-x));

  def softmax(self, x):
    exps = np.exp(x)
    return exps / np.sum(exps)

  def forward_pass(self, x): 
    self.A = {}
    self.H = {}
    self.H[0] = x.reshape(1, -1);
    for i in range(self.nh):
      self.A[i+1] = np.matmul(self.H[i], self.W[i+1])+self.B[i+1];
      self.H[i+1] = self.sigmoid(self.A[i+1]);
    self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1])+self.B[self.nh+1]
    self.H[self.nh+1] = self.softmax(self.A[self.nh+1])
    return self.H[self.nh+1]

  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.forward_pass(x);
      Y_pred.append(y_pred)
    return np.array(Y_pred).squeeze();

  def grad_sigmoid(self, x):
    return x*(1-x) 
  
  def cross_entropy(self,label,pred):
    yl=np.multiply(pred,label)
    yl=yl[yl!=0]
    yl=-np.log(yl)
    yl=np.mean(yl)
    return yl
 
  def grad(self, x, y):
    self.forward_pass(x)
    self.dW = {}
    self.dB = {}
    self.dH = {}
    self.dA = {}
    L = self.nh + 1
    self.dA[L] = (self.H[L] - y)
    for k in range(L, 0, -1):
      self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])
      self.dB[k] = self.dA[k]
      self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)
      self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1])) 
  
  def fit(self, X, Y, epochs = 1, learning_rate = 1, initialize = True, display_loss = False):
    if display_loss:
      loss = {}

    if initialize: 
      for i in range(self.dh+1):
        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
        self.B[i+1] = np.zeros((1, self.sizes[i+1]))

    for e in tqdm_notebook(range(epochs), total = epochs, unit = "epoch"): 
      dW = {}
      dB = {}
      for i in range(self.nh+1):
        dW[i+1] = np.zeros(self.sizes[i], self.sizes[i+1])
        dB[i+1] = np.zeros(1, self.nh+1)
      for x,y in zip(X, Y):
        self.grad(x, y)
        for i in range(self.nh+1):
          dW[i+1] += self.dW[i+1]
          dB[i+1] += self.dB[i+1]

      m = X.shape[1]
      for i in range(self.nh+1):
        self.W[i+1] -= learning_rate * (dW[i+1]/m)
        self.B[i+1] -= learning_rate * (dB[i+1]/m)
        
      if display_loss:
        Y_pred = self.predict(X) 
        loss[epoch] = self.cross_entropy(Y, Y_pred)
    
    if display_loss:
      plt.plot(loss.values())
      plt.xlabel('Epochs')
      plt.ylabel('CE')
      plt.show() 



