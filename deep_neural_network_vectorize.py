import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.datasets import make_blobs
from tqdm.notebooks import tqdm

data, labels = make_blobs(n_samples=1000, features=2, centers=4, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify = labels, random_state = 0)
enc = OneHotEncoder()

y_OH_train = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()
y_OH_val = enc.fit_transform(np.expand_dims(Y_val,1)).toarray()

class FF_NN_Vector:
  def __init__(self): 
    self.W1 = np.random.randn(2, 2)
    self.W1 = np.random.randn(2, 4)
    self.B1 = np.zeros(1, 2)
    self.B2 = np.zeros(1, 4)

  def sigmoid(self, X):
    return 1/(1+np.exp(-X))

  def softmax(self, X):
    return np.exp(X)/np.sum(np.exp(X), axis = 1).reshape(1, -1)

  def forward_pass(self, X):
    self.A1 = np.matmul(self.W1, X) + self.B1 #(1,2)*(2,2)+(1,2) = (1,2)
    self.H1 = self.sigmoid(self.A1) #(1,2)
    self.A2 = np.matmul(self.H1, self.W2) + self.B2 #(1,2)*(2,4)+(1,4) = (1,4)
    self.H2 = self.softmax(self.A2) #(1,4)
    return self.H2

  def grad_sigmoid(self, X):
    return X*(1-X)

  def grad(self, X, Y):
    self.forward_pass(X)

    self.dA2 = self.H2 - Y #(1,4)
    self.dW2 = np.matmul(self.H1.T, self.dA2) #(2,1) * (1,4) =  (2,4)
    self.dB2 = np.sum(self.dA2, axis = 0).reshape(1, -1) #(1,4)
    self.dH1 = np.matmul(self.dA2, self.W2.T) #(1,4) * (4,2) = (1,2)
    self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1)) #(1,2)
    self.dW1 = np.matmul(X.T, self.dA1) #(2,1)*(1,2) = (2,2)
    self.dB1 = self.dA1

  def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):
      
    if display_loss:
      loss = {}
    
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
     self.grad(X, Y) 
        
      m = X.shape[0]
      self.W2 -= learning_rate * (dW2/m)
      self.B2 -= learning_rate * (dB2/m)
      self.W1 -= learning_rate * (dW1/m)
      self.B1 -= learning_rate * (dB1/m)

      if display_loss:
        Y_pred = self.predict(X)
        loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)
        
    
    if display_loss:
      plt.plot(loss.values())
      plt.xlabel('Epochs')
      plt.ylabel('Log Loss')
      plt.show()
      
  def predict(self, X):
    Y_pred = self.forward_pass(X)
    return np.array(Y_pred).squeeze()