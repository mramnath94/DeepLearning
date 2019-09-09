class Perceptron:
  def __init__(self):
    self.w = None;
    self.b = None;
    
  def model(self, x):
    dotp = np.dot(self.w, x)
    if((dotp >= self.b).all()):
      return 1
    else:
      return 0
    
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  
  def fit(self, X, Y, epochs = 1, learning_rate = 1):
    max_accuracy = 0
    accuracy = {}
    max_b = {}
    max_w = {}
    self.w = np.ones(X.shape[1])
    self.b = 0
    wts = []
    for i in range(epochs):
      for x,y in zip(X,Y): 
        pred = self.model(x)
        if y == 1 and pred == 0:
          self.w = self.w + learning_rate*x;
          self.b = self.b - learning_rate*x;
        if y == 0 and pred == 1:
          self.w = self.w - learning_rate*x;
          self.b = self.b + learning_rate*x;
      accuracy[i] = accuracy_score(self.predict(X), Y)
      if(accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        max_b = self.b
        max_w = self.w
    print(max_accuracy)


#Below code helps in setting up data to run the perceptron model for breast cancer data
#import pandas as pd
#import numpy as np
#import sklearn.datasets
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score 
#
#data = sklearn.datasets.load_breast_cancer()
#pd_data = pd.DataFrame(data.data, columns = data.feature_names)
#pd_data['class'] = data.target
#X = pd_data.drop('class', axis = 1)
#Y = pd_data['class']
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#Below code helps to train and predict breast cancer results with Perceptron model

#perceptron = Perceptron();
#perceptron.fit(X_train.values, Y_train, 10000, 0.1)
#predictions = perceptron.predict(X_test.values)
#test_accuracy = accuracy_score(predictions, Y_test)
#print(test_accuracy)
