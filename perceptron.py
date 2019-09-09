class Perceptron:
  def __init__(self):
    self.w = None;
    self.b = None;
    
  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0
    
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  
  def fit(self, X, Y, epochs = 1, learning_rate = 0.1):
    max_accuracy = 0
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
    print(max_accuracy, self.w, self.b)