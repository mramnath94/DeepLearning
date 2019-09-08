class MPNeuron:
  def __init__(self):
    self.b = None;
  
  def model(self, x): 
    return np.sum(x) >= self.b
  
  def predict(self, X): 
    Y = []
    for x in X:
      y = self.model(x)
      Y.append(y)
    return np.array(Y)
  
  def fit(self, X, Y): 
    max_b = 0
    max_accuracy = 0
    for i in range (X.shape[1]+1):
      self.b = i
      result = self.predict(X)
      accuracy = accuracy_score(result, Y)
      if(accuracy > max_accuracy):
        max_accuracy = accuracy
        max_b = self.b
    self.b = max_b    
    print(max_b, max_accuracy)