class SigmoidNeuron:
  
  def __init__(self):
    self.w = None;
    self.b = None;
    
  def perceptron(self, x):
    return np.dot(x, self.w)+self.b;
  
  def sigmoid(self, x):
    return 1.0/1.0+np.exp(-x);
  
  def grad_w(self, x, y):
    f_x = self.sigmoid(self.perceptron(x))
    return (f_x - y) * f_x * (1 - f_x) * x
  
  def grad_b(self, x, y):
    f_x = self.sigmoid(self.perceptron(x))
    return (f_x - y) * f_x * (1 - f_x)
  
  def fit(self, X, Y, epochs = 1, learning_rate = 1):
    self.w = np.random.randn(1, X.shape[1])
    self.b = 0
    
    for i in range(epochs):
      dw, db = 0, 0;
      for x,y in zip(X, Y):
        dw += self.grad_w(x, y);
        db += self.grad_b(x, y);
      self.w -= learning_rate * dw;
      self.b -= learning_rate * db;