import numpy as np
from sklearn.metrics import accuracy_score, f1_score
class LinearSVM:
  def __init__(self, C = 1.0, max_iter = 1000, verbose = 0, learning_rate = 0.01):
    self.C = C # penalty parameter C of the error term.
    self.max_iter = max_iter # max iteration
    self.w = None
    self.verbose = verbose
    self.b = None
    self.learning_rate = learning_rate
    self.E = None

  def g(self, x):
    """
      g(x) > 0, label is 1; g(x) < 0, label is 0
      Parameter
      -------
      x : np.array((1, n))
    """
    return np.dot(self.w, x) + self.b

  def updateE(self, X, y):
    """
      calculate difference between g(x) and y
    """
    self.E = 1 - y*(np.dot(X, self.w.T) + self.b)

  def fit(self, X, y):
    n_iter = 0
    n = X.shape[0]
    # inital
    self.w = np.zeros((1, X.shape[1]))
    self.b = 0
    self.E = np.zeros((n, 1))
    self.updateE(X, y)
    temp1 = np.sum(y*X, axis=0).reshape(1, X.shape[1])
    temp2 = np.sum(y)
    while n_iter < self.max_iter:
      if np.max(self.E) <= 0:
        # converge
        print('congratulation converge!')
        continue
      self.w = (1 - self.learning_rate) * self.w + self.learning_rate * self.C * temp1
      self.b = self.b + self.learning_rate * self.C * temp2
      self.updateE(X, y)
      n_iter += 1
      if n_iter % self.verbose == 0:
        eval_set = self.predict(X)
        print('finish ', n_iter, 'ac ', accuracy_score(y, eval_set), 'f1_score ', f1_score(y, eval_set))
  
  def predict(self, X):
    """
    Parameters
    ------------
    X: np.array((n_samples, n_features)) the test data
    ------------
    Return
    ------------
    res: np.array((n_samples, 1)) only support one classification
    """
    result = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
      result[i] = 1 if self.g(X[i]) > 0 else -1
    return result
        



  