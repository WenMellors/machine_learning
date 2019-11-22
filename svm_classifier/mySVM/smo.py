import numpy as np

class SMO:
  def __init__(self, C = 1.0, tol = 0.0001, max_iter = 1000, verbose = 0):
    self.C = C # penalty parameter C of the error term.
    self.tol = tol # tolerance for stopping criteria.
    self.max_iter = max_iter # max iteration
    self.E = None # the difference between predict_y and y
    self.w = None
    self.b = 0 # wx + b = 0
    self.verbose = verbose
  
  def k(self, x, z):
    """
    kernel function
    now just use f(x) = x
    Parameters
    -----------
    a, b : should be np.array((n, 1))
    """
    return np.dot(x, z)

  def updateW(self, X, y, a):
    """
    calculate w
    """
    self.w = np.dot(np.array(a*y).reshape(1, X.shape[0]), X)

  def updateE(self, X, y):
    #for i in range(X.shape[0]):
    #  self.E[i] = self.g(X[i]) - y[i]
    # use np to accelerate
    self.E = np.apply_along_axis(lambda x: self.g(x), axis=1, arr=X) - y

  def g(self, x):
    """
      g(x) > 0, label is 1; g(x) < 0, label is 0
      Parameter
      -------
      x : np.array((1, n))
    """
    return np.dot(self.w, x) + self.b


  def fit(self, X, y):
    """
    Parameters
    ------------
    X: np.array((n_samples, n_features)) the training data
    y: np.array((n_samples, 1)) the training label
    """
    a = np.zeros((X.shape[0], 1)) # init a with 0
    n_iter = 0
    self.w = np.ones((1, X.shape[1])) # init is 1
    self.E = np.zeros((X.shape[0], 1))
    self.updateE(X, y)
    while n_iter < self.max_iter:
      maxKKTBias = 0 # 0 is match KKT
      first_i = 0
      for i in range(a.shape[0]): # most not match KKT a_i
        if a[i] == 0 and y[i] * (self.E[i] + y[i]) < 1 and 1 - y[i] * (self.E[i] + y[i]) > maxKKTBias:
          maxKKTBias = 1 - y[i] * (self.E[i] + y[i])
          first_i = i
        elif a[i] == self.C and y[i] * (self.E[i] + y[i]) > 1 and y[i] * (self.E[i] + y[i]) - 1 > maxKKTBias:
          maxKKTBias = y[i] * (self.E[i] + y[i]) - 1
          first_i = i
        elif a[i] > 0 and a[i] < self.C and y[i] * (self.E[i] + y[i]) != 1 and abs(y[i] * (self.E[i] + y[i]) - 1) > maxKKTBias:
          maxKKTBias = abs(y[i] * (self.E[i] + y[i]) - 1)
          first_i = i
      # check whether convergence
      if n_iter != 0 and maxKKTBias < self.tol:
        print('converge!')
        return
      second_i = first_i
      while second_i == first_i and np.linalg.norm(X[first_i] - X[second_i]) == 0:
        if self.E[first_i] > 0:
          second_i = np.random.choice(np.where(self.E == np.min(self.E))[0])
        else:
          second_i = np.random.choice(np.where(self.E == np.max(self.E))[0])
      # calculate new a2, a1
      a2_newunc = a[second_i] + y[second_i] * (self.E[first_i] - self.E[second_i]) / np.linalg.norm(X[first_i] - X[second_i]) # TODO: use kernel func
      # calculate H and L
      if y[first_i] == y[second_i]:
        L = max(0, a[second_i] + a[first_i] - self.C)
        H = min(self.C, a[second_i] + a[first_i])
      else:
        L = max(0, a[second_i] - a[first_i])
        H = min(self.C, self.C + a[second_i] - a[first_i])
      if a2_newunc > H:
        a2_new = H
      elif a2_newunc < L:
        a2_new = L
      else:
        a2_new = a2_newunc
      a1_new = a[first_i] +  y[first_i] * y[second_i] *(a[second_i] - a2_new)
      # update b
      if a1_new > 0 and a1_new < self.C:
        self.b = - self.E[first_i] - y[first_i] * self.k(X[first_i], X[first_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[first_i]) * (a2_new - a[second_i]) + self.b
      elif a2_new > 0 and a2_new < self.C:
        self.b = - self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b
      else:
        self.b = ((- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b) + (- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b)) / 2
      # update a
      a[first_i] = a1_new
      a[second_i] = a2_new
      # update W
      self.updateW(X, y, a)
      # update E
      self.updateE(X, y)
      n_iter += 1
      if self.verbose != 0 and n_iter % self.verbose == 0:
        print('finish {0}'.format(n_iter))
    print(maxKKTBias)
    print('fail to converge')

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

      
