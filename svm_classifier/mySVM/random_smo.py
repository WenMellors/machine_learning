import numpy as np

class RandomSMO:
  def __init__(self, C = 1.0, tol = 0.0001, max_iter = 1000, verbose = 0, random_seed = 0):
    self.C = C # penalty parameter C of the error term.
    self.tol = tol # tolerance for stopping criteria.
    self.max_iter = max_iter # max iteration
    self.E = None # the difference between predict_y and y
    self.w = None
    self.b = 0 # wx + b = 0
    self.verbose = verbose
    self.random = np.random.RandomState(seed = random_seed)
    self.failKKTCnt = 0
  
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

  def checkStop(self, a, y):
    """
    check all KKT with tolerance
    consider singularity, if 99% points match KKT is ok
    """
    cnt = 0
    for i in range(a.shape[0]): # most not match KKT a_i
      if a[i] == 0 and y[i] * (self.E[i] + y[i]) < 1 - self.tol:
        cnt += 1
      elif a[i] == self.C and y[i] * (self.E[i] + y[i]) > 1 + self.tol:
        cnt += 1
      elif a[i] > 0 and a[i] < self.C and (y[i] * (self.E[i] + y[i]) < 1 - self.tol or y[i] * (self.E[i] + y[i]) > 1 + self.tol):
        cnt += 1
    self.failKKTCnt = cnt
    if cnt > a.shape[0] * 0.01:
      return False
    else:
      return True


  def fit(self, X, y):
    """
    Parameters
    ------------
    X: np.array((n_samples, n_features)) the training data
    y: np.array((n_samples, 1)) the training label
    """
    a = np.zeros((X.shape[0], 1)) # init a with 0
    n_iter = 0
    self.w = np.zeros((1, X.shape[1])) # init is 0
    self.E = np.zeros((X.shape[0], 1))
    self.updateE(X, y)
    self.checkStop(a, y)
    print('init {0} point not match KKT'.format(self.failKKTCnt))
    while n_iter < self.max_iter:
      # random select a not match KKT point
      # many we select a1 a2 cannot improve the score,so just random choose a1 a2
      """
      while True:
        first_i = self.random.randint(0, a.shape[0])
        if a[first_i] == 0 and y[first_i] * (self.E[first_i] + y[first_i]) < 1:
          break
        elif a[first_i] == self.C and y[first_i] * (self.E[first_i] + y[first_i]) > 1:
          break
        elif a[first_i] > 0 and a[first_i] < self.C and y[first_i] * (self.E[first_i] + y[first_i]) != 1:
          break
      second_i = first_i
      while second_i == first_i and np.linalg.norm(X[first_i] - X[second_i]) == 0:
        if self.E[first_i] > 0:
          second_i = np.random.choice(np.where(self.E == np.min(self.E))[0])
        else:
          second_i = np.random.choice(np.where(self.E == np.max(self.E))[0])
        if np.linalg.norm(X[first_i] - X[second_i]) == 0:
          second_i = self.random.randint(0, a.shape[0])
      """
      while True:
        first_i = self.random.randint(0, a.shape[0])
        second_i = self.random.randint(0, a.shape[0])
        if first_i == second_i or np.linalg.norm(X[first_i] - X[second_i]) == 0:
          continue # repeat select
        # calculate new a2, a1
        a2_newunc = a[second_i] + y[second_i] * (self.E[first_i] - self.E[second_i]) / np.linalg.norm(X[first_i] - X[second_i])
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
        if a[second_i] == a2_new:
          continue # if not improve a2, reselect
        else:
          break
      
      a1_new = a[first_i] +  y[first_i] * y[second_i] *(a[second_i] - a2_new)
      # update b
      if a1_new > 0 and a1_new < self.C:
        self.b = - self.E[first_i] - y[first_i] * self.k(X[first_i], X[first_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[first_i]) * (a2_new - a[second_i]) + self.b
      elif a2_new > 0 and a2_new < self.C:
        self.b = - self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b
      else:
        self.b = ((- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b) + (- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a[first_i]) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a[second_i]) + self.b)) / 2
      # update a
      # print(a[first_i], a[second_i])
      # print(a1_new, a2_new)
      a[first_i] = a1_new
      a[second_i] = a2_new
      # update W
      self.updateW(X, y, a)
      # print("w ", self.w, " b ", self.b)
      # update E
      self.updateE(X, y)
      n_iter += 1
      if self.verbose != 0 and n_iter % self.verbose == 0:
        print('finish {0}'.format(n_iter))
      # check stop
      if self.checkStop(a, y):
        print("converge {0}".format(n_iter))
        return
    print('fail to converge')
    print('{0} point not match KKT'.format(self.failKKTCnt))

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

      
