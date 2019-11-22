import numpy as np

class FinalSMO:
  def __init__(self, C = 1.0, tol = 0.0001, max_iter = 1000, verbose = 0, random_seed = 0, KKTdelta = 0.01, w=None, b = 0):
    self.C = C # penalty parameter C of the error term.
    self.tol = tol # tolerance for stopping criteria.
    self.max_iter = max_iter # max iteration
    self.E = None # the difference between predict_y and y
    self.w = w
    self.b = b # wx + b = 0
    self.verbose = verbose
    self.random = np.random.RandomState(seed = random_seed)
    self.KKTdelta = KKTdelta # we accept KKTdelta * n_smaple not match KKT
    self.failKKTList = []
    self.failKKTBorderList = []
    self.targetValue = 0
    self.bestW = 0
    self.bestB = 0
    self.bestTargetValue = 0
    self.bestNotMatchKKT = 0
    self.bestA = None
    self.bestIter = 0
    self.bestE = None
    self.notMatchKKT = 0
  
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
    self.failKKTList = []
    self.failKKTBorderList = []
    cnt = 0
    for i in range(a.shape[0]): # most not match KKT a_i
      if a[i] == 0 and y[i] * (self.E[i] + y[i]) < 1 - self.tol:
        self.failKKTList.append(i)
        cnt += 1
      elif a[i] == self.C and y[i] * (self.E[i] + y[i]) > 1 + self.tol:
        self.failKKTList.append(i)
        cnt += 1
      elif a[i] > 0 and a[i] < self.C and (y[i] * (self.E[i] + y[i]) < 1 - self.tol or y[i] * (self.E[i] + y[i]) > 1 + self.tol):
        self.failKKTBorderList.append(i)
        cnt += 1
    self.notMatchKKT = cnt
    if cnt != 0:
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
    self.bestA = np.copy(a)
    self.bestE = np.copy(self.E)
    self.updateE(X, y)
    self.checkStop(a, y)
    self.bestNotMatchKKT = int(self.notMatchKKT)
    print('start')
    while n_iter < self.max_iter:
      # random select a not match KKT point
      # many we select a1 a2 cannot improve the score,so just random choose a1 a2
      reselect_time = 0
      while True:
        # print('reselect time {} ',format(reselect_time), a)
        if len(self.failKKTBorderList) == 0:
          first_i = self.random.choice(self.failKKTList) # select a1 from fail KKT List not border
        else:
          first_i = self.random.choice(self.failKKTBorderList)
        if reselect_time < 3:
          if self.E[first_i] > 0:
            second_i = self.random.choice(np.where(self.E == np.min(self.E))[0])
          else:
            second_i = self.random.choice(np.where(self.E == np.max(self.E))[0])
          while first_i == second_i or np.linalg.norm(X[first_i] - X[second_i]) == 0:
            second_i = self.random.randint(0, a.shape[0]) # select a random a2
        else:
          # fail to inspire
          second_i = self.random.randint(0, a.shape[0])
          while first_i == second_i or np.linalg.norm(X[first_i] - X[second_i]) == 0:
            second_i = self.random.randint(0, a.shape[0]) # select a random a2
        # calculate a2_newunc
        a2_newunc = a[second_i] + y[second_i] * (self.E[first_i] - self.E[second_i]) / np.linalg.norm(X[first_i] - X[second_i])
        # print('a2_newunc ', a2_newunc)
        # calculate H and L
        if y[first_i] == y[second_i]:
          L = max(0, a[second_i] + a[first_i] - self.C)
          H = min(self.C, a[second_i] + a[first_i])
        else:
          L = max(0, a[second_i] - a[first_i])
          H = min(self.C, self.C + a[second_i] - a[first_i])
        # calculate a2_new
        if a2_newunc > H:
          a2_new = H
        elif a2_newunc < L:
          a2_new = L
        else:
          a2_new = a2_newunc
        a1_new = a[first_i] +  y[first_i] * y[second_i] *(a[second_i] - a2_new)
        oldValue = self.targetValue
        a1_old = float(a[first_i])
        a2_old = float(a[second_i]) # TODO:solve this numpy bug
        a[first_i] = 0
        a[second_i] = 0
        # if data is too big re calculate targetValue is not accepted
        a1_diff = a1_old - a1_new
        a2_diff = a2_old - a2_new
        self.targetValue = self.targetValue - a1_diff - a2_diff + (a1_old**2 - a1_new**2)*(np.dot(X[first_i].T, X[first_i]))*(y[first_i]**2)/2 + (
          (a2_old**2 - a2_new**2)*np.dot(X[second_i].T, X[second_i])*(y[second_i]**2)/2) + np.dot(X[first_i].T, X[second_i])*y[first_i]*y[second_i]*(a1_old*a2_old - a1_new*a2_new) + (
          y[first_i]*a1_diff*np.dot(X[first_i], np.sum(a*y*X, axis=0))) + y[second_i]*a2_diff*np.dot(X[second_i], np.sum(a*y*X, axis=0))
        a[first_i] = a1_new
        a[second_i] = a2_new
        if self.targetValue <= oldValue - self.tol or a2_new == a2_old:
          a[first_i] = a1_old
          a[second_i] = a2_old
          reselect_time += 1
          self.targetValue = oldValue
          continue
        else:
          break
      
      a1_new = a[first_i] +  y[first_i] * y[second_i] *(a[second_i] - a2_new)
      # update b
      if a1_new > 0 and a1_new < self.C:
        self.b = - self.E[first_i] - y[first_i] * self.k(X[first_i], X[first_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[first_i]) * (a2_new - a2_old) + self.b
      elif a2_new > 0 and a2_new < self.C:
        self.b = - self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b
      else:
        self.b = ((- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b) + (- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b)) / 2
      # update a
      # print(a[first_i], a[second_i])
      # print(a1_new, a2_new)
      # update W
      self.updateW(X, y, a)
      # print("w ", self.w, " b ", self.b)
      # update E
      self.updateE(X, y)
      n_iter += 1
      # check stop
      if self.checkStop(a, y):
        print("converge {0} not point not match KKT".format(n_iter))
        return
      elif (self.bestNotMatchKKT > self.notMatchKKT or self.notMatchKKT < X.shape[0] * self.KKTdelta) and self.targetValue - self.bestTargetValue + self.tol > 0:
        # good improve
        self.bestW = np.copy(self.w)
        self.bestB = np.copy(self.b)
        self.bestNotMatchKKT = int(self.notMatchKKT)
        self.bestTargetValue = float(self.targetValue)
        self.bestIter = n_iter
        self.bestA = np.copy(a)
      if self.verbose != 0 and n_iter % self.verbose == 0:
        print('finish {0}: best iter {3} best target value {1} and {2} point not match KKT'.format(n_iter, self.bestTargetValue, self.bestNotMatchKKT, self.bestIter))

    print("best targetValue {0} best not not KKTCnt {1} best Iter {2}".format(float(self.bestTargetValue), self.bestNotMatchKKT, self.bestIter))
    print('finish')
    self.b = self.bestB # rollback to best
    self.w = self.bestW
    

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

  def storeWBA(self):
    np.savetxt('final_smo_b.txt', self.b)
    np.savetxt('final_smo_w.txt', self.w)
    np.savetxt('final_smo_w.txt', self.bestA)
