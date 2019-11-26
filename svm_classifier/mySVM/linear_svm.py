import numpy as np
import cupy as cp
class LenearSVM:
  def __init__(self, C = 1.0, tol = 0.0001, max_iter = 1000, verbose = 0, random_seed = 0, w=None, maxWorkSet = 100):
    self.C = C # penalty parameter C of the error term.
    self.tol = tol # tolerance for stopping criteria.
    self.max_iter = max_iter # max iteration
    self.w = w
    self.workSet = []
    self.verbose = verbose
    self.random = np.random.RandomState(seed = random_seed)
    self.b = None
    self.A = None
    self.maxWorkSet = maxWorkSet

  def fit(self, X, y):
    n_iter = 0
    n = X.shape[0]
    # inital
    self.w = cp.zeros((1, X.shape[1]))
    self.workSet = []
    tempC = cp.ones((X.shape[0], 1)) # when init tempC is ones
    tempXY = y*X # use to calculate b
    self.workSet.append(tempC)
    self.A = cp.sum(tempC*tempXY, axis=0)
    self.b = cp.linalg.norm(tempC, ord=1) / n
    n_iter += 1
    while n_iter < self.max_iter:
      # solve qp, I just use first derivative equal 0 to calculate w
      # may be wrong, but I don't know how to solve this qp problem
      # A * w = b
      cp.linalg.solve

  