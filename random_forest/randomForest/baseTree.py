import numpy as np
import pandas as pd
class Node:
  """
    binary tree node
  """
  def __init__(self):
    """
      key: feature name in this node
      value: this node split value
      mode: nominal indicate the split feature is nominal
            leaf use for indicate this is leaf node and value will be the node label
    """
    self.key = 'leaf'
    self.value = 0
    self.left = None
    self.right = None
    self.mode = ''

  def predict(self, x):
    """
      x: pandas dataframe a row
    """
    tempRoot = self
    while True:
      if tempRoot.mode == 'nominal':
        if tempRoot.value == x[tempRoot.key]:
          tempRoot = tempRoot.left
        else:
          tempRoot = tempRoot.right
      else:
        if tempRoot == None:
          print('error in predict')
          exit(-1)
        return tempRoot.value
  
class BaseTree:
  """
    基分类器，决策树
  """
  def __init__(self, depth=1, min_sample_leaf=13, min_gini=0.001):
    self.depth = depth
    self.min_sample_leaf = min_sample_leaf
    self.min_gini = min_gini
    self.treeRoot = Node()
    self.ratio_feature = []
  
  def fit(self, X):
    """
      X: pd.DataFrame, and should contains 'label'
      ratio_feature: the ratio feature's name
    """
    # init
    self.createTree(0, self.treeRoot, X)
  
  def predict(self, X):
    result = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
      result[i] = self.treeRoot.predict(X.iloc[i])
    return result

  def gini(self, y):
    """
     calculate gini of data set
    """
    pos = y[y == 1.0].shape[0]
    neg = y[y == -1.0].shape[0]
    p = np.divide(pos, pos+neg)
    return 2*p*(1-p)


  def createTree(self, depth, root, X):
    """
      depth: now depth
      root: new node append to
      X: this node data set
    """
    if depth == self.depth or X.shape[0] < self.min_sample_leaf or X.shape[1] <= 2 or self.gini(X['label']) < self.min_gini:
      # the mode in y is the node's label
      root.mode = 'leaf'
      root.value = X['label'].mode()[0]
    else:
      features = X.drop(['index', 'label'], axis=1).columns
      drop = []
      for feature in features:
        if pd.unique(X[feature]).shape[0] == 1:
          drop.append(feature)
      X = X.drop(drop, axis=1)
      features = X.drop(['index', 'label'], axis=1).columns
      selectN = np.log2(features.shape[0])
      select_feature = np.random.choice(features, size=np.ceil(selectN).astype(int))
      min_feature_gini = 1.2
      split_feature = None
      split_feature_value = 0
      for feature in select_feature:
        feature_value = pd.unique(X[feature])
        min_gini = 1.2
        split_value = 0
        for value in feature_value:
          left_part = X[X[feature] == value] # eqaul value
          right_part = X[X[feature] != value]
          value_gini = np.divide(left_part.shape[0], X.shape[0]) * self.gini(left_part['label']) + np.divide(right_part.shape[0], X.shape[0]) * self.gini(right_part['label'])
          if min_gini >= value_gini:
            min_gini = value_gini
            split_value = value
        if min_gini <= min_feature_gini:
          min_feature_gini = min_gini
          split_feature = feature
          split_feature_value = split_value
        # next feature
      left_part = X[X[split_feature] == split_feature_value].drop([split_feature], axis = 1)
      right_part = X[X[split_feature] != split_feature_value]
      if pd.unique(right_part[split_feature]).shape[0] == 1:
        # should drop
        right_part = right_part.drop([split_feature], axis=1)
      root.value = split_feature_value
      root.key = split_feature
      root.mode = 'nominal'
      root.left = Node()
      root.right = Node()
      self.createTree(depth+1, root.left, left_part)
      self.createTree(depth+1, root.right, right_part)
    
