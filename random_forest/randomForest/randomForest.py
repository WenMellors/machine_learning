import numpy as np
import pandas as pd
from .baseTree import BaseTree, Node
class RandomForest:
  def __init__(self, depth=1, min_sample_leaf=13, min_gini=0.001, n_tree = 5):
    self.depth=depth
    self.min_sample_leaf = min_sample_leaf
    self.min_gini = min_gini
    self.n_tree = n_tree
    self.trees = []
  
  def fit(self, X):
    """
      X: pd.Dataframe and contains label
    """
    # check X contains one value feature
    features = X.drop(['index', 'label'], axis=1).columns
    drop = []
    for feature in features:
      if pd.unique(X[feature]).shape[0] == 1:
        drop.append(feature)
    X = X.drop(drop, axis=1)
    for i in range(self.n_tree):
      new_tree = BaseTree(depth=self.depth, min_sample_leaf=self.min_sample_leaf, min_gini=self.min_gini)
      sample = X.sample(n=X.shape[0], replace=True)
      new_tree.fit(sample)
      print('finish build {} tree'.format(i))
      self.trees.append(new_tree)

  def predict(self, X):
    result = np.zeros((X.shape[0], 1))
    print('start predict')
    for i in range(X.shape[0]):
      for tree in self.trees:
        result[i] += tree.treeRoot.predict(X.iloc[i])
    print('finish')
    return np.apply_along_axis(lambda x: 1 if x >= 0 else -1, axis=1, arr=result).reshape(X.shape[0],1)
  
  def save(self):
    out = open('forest_save.txt', 'w')
    i = 0
    for tree in self.trees:
      out.write('new tree\n')
      i += 1
      # tree save bfs
      nowLayer = []
      nextLayer = []
      nowLayer.append(tree.treeRoot)
      no = 0
      numberDict = {}
      while True:
        while len(nowLayer) != 0:
          head = nowLayer.pop(0)
          if numberDict.get(head) == None:
            numberDict[head] = no
            no += 1
          if head.left != None:
            numberDict[head.left] = no
            no += 1
            nextLayer.append(head.left)
          if head.right != None:
            numberDict[head.right] = no
            no += 1
            nextLayer.append(head.right)
          out.write('{} {} {} {} {} {} \n'.format(numberDict[head], head.key, head.value, head.mode, numberDict[head.left] if head.left != None else 'None', numberDict[head.right] if head.right != None else 'None'))
        # print a new line
        if len(nextLayer) == 0:
          break
        else:
          nowLayer = nextLayer.copy()
          nextLayer = []
      # end tree save
      print('finish ', i)
    out.close()

  def load(self):
    source = open('forest_save.txt', 'r')
    lines = source.readlines()
    for i in range(self.n_tree):
      self.trees.append(BaseTree(depth=self.depth, min_sample_leaf=self.min_sample_leaf, min_gini=self.min_gini))
    i = 1 # skip 0 line
    for tree in self.trees:
      nodes = {}
      nodes[0] = tree.treeRoot
      while i < len(lines) and lines[i] != 'new tree\n':
        strs = lines[i].split(' ')
        no = int(strs[0])
        # insert by father
        tempNode = nodes[no]
        tempNode.key = strs[1]
        tempNode.value = int(strs[2])
        tempNode.mode = strs[3]
        if strs[4] != 'None':
          newNode = Node()
          tempNode.left = newNode
          nodes[int(strs[4])] = newNode
        if strs[5] != 'None':
          newNode = Node()
          tempNode.right = newNode
          nodes[int(strs[5])] = newNode
        i += 1
      # end build tree
      print('finish buld tree')
      if i < len(lines):
        break
      i += 1




    
    
