## 文件说明

* my_random_forest.py: 调用手写的随机森林模型进行训练预测的脚本。
* predict.py: 加载之前训练保存下来的随机森林参数（forest_save.txt 文件里），去进行预测的脚本。
* randomForest：文件夹是自己写的随机森林包，里面 baseTree.py 里实现了cart 分类决策树（无剪枝），randomForest.py 则是根据 cart 决策树实现的随机森林。
* 运行说明：数据放在 predict.py 脚本同目录的 data 文件夹里，具体的加载代码如下：

```python
train_x = pd.read_csv('./data/x_train.csv')
train_y = pd.read_csv('./data/y_train.csv')
```

​		相应的数据处理调用模型的代码如下：

```python
"""
predict.py
"""
train_data = pd.merge(train_x, train_y) # 将两个 Dataframe 按照 Index 连接起来
forest = RandomForest(depth=5, min_sample_leaf=13, min_gini=0.001, n_tree = 15) # 初始化模型，参数应保证 predict.py 与 my_random_forest.py 一致
train_set, eval_set = train_test_split(train_data, test_size=0.2) # 划分测试集与训练集
forest.load() # 加载之前的训练结果
result = forest.predict(eval_set) # 预测，结果为 np.array(n_samples, 1)
"""
my_random_forest.py
"""
train_data = pd.merge(train_x, train_y)
forest = RandomForest(depth=5, min_sample_leaf=13, min_gini=0.001, n_tree = 15)
train_set, eval_set = train_test_split(train_data, test_size=0.2)
forest.fit(train_set) # 训练模型，输入应为 pd.Dataframe 而且要将 train_x train_y 连接
result = forest.predict(eval_set)
forest.save() # 保存模型训练结果
```

## 随机森林模型实现

#### cart 决策树实现

实现代码为 randomForest/baseTree.py。cart 决策树基于 Node 节点类和 BaseTree 树类实现。

###### Node 类

```python
def __init__(self):
    """
      key: feature name in this node
      value: this node split value
      mode: nominal indicate the split feature is nominal
            leaf use for indicate this is leaf node and value will be the node label
    """
    self.key = 'leaf' # 标识当前节点分支的特征名
    self.value = 0 # 当前节点分支的特征对应的值
    self.left = None # 左子树节点
    self.right = None # 右子树节点
    self.mode = '' # 标识该节点是否是叶节点
```

节点通过 Key、value、mode 三个属性，可以在预测的时候确定该输入数据去哪一个子节点或者标签是什么。具体预测策略为：设输入数据为 x，若 mode != 'leaf'，则判断 x[key] 与 value 是否相等，相等则去左子节点，不等则去右子节点；若 mode == 'leaf'，则 x 的标签为 value。具体实现代码如下：

```python
def predict(self, x):
    """
      x: pandas dataframe a row
      实际预测的时候只需调用树根节点的该方法即可
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
```

###### BaseTree

BaseTree 主要实现决策树的生成。BaseTree 支持设置树的最大深度、叶节点的最小样本数、要分支的最小基尼指数值。因为我不需要我的基估计器有多强，所以设置这些参数来控制训练单棵树的时间。

而建树的过程采用递归构建的方式，具体递归函数的算法如下：

输入：当前深度 depth，当前节点 root，当前节点的数据集 X

1. 根据树初始化设置的最大深度、节点最小样本树、最小基尼指数判定是否结束当前递归。
2. 若未结束，设 X 特征数为 M，则从 M 个特征中随机选取 $log_2(M)$ 个作为候选的分支特征（这一步是给树的训练带来随机性，保证随机森林的性能）。对候选集内的每一个特征，遍历其的每一个值作为分支值，计算分支后的 gini 指数值。选取分治后 gini 指数值最小的那一个特征和其分支值作为当前节点的分支特征与值。
3. 依据分支特征与值划分 X，所有 X[分支特征] == 分支值的数据投入左子节点中，剩下的投入右子节点中。 
4. 分别递归构建左右子节点

实现代码如下：

```python
def createTree(self, depth, root, X):
    """
      depth: now depth
      root: new node append to
      X: this node data set
    """
    if depth == self.depth or X.shape[0] < self.min_sample_leaf or X.shape[1] <= 2 or self.gini(X['label']) < self.min_gini:
      # 判断是否停止分支，若停止，则选取样本集中的标签的众数为当前叶节点的标签。
      root.mode = 'leaf'
      root.value = X['label'].mode()[0]
    else:
      # 先筛去 X 中的只有单一取值的特征
      features = X.drop(['index', 'label'], axis=1).columns
      drop = []
      for feature in features:
        if pd.unique(X[feature]).shape[0] == 1:
          drop.append(feature)
      X = X.drop(drop, axis=1)
      features = X.drop(['index', 'label'], axis=1).columns
      selectN = np.log2(features.shape[0])
      # 随机选取 log2(M) 个特征作为候选特征
      select_feature = np.random.choice(features, size=np.ceil(selectN).astype(int))
      min_feature_gini = 1.2
      split_feature = None
      split_feature_value = 0
      # 两层遍历，获得分支后 gini 最小的分支特征及其值
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
      # 将样本集划分为两个部分
      left_part = X[X[split_feature] == split_feature_value].drop([split_feature], axis = 1)
      right_part = X[X[split_feature] != split_feature_value]
      if pd.unique(right_part[split_feature]).shape[0] == 1:
        # should drop
        right_part = right_part.drop([split_feature], axis=1)
      # 设置该节点的属性
      root.value = split_feature_value
      root.key = split_feature
      root.mode = 'nominal'
      root.left = Node()
      root.right = Node()
      # 递归构建左右子节点
      self.createTree(depth+1, root.left, left_part)
      self.createTree(depth+1, root.right, right_part)
```

#### 随机森林实现

RadomForest 类除了有与 BaseTree 相同的三个属性之外，还有 n_tree 标识该森林由多少棵树组成。在训练构建森林的时候，森林只负责 boostrapping sample 为每个树生成对应的输入数据集；而在预测的时候，只需要调用每棵树的根节点的 predict 方法进行预测，将预测结果统计起来，按照少数服从多数的原则得出最后的预测结果。

训练部分代码：

```python
 def fit(self, X):
    """
      X: pd.Dataframe and contains label
    """
    # 筛去 X 中单值的特征
    features = X.drop(['index', 'label'], axis=1).columns
    drop = []
    for feature in features:
      if pd.unique(X[feature]).shape[0] == 1:
        drop.append(feature)
    X = X.drop(drop, axis=1)
    for i in range(self.n_tree):
      new_tree = BaseTree(depth=self.depth, min_sample_leaf=self.min_sample_leaf, min_gini=self.min_gini)
      # 有放回的抽取样本生成新的输入数据集
      sample = X.sample(n=X.shape[0], replace=True)
      new_tree.fit(sample)
      print('finish build {} tree'.format(i))
      self.trees.append(new_tree)
```

预测部分的代码：

```python
  def predict(self, X):
    result = np.zeros((X.shape[0], 1))
    print('start predict')
    # 遍历待测数据的每一行，将其投入每一棵树中进行预测，并将预测结果 += 起来
    for i in range(X.shape[0]):
      for tree in self.trees:
        result[i] += tree.treeRoot.predict(X.iloc[i])
    print('finish')
    # 整合统计生成最后的标签
    return np.apply_along_axis(lambda x: 1 if x >= 0 else -1, axis=1, arr=result).reshape(X.shape[0],1)
```

剩下两个函数 save 与 load 主要是用于保存模型训练结果与加载训练结果，主要策略就是把每一棵树的节点信息存下来，具体保存格式如下：

```
new tree
节点编号 节点key 节点value 节点mode 左子节点编号 右子节点编号
节点编号 .....
....
```

new tree 标识下面的节点信息属于一棵新的树。

save 函数只需要采用 bfs 的策略遍历树，将每一个节点的信息输出即可，按遍历的先后顺序给节点分配编号。load 函数需要维护一个 {节点编号 -> 节点} 的字典，每读取一行节点信息，根据节点编号，从字典中取出当前节点，然后根据后面的信息赋值该节点的属性，如果左右节点编号不为 'None'（标识当前节点无子节点），则为子节点生成一个新的节点实例，插入字典之中。