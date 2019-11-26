## 代码运行

#### 文件作用阐明

压缩包由存储训练集 csv 的 data 文件夹，包含自己搭建的 smo 算法 svm 模型的 python 包 mySVM，两个可执行 python 脚本文件，以及保存好模型训练结果的五个 txt 文件。

- txt 文件：保存有两次训练的结果，以 _no_test 结尾的是没有划分测试集所有样本做训练集训练完成之后的 svm 划分平面的 w、b，剩下三个文件是按 8 : 2 比例划分训练集与测试集训练之后的 svm 划分平面的 w、b以及 svm 目标函数的对偶问题的解 a 的值，这里保存 a 的值是为了以防课上效果不好，可以进行模型微调。
- load_txt_svm.py：单纯加载 txt 中训练好的 w、b，然后进行预测。
- smo.py: 训练模型的脚本。
- mySVM: 模型包，final_smo 与 gpu_smo 两者之间的区别在于 gpu_smo 调用了 cupy 来尝试加速迭代，而 final_smo 就是纯 numpy 计算。linear _svm 是尝试用割平面法求 svm 分割平面但失败了的代码。

#### 文件运行说明

- python load_txt_svm.py ：直接调用训练好的结果进行预测，并将预测结果保存在 data 文件夹下的 predict.csv 中。
- python smo.py: 重新训练模型。
- 所用到的包：
  - numpy、cupy：计算用，无 cupy 仍可调用 final_smo 模型（只需将 smo.py 中的模型换成 FinalSMO）。
  - pandas: 用于读取数据以及存储预测结果。
  - sklearn: 用于数据预处理以及模型评估。

## 代码实现说明

#### 数据预处理

考虑序数类型特征以及标称类型特征距离度量方式不是简单的欧式距离，故直接投入 svm 中进行训练可能会干扰训练的结果。

根据调用 sklearn.svm.linearnSVC 测试的结果来看（测试代码未展示在上传的压缩包中），决定不选用序数特征，因为投入了序数特征，sklearn 的 svm 都不能够收敛拟合了。

而对于标称特征，最好的处理方式是进行独热码，但考虑到训练集中标称特征较多，且单个特征的取值也有多种，直接所有独热码会导致数据维数上升很多。故采用折中的方式，先进行卡方相关性计算，然后选取相关性最强的特征进行独热码编码。

对于比率特征，观察到各特征的取值范围差别很大（考虑到可能是不同的量纲的数据），故进行了标准化处理。

对应实现代码如下：

```python
ratio_feature = ['x1', 'x3', 'x10', 'x11','x12']
ordinal_feature = ['x4']
nominal_feature = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
y = data['label']
ratio_x = data.drop(['label'], axis=1)[ratio_feature]
scal_x = StandardScaler().fit_transform(ratio_x) # 标准化比率特征
nominal_x = data.drop(['label'], axis=1)[nominal_feature]
ordinal_x = data.drop(['label'], axis=1)[ordinal_feature]
select_nominal = SelectKBest(chi2, k=1).fit_transform(nominal_x, y) # 选取卡方相关性最强的特征
enc = OneHotEncoder(categories='auto')
one_hot_nominal = pd.DataFrame(enc.fit_transform(select_nominal).toarray())
x = pd.concat([pd.DataFrame(scal_x, columns=ratio_feature), one_hot_nominal], axis=1) # 最后的训练集中不加入序数特征
```

#### 调用模型

```python
train_x, eval_x, train_y, eval_y = train_test_split(x,y, test_size=0.2) # 调用 sklearn train_test_split 来随机划分训练集与验证集
clf = GPUSMO(tol=0.1, max_iter=5000, random_seed=0, verbose=100)
clf.fit(np.array(train_x), np.array(train_y).reshape((train_x.shape[0], 1))) # 模型训练
clf.storeWB() # 保存训练的结果
eval_predict = clf.predict(np.array(eval_x)) # 预测验证集
```

#### 模型评估

```python
# for train_index, eval_index in sk.split(train_x, train_y):
#   clf = GPUSMO(tol=0.00001, max_iter=5000)
#   train_set = train_x.iloc[train_index]
#   eval_set = train_x.iloc[eval_index]
#   train_label = train_y.iloc[train_index]
#   clf.fit(train_set, train_label)
#   eval_predict[eval_index] = clf.predict(eval_set).reshape((eval_index.shape[0] ,1))
# 曾进行过模型五折交叉验证，但在大数据集上耗时过长，不推荐故注释掉（迭代 20000 次要一个多小时）
print('ac is', accuracy_score(eval_y, eval_predict))
print('f1 score is ', f1_score(eval_y, eval_predict))
```

#### 模型初始化

```python
def __init__(self, C = 1.0, tol = 0.0001, max_iter = 1000, verbose = 0, random_seed = 0, KKTdelta = 0.01, w=None, b = 0):
    self.C = C # 惩罚系数 C
    self.tol = tol # 若在 tol 精确度下满足所有点符合 KKT 条件则终止迭代
    self.max_iter = max_iter # 最大迭代数
    self.E = None # 记录每一个 x 的模型预测值与真实 y 之间的差
    self.w = w
    self.b = b # wx + b = 0
    self.verbose = verbose # 每进行多少次迭代输出一次当前迭代最优解
    self.random = np.random.RandomState(seed = random_seed) # 用于各种随机
    self.KKTdelta = KKTdelta # 模型接受 KKTdelta * n_smaple 个点不满足 KKT 条件
    self.failKKTList = [] # 不满足 KKT 的点的集合
    self.failKKTBorderList = [] # 处在间隔边界中，且不满足 KKT 的点的集合
    self.targetValue = 0 # 目标函数的值
    self.bestW = 0 # 最优解对应的 w
    self.bestB = 0
    self.bestTargetValue = 0
    self.bestNotMatchKKT = 0
    self.bestA = None
    self.bestIter = 0
    self.bestE = None # 保存最有解的一切
    self.notMatchKKT = 0 # 不满足 KKT 的点的个数
```

#### 模型预测函数

```python
def g(self, x):
    """
      g(x) > 0, label is 1; g(x) < 0, label is 0
      Parameter
      -------
      x : np.array((1, n))
    """
    return np.dot(self.w, x) + self.b
```

#### 模型判断终止条件

```python
def checkStop(self, a, y):
    """
    check all KKT with tolerance
    consider singularity, if 99% points match KKT is ok
    就是简单的遍历 a 判断满不满足 KKT
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
```

#### 模型训练函数

```python
def fit(self, X, y):
    """
    Parameters
    ------------
    X: np.array((n_samples, n_features)) the training data
    y: np.array((n_samples, 1)) the training label
    """
    # 初始化各个变量
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
      # 模型选点，代码解释在下面，在选点的过程中会计算 a1_new a2_new 并更新 a
      # 更新 b
      if a1_new > 0 and a1_new < self.C:
        self.b = - self.E[first_i] - y[first_i] * self.k(X[first_i], X[first_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[first_i]) * (a2_new - a2_old) + self.b
      elif a2_new > 0 and a2_new < self.C:
        self.b = - self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b
      else:
        self.b = ((- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b) + (- self.E[second_i] - y[first_i] * self.k(X[first_i], X[second_i]) * (a1_new - a1_old) - y[second_i] * self.k(X[second_i], X[second_i]) * (a2_new - a2_old) + self.b)) / 2
      # update W
      self.updateW(X, y, a)
      # update E
      self.updateE(X, y)
      n_iter += 1
      # 检查是否应该停止迭代
      if self.checkStop(a, y):
        print("converge {0} not point not match KKT".format(n_iter))
        return
      elif (self.bestNotMatchKKT > self.notMatchKKT or self.notMatchKKT < X.shape[0] * self.KKTdelta) and self.targetValue - self.bestTargetValue + self.tol > 0:
        # 如果本次迭代不满足 KKT 条件的点数未增多或不超过可接受的阈值（n_sample * KKTdelta）,且目标函数有下降（targetValue 为目标函数的负值），则接受该次迭代结果为当前最优
        self.bestW = np.copy(self.w)
        self.bestB = np.copy(self.b)
        self.bestNotMatchKKT = int(self.notMatchKKT)
        self.bestTargetValue = float(self.targetValue)
        self.bestIter = n_iter
        self.bestA = np.copy(a)
      # 依据 verbose 输出迭代信息
      if self.verbose != 0 and n_iter % self.verbose == 0:
        print('finish {0}: best iter {3} best target value {1} and {2} point not match KKT'.format(n_iter, self.bestTargetValue, self.bestNotMatchKKT, self.bestIter))
	# 结束迭代，仍未达到拟合，选择最优的 w b 覆盖当前的 w b
    print("best targetValue {0} best not not KKTCnt {1} best Iter {2}".format(float(self.bestTargetValue), self.bestNotMatchKKT, self.bestIter))
    print('finish')
    self.b = self.bestB # rollback to best
    self.w = self.bestW
```

##### 模型选择优化点

```python
	  # 选点的策略：第一个点需要选择不满足 KKT 条件的点，优先选择间隔边界上不满足的点，如果没有则选择其他不满足 KKT 条件的点。对于第二个点，先按照是得|E1 - E2|最大的标准进行选择，但如果选择的两个点不能够使目标函数的值下降或者 a2_new 和 a2_old 一样则重新选点。
      reselect_time = 0 # 记录重新选点的次数
      while True:
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
          second_i = self.random.randint(0, a.shape[0])
          while first_i == second_i or np.linalg.norm(X[first_i] - X[second_i]) == 0:
            second_i = self.random.randint(0, a.shape[0]) # select a random a2
        # 计算 a2_new 剪切之前的值
        a2_newunc = a[second_i] + y[second_i] * (self.E[first_i] - self.E[second_i]) / np.linalg.norm(X[first_i] - X[second_i])
        # 计算 H 和 L
        if y[first_i] == y[second_i]:
          L = max(0, a[second_i] + a[first_i] - self.C)
          H = min(self.C, a[second_i] + a[first_i])
        else:
          L = max(0, a[second_i] - a[first_i])
          H = min(self.C, self.C + a[second_i] - a[first_i])
        # 计算 a2_new
        if a2_newunc > H:
          a2_new = H
        elif a2_newunc < L:
          a2_new = L
        else:
          a2_new = a2_newunc
        # 计算 a1_new
        a1_new = a[first_i] +  y[first_i] * y[second_i] *(a[second_i] - a2_new)
        # 计算更新之后的目标函数
        oldValue = self.targetValue
        a1_old = float(a[first_i])
        a2_old = float(a[second_i]) # TODO:solve this numpy bug
        a[first_i] = 0
        a[second_i] = 0
        # 考虑到，在大数据上每次重新计算目标函数值开销很大，其实只需要考虑 a1 a2 改变对目标函数的影响
        a1_diff = a1_old - a1_new
        a2_diff = a2_old - a2_new
        temp = np.sum(a*y*X, axis=0)
        # 这里计算的是对偶问题目标函数的负（刚好弄反了，所以下面要求的是 targetValue 值上升）
        self.targetValue = self.targetValue - a1_diff - a2_diff + (a1_old**2 - a1_new**2)*(np.dot(X[first_i].T, X[first_i]))*(y[first_i]**2)/2 + (
          (a2_old**2 - a2_new**2)*np.dot(X[second_i].T, X[second_i])*(y[second_i]**2)/2) + np.dot(X[first_i].T, X[second_i])*y[first_i]*y[second_i]*(a1_old*a2_old - a1_new*a2_new) + (
          y[first_i]*a1_diff*np.dot(X[first_i], temp)) + y[second_i]*a2_diff*np.dot(X[second_i], temp)
        a[first_i] = a1_new
        a[second_i] = a2_new
        # 如果目标函数没有下降或 a2_new = a2_old 则重新选点
        if self.targetValue <= oldValue - self.tol or a2_new == a2_old:
          a[first_i] = a1_old
          a[second_i] = a2_old
          reselect_time += 1
          self.targetValue = oldValue
          continue
        else:
          break
```

