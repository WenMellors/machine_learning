import pandas as pd
import numpy as np
import sklearn.svm as svm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('./data/svm_training_set.csv')

ratio_feature = ['x1', 'x3', 'x10', 'x11', 'x12']

train_y = data['label']
train_x = data.drop(['label'], axis=1)[ratio_feature]
scal_x = StandardScaler().fit_transform(train_x)

sk = StratifiedKFold(n_splits=5)
eval_predict = np.zeros((scal_x.shape[0], 1))
for train_index, eval_index in sk.split(scal_x, train_y):
  clf = svm.LinearSVC(random_state=0, max_iter=5000)
  train_set = scal_x[train_index]
  eval_set = scal_x[eval_index]
  train_label = train_y.iloc[train_index]
  clf.fit(train_set, train_label)
  eval_predict[eval_index] = clf.predict(eval_set).reshape((eval_index.shape[0] ,1))

print('f1 score is ', f1_score(train_y, eval_predict))