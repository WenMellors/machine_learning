import pandas as pd
import numpy as np
from mySVM import SMO, RandomSMO, FinalSMO
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('./data/svm_training_set.csv')

ratio_feature = ['x1', 'x3', 'x10', 'x11', 'x12']

train_y = data['label'].iloc[0:500]
train_x = data.drop(['label'], axis=1)[ratio_feature].iloc[0:500]
scal_x = StandardScaler().fit_transform(train_x)

sk = StratifiedKFold(n_splits=5)
eval_predict = np.zeros((scal_x.shape[0], 1))
clf = FinalSMO(tol=0.1, max_iter=5000, random_seed=0, verbose=500)
clf.fit(scal_x, np.array(train_y).reshape((train_x.shape[0], 1)))
eval_predict = clf.predict(scal_x)

# for train_index, eval_index in sk.split(scal_x, train_y):
#   clf = SMO(tol=0.00001, max_iter=5000)
#   train_set = scal_x[train_index]
#   eval_set = scal_x[eval_index]
#   train_label = train_y.iloc[train_index]
#   clf.fit(train_set, train_label)
#   eval_predict[eval_index] = clf.predict(eval_set).reshape((eval_index.shape[0] ,1))
print('ac is', accuracy_score(train_y, eval_predict))
print('f1 score is ', f1_score(train_y, eval_predict))