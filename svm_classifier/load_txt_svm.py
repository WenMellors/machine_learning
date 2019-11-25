import pandas as pd
import numpy as np
from mySVM import SMO, RandomSMO, FinalSMO, GPUSMO
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2, SelectKBest

data = pd.read_csv('./data/svm_training_set.csv')

ratio_feature = ['x1', 'x3', 'x10', 'x11','x12'] # x10, x11 is almost 0
ordinal_feature = ['x4']
nominal_feature = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
train_y = data['label']
ratio_x = data.drop(['label'], axis=1)[ratio_feature]
scal_x = StandardScaler().fit_transform(ratio_x)
nominal_x = data.drop(['label'], axis=1)[nominal_feature]
ordinal_x = data.drop(['label'], axis=1)[ordinal_feature]
select_nominal = SelectKBest(chi2, k=1).fit_transform(nominal_x, train_y) # select nominal var according to chi2
enc = OneHotEncoder(categories='auto')
one_hot_nominal = pd.DataFrame(enc.fit_transform(select_nominal).toarray())
train_x = pd.concat([pd.DataFrame(scal_x, columns=ratio_feature), one_hot_nominal], axis=1)

sk = StratifiedKFold(n_splits=5)
eval_predict = np.zeros((scal_x.shape[0], 1))
w = np.loadtxt('final_smo_w.txt')
b = np.loadtxt('final_smo_b.txt')
clf = GPUSMO(tol=0.1, max_iter=1000, random_seed=0, verbose=100, w=w, b=b)
eval_predict = clf.predict(np.array(train_x))

print('ac is', accuracy_score(train_y, eval_predict))
print('f1 score is ', f1_score(train_y, eval_predict))