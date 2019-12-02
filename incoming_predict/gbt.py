import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('./data/train.csv')
train['income']  = train['income'].map(lambda item : 1 if item == '>50K' else -1 ) # >50K is 1 and <=50K is 0 
test = pd.read_csv('./data/test.csv').drop(['Unnamed: 15'], axis=1)
data = train.append(test, ignore_index=True, sort=True)
is_train = ~data['income'].isnull()
ratio_feature = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'age']
nominal_feature = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

train_label = data[is_train]['income']
ratio_data = data[ratio_feature]
ratio_data = pd.DataFrame(StandardScaler().fit_transform(ratio_data), columns=ratio_feature)
# label encode
nominal_data = data[nominal_feature]
encode_nominal = pd.DataFrame()
for i in nominal_feature:
  encode_nominal[i] = LabelEncoder().fit_transform(nominal_data[i])

# one hot encode
# select = SelectKBest(chi2, k=4)
# select.fit(encode_nominal[is_train], train_label)
# select_nominal = select.transform(encode_nominal)
enc = OneHotEncoder(categories='auto')
one_hot_nominal = pd.DataFrame(enc.fit_transform(encode_nominal).toarray())
train_data = pd.concat([ratio_data[is_train], one_hot_nominal[is_train]], axis=1)
predict_data = pd.concat([ratio_data[~is_train], one_hot_nominal[~is_train]], axis=1)
# train_data = ratio_data[is_train]
# predict_data = ratio_data[~is_train]
# training
print(train_data.shape)
skf = StratifiedKFold(n_splits=5)
eval_result = np.zeros((train_data.shape[0], 1))
predict_label = np.zeros((predict_data.shape[0], 2))
i = 0
for train_index, eval_index in skf.split(train_data, train_label):
  print('start ', i)
  i += 1
  split_train, split_train_label = train_data.iloc[train_index], train_label.iloc[train_index]
  eval_data, eval_label = train_data.iloc[eval_index], train_label.iloc[eval_index]
  classifier = GradientBoostingClassifier(n_estimators= 500)
  classifier.fit(split_train, split_train_label)
  eval_result[eval_index] = classifier.predict(eval_data).reshape(eval_data.shape[0], 1)
  predict_label += classifier.predict_proba(predict_data).reshape(predict_data.shape[0], 2)

# test accuracy 
print('ac ', accuracy_score(train_label, eval_result))
print('precision ', precision_score(train_label, eval_result))
print('recall ', recall_score(train_label, eval_result))
print('f1_score ', f1_score(train_label, eval_result))

# predict
predict_label = np.argmax(predict_label, axis=1)
predict_label = pd.DataFrame(predict_label, columns=['income'])
predict_label = pd.DataFrame(predict_label['income'].map(lambda item : '>50K' if item == 1.0 else '<=50K' ), columns=['income'])
predict_label.to_csv('./data/submission.csv', index_label='id')