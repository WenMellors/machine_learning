import pandas as pd
import numpy as np
from randomForest import RandomForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

train_x = pd.read_csv('./data/x_train.csv')
train_y = pd.read_csv('./data/y_train.csv')

train_data = pd.merge(train_x, train_y)
forest = RandomForest(depth=5, min_sample_leaf=13, min_gini=0.001, n_tree = 20)
train_set, eval_set = train_test_split(train_data, test_size=0.2)
forest.load()
result = forest.predict(eval_set)

print('ac ', accuracy_score(eval_set['label'], result))
print('precision ', precision_score(eval_set['label'], result))
print('recall ', recall_score(eval_set['label'], result))
print('f1_score ', f1_score(eval_set['label'], result))