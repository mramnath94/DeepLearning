import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

data = sklearn.datasets.load_breast_cancer()
pd_data = pd.DataFrame(data.data, columns = data.feature_names)
pd_data['class'] = data.target
X = pd_data.drop('class', axis = 1)
Y = pd_data['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
X_binarised_train = X_train.apply(pd.cut, bins = 2, labels = [1, 0])
X_binarised_test = X_test.apply(pd.cut, bins = 2, labels = [1, 0])