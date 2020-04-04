import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.datasets import make_blobs

np.random.seed(0)

N=100
M=200
a = np.random.randn(N, M)
b = np.random.randn(N, M)
c = np.zeros((N, M))

%%time
for i in range(N):
  for j in range(M):
    c[i, j] = a[i, j]+b[i, j]


%%time
c = a + b