import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


data,labels = make_blobs(n_samples = 1000, centers = 4, n_features = 2, random_state = 0)
plt.scatter(data[:,0], data[:,1], c = labels)
plt.show()

labels_orig = labels
labels = np.mod(labels_orig, 2)
plt.scatter(data[:,0], data[:,1], c = labels)
plt.show()