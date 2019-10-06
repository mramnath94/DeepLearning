import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Generating a random 100*1 array
a = np.random.random([100,1])
plt.plot(a)
plt.show(a)
print(np.mean(a))
print(np.std(a))

#Standardising the i/p which would change the mean to 0 and standard deviation to 1
scaler = StandardScaler()
scaler.fit(a)
aT = scaler.transform(a)
plt.plot(aT)
plt.show(aT)
print(np.mean(aT))
print(np.std(aT))

#the plots of a and aT would look similar even though their mean and SD are different