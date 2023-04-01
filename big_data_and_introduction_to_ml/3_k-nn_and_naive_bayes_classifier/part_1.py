import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]

train_predictor = np.array(data.drop(columns=['id', 'Class']))
train_response = np.array(data.Class)

neigh = KNeighborsClassifier(n_neighbors=3, p=2)
neigh.fit(train_predictor, train_response)

print('for k = 3, p = 2:')
print('K-neighbors:', neigh.kneighbors([[30, 30]]))  # distances and indexes
print('Class predict:', neigh.predict([[30, 30]]))  # class prediction

print('\n')

neigh1 = KNeighborsClassifier(n_neighbors=3, p=1)
neigh1.fit(train_predictor, train_response)

print('for k = 3, p = 1:')
print('K-neighbors:', neigh1.kneighbors([[30, 30]]))
print('Class predict:', neigh1.predict([[30, 30]]))

plt.grid()
plt.axis('equal')
plt.scatter(data_class_0['X'], data_class_0['Y'])
plt.scatter(data_class_1['X'], data_class_1['Y'])
plt.scatter(30, 30)
plt.show()
