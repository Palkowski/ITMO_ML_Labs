import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

print('# Task 1')

X = pd.read_csv('13_36.csv', header=None)  # 60 objects (rows) and 10 features (columns)
pca = PCA(n_components=2, svd_solver='full')
X_pca = pca.fit_transform(X)
print(X_pca.shape, '\n', X_pca)
print('answer 1:', round(X_pca[0][0], 3))
print('answer 2:', round(X_pca[0][1], 3))
print('answer 3:', round(pca.explained_variance_ratio_.sum(), 3))
print('answer 4:', 4)  # adjust n_components=4 parameter of PCA() and look at answer 3
print('answer 5:', int(pca.explained_variance_ratio_.sum() * X.shape[1]))

print('# Task 2')

Z = np.genfromtxt('X_reduced_408.csv', delimiter=';')
phi = np.genfromtxt('X_loadings_408.csv', delimiter=';').T
ans = np.matmul(Z, phi)
plt.imshow(ans, interpolation='nearest')
plt.show()
print('answer: 4')
