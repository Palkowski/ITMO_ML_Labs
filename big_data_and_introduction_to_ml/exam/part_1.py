import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# def psi(x1, x2):
#     return 1 / (1 + np.exp(-(3.610 + 1.275 * x1 + 1.965 * x2)))
# print(psi(-3, 3))

df = pd.read_csv('report.csv')
X = df.drop(columns='TARGET')
X = np.array((X - X.min()) / (X.max() - X.min()))
print(X)

y = np.array(df['TARGET'])
print(y)

clf = LogisticRegression(random_state=2019, solver='lbfgs')
clf.fit(X, y)

star = np.array([[0.254, 0.19, 0.939, 0.624, 0.935, 0.875, 0.151, 0.312]])

print(clf.predict(star))
print(clf.predict_proba(star))

knnclf = KNeighborsClassifier()
knnclf.fit(X, y)

print(knnclf.kneighbors(X=star, n_neighbors=1, return_distance=True))
