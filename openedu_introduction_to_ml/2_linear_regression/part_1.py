import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

y = np.array(data.Y)
x = np.array(data.X)

reg = LinearRegression().fit(x.reshape(-1, 1), y)

print('mean X:', np.mean(x))
print('mean Y:', np.mean(y))
print('slope (theta_1):', reg.coef_)
print('intercept (theta_0):', reg.intercept_)
print('R^2:', reg.score(x.reshape(-1, 1), y))

plt.grid()
plt.scatter(x, y)
plt.plot(x, reg.predict(x.reshape(-1, 1)))
plt.show()
