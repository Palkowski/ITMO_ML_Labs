import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('candy-data.csv')

train_data = data[(data.competitorname != 'Charleston Chew') & (data.competitorname != 'Dum Dums')]
train_predictor = np.array(train_data.drop(columns=['competitorname', 'winpercent', 'Y']))
train_response = np.array(train_data.winpercent)

test_data = data[(data.competitorname == 'Charleston Chew') | (data.competitorname == 'Dum Dums')]
test_predictor = np.array(test_data.drop(columns=['competitorname', 'winpercent', 'Y']))

reg = LinearRegression().fit(train_predictor, train_response)
print('winpercent for Charleston Chew, Dum Dums:\n', reg.predict(test_predictor))
print('winpercent for [0, 0, 0, 1, 0, 1, 1, 0, 1, 0.885, 0.649]:\n',
      reg.predict(np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0.885, 0.649]).reshape(1, -1)))
