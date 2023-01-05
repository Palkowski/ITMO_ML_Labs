import numpy as np
import pandas as pd

data = pd.read_csv('pulsar_stars.csv')
simple_data = data[((data.TG == 0) & (data.MIP >= 95.8984375) & (data.MIP <= 96.4140625)) | (
        (data.TG == 1) & (data.MIP >= 77.4921875) & (data.MIP <= 83.7734375))]

print('number of rows:', simple_data.shape[0])
print('mean for MIP:', round(simple_data.MIP.mean(), 3))
