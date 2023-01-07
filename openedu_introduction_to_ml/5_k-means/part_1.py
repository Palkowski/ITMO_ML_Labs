from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data_copy = data
training_data = np.array(data.drop(columns=['Object', 'Cluster']))

data_c0 = data[data.Cluster == 0]
data_c1 = data[data.Cluster == 1]
data_c2 = data[data.Cluster == 2]

centroids = np.array([[14.0, 9.75], [14.25, 15.25], [5.29, 9.57]])
kms = KMeans(n_clusters=3, init=centroids, max_iter=100, n_init=1)
kms.fit(training_data)
predicted_clusters = kms.predict(training_data)

data_copy.Cluster = predicted_clusters

print('Predicted clusters:\n', data_copy)

data_pred_c0 = data[data_copy.Cluster == 0]
data_pred_c1 = data[data_copy.Cluster == 1]
data_pred_c2 = data[data_copy.Cluster == 2]

distances = kms.transform(np.array(data_pred_c0.drop(columns=['Object', 'Cluster'])))
print('mean distance between objects of class 0 and centriod 0:', np.mean(distances.T[0]))

# plots
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

ax1.grid()
ax2.grid()

ax1.axis('equal')
ax2.axis('equal')

ax1.set_title('initial data')
ax2.set_title('predicted clusters')

ax1.scatter(data_c0.X, data_c0.Y)
ax1.scatter(data_c1.X, data_c1.Y)
ax1.scatter(data_c2.X, data_c2.Y)

ax2.scatter(data_pred_c0.X, data_pred_c0.Y)
ax2.scatter(data_pred_c1.X, data_pred_c1.Y)
ax2.scatter(data_pred_c2.X, data_pred_c2.Y)
# ax2.scatter(centroids.T[0], centroids.T[1], color='red')  # uncomment to see centroids
plt.show()
