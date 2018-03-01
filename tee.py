#pip.main(['install', 'scipy'])
#import pip
#TO WPISAC W KONSOLI NA DOLE PYTHOON CONSOLE
import sys
print(sys.version)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Generate Data - two means
mean1 = [np.random.randint(50), np.random.randint(50)]
mean2 = [np.random.randint(50), np.random.randint(50)]
#
# # Make diagonal covariance
cov = [[100,0], [0, 100]]
import	tensorflow	as	tf
#we	import	tensorflow
import	numpy	as	np
sess	=	tf.Session()
# start	a	new	Session	Object
x_data	=	np.array([[1.,2.,3.],	[3.,2.,6.]])
#	2x3	matrix
x	=	tf.convert_to_tensor(x_data,	dtype=tf.float32)

x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T

x = np.append(x1, x2)
y = np.append(y1, y2)

# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()

# Make KMeans model
X = np.array(list(zip(x, y)))
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(labels)
print(centroids)

colors = ["g.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=150, zorder=10)

plt.show()
