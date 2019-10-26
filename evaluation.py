import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import scipy.cluster.vq as spcv
from mpl_toolkits.mplot3d import Axes3D

from subtractive_clustering import subtractive_clustering_algorithm



data = pd.read_csv("data.csv", header=None)
sample_data = data.copy()  # working with full data

# applying the algorithm
centers, newdata = subtractive_clustering_algorithm(0.5,0.75,0.5,0.15,data)


# plot of subtractive clustering results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(centers)):
    ax.scatter(xs = centers[i][0], ys = centers[i][1],  zs = centers[i][2], c="black", s = 100)
for j in range(0, len(newdata)):
    if newdata[j][3] == 1:
        c = "blue"
    elif newdata[j][3] == 2:
        c = "green"
    else:
        c = "red"
    ax.scatter(xs=newdata[j][0], ys=newdata[j][1], zs=newdata[j][2], c=c, s=2)
red_patch = mpatches.Patch(color='red', label='Cluster 1')
blue_patch = mpatches.Patch(color='blue', label='Cluster 2')
green_patch = mpatches.Patch(color='green', label='Cluster 3')
black_patch = mpatches.Patch(color='black', label='Respective centers')
plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])
ax.set_title("Subtractive clustering method")
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# k-means, using elbow method to determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(sample_data)
    kmeanModel.fit(sample_data)
    distortions.append(sum(np.min(cdist(sample_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / sample_data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()  # clearly 3 is the best choice of k


# plot of k-means clustering results with 3 clusters
num_of_clusters = 3
np.random.seed(57)
centre, var = spcv.kmeans(sample_data, num_of_clusters)
id, dist = spcv.vq(sample_data, centre)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(centre)):
    ax.scatter(xs = centre[i][0], ys = centre[i][1],  zs = centre[i][2], c="black", s = 100)
for j in range(0, len(id)):
    if id[j] == 1:
        c = "blue"
    elif id[j] == 2:
        c = "green"
    else:
        c = "red"
    ax.scatter(xs=data[0][j], ys=data[1][j], zs=data[2][j], c=c, s=2)
red_patch = mpatches.Patch(color='red', label='Cluster 1')
blue_patch = mpatches.Patch(color='blue', label='Cluster 2')
green_patch = mpatches.Patch(color='green', label='Cluster 3')
black_patch = mpatches.Patch(color='black', label='Respective centers')
plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])
ax.set_title("K-means clustering method")
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()



# comparison of the centers
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, len(centre)):
    if i == (len(centre) - 1):
        l1 = "K-means centers"
    else:
        l1= None
    ax.scatter(xs = centre[i][0], ys = centre[i][1],  zs = centre[i][2], c="red", s = 100, label = l1)
for i in range(0, len(centers)):
    if i == (len(centre) - 1):
        l2 = "Subtractive clustering centers"
    else:
        l2 = None
    ax.scatter(xs = centers[i][0], ys = centers[i][1],  zs = centers[i][2], c="blue", s = 100, label = l2)
for j in range(0, len(data)):
    ax.scatter(xs=data[0][j], ys=data[1][j], zs=data[2][j], c="grey", s=2)
plt.legend(loc = 0, bbox_to_anchor=(0.4, 0.3))
ax.set_title("Comparison of centers")
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# agreement/disagreement between two methods
testdata = sample_data.copy()  # original data
testdata = np.array(testdata)
kmeans_labels = np.concatenate((testdata, id[:,None]), axis=1)
subtractive_labels = newdata

# sorting to compare labels
kmeans_labels = kmeans_labels[kmeans_labels[:,0].argsort()]
subtractive_labels = subtractive_labels[subtractive_labels[:,0].argsort()]


# lets examine how many examples have the same label
same = 0
different = 0
for i in range(0,len(subtractive_labels)):
    if subtractive_labels[i][3] == kmeans_labels[i][3]:
        same += 1
    else:
        different += 1

# the two methods agree in their classifications
print('Agreement: ', same)
print('Disagreement: ', different)
