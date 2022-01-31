'''from machine learning A-Z
   coded by trishit nath thakur'''


# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  #X contains annual income and spending score


#dendrogram to find the optimal number of clusters


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) 

#linkage function takes two input matrix of features(X) and clustering technique(ward is min variance technique)

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')

plt.show()


# Training the Hierarchical Clustering model on the dataset
#after analysing dendrogram found 5 to be proper no of clusters here

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

#affinity is type of distance used to measure variance values in clusters while ward is min variance method

y_hc = hc.fit_predict(X)  #fit_predict will not only train on X but will also return dependent variable of 5 clusters


# Visualising the clusters


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

#y_hc == 0 correspond to first cluster and so on while the other part is value of annual income(0 on x) or spending score(1 on y)

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.legend()
plt.show()