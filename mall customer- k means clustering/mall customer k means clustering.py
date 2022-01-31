'''from machine learning A-Z
   coded by trishit nath thakur'''

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  #X contains annual income and spending score


# Using the elbow method to find the optimal number of clusters


from sklearn.cluster import KMeans

wcss = []    #initialising within cluster sum of squares and storing for different no of clusters

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #kmeans++ used to protect from random initialisation trap

    #random_state for same result by fixing seed for all outcomes

    kmeans.fit(X) #object fitted on data
    wcss.append(kmeans.inertia_) #store for different values of clusters in wcss, inertia gives wcss values

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')

plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()


# Training the K-Means model on the dataset  
#after finding a proper value of no of clusters we apply kmeans

 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42) #object initialized
y_kmeans = kmeans.fit_predict(X)  #fit_predict will not only train on X but will also return dependent variable of 5 clusters


# Visualising the clusters


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') 

   #y_kmeans == 0 correspond to first cluster and so on while the other part is value of annual income(0 on x) or spending score(1 on y)

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.legend()
plt.show()