'''from machine learning A-Z
   coded by trishit nath thakur'''

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Wine.csv')

X = dataset.iloc[:, :-1].values #all columns except the last
y = dataset.iloc[:, -1].values #the last column


# Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# when random_state set to an integer, train_test_split will return same results for each execution.

# Feature Scaling


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

#fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data

X_test = sc.transform(X_test)

#Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data

# Applying PCA


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  # n_components is final number of extracted features you want 

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Training the Logistic Regression model on the Training set


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)  #creating object of logistic regression with random_state seed used by random number generator
classifier.fit(X_train, y_train)    #object fitted with training set


# Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

'''A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes
    The target variable has two values: Positive or Negative
     The columns represent the actual values of the target variable
     The rows represent the predicted values of the target variable'''


print(cm)
accuracy_score(y_test, y_pred)


# Visualising the Training set results


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

                  #making a grid of high density that pixels are seperated closely by 0.01

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.legend()
plt.show()


# Visualising the Test set results


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

                  #making a grid of high density that pixels are seperated closely by 0.01

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.legend()
plt.show()