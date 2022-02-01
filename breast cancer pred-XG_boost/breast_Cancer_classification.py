'''from machine learning A-Z
   coded by trishit nath thakur'''

#XGBoost applies a better regularization technique to reduce overfitting, and it is one of the differences from the gradient boosting.


# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 # when random_state set to an integer, train_test_split will return same results for each execution.

# Training XGBoost on the Training set


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


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


# Applying k-Fold Cross Validation


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# estimator is the object to use to fit the data.
# x is the data to fit
# y is target variable to try to predict in the case of supervised learning.

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))   # printing the mean
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))   # printing the standard deviation