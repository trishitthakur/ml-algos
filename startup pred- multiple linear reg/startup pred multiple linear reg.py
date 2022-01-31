'''from the machine learning A-Z course
   coded by trishit nath thakur'''

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values


# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') 

  '''[3] is the column in which we apply encoding while remainder means all columns not 
     specified in the list of “transformers” will be passed through without transformation'''

X = np.array(ct.fit_transform(X))


# Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
 # when random_state set to an integer, train_test_split will return same results for each execution.

# Training Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  #creating object
regressor.fit(X_train, y_train)  #object fitted on the dataset


# Predicting Test set results

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)  #numerical values with 2 decimal after comma
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

 #to print real values and predicted values side by side to compare

# Evaluating the model performance

from sklearn.metrics import r2_score
r2_score(y_test,  y_pred)

