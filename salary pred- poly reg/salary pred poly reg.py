'''from the machine learning A-Z course
   coded by trishit nath thakur '''

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training the Linear Regression model on whole dataset

from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()   #regressor object created
lin_reg.fit(X, y)    #object fitted on the dataset


# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)  

    # y = bo + b1x1 + b2x2^2 +... bnxn^n, degree is the n in above line

X_poly = poly_reg.fit_transform(X) # to change matrix of single feature to n features 
lin_reg_2 = LinearRegression() #object created
lin_reg_2.fit(X_poly, y) #object fitted on the transformed dataset


# Visualising the Linear Regression results

plt.scatter(X, y, color = 'red')     #plot for the real values
plt.plot(X, lin_reg.predict(X), color = 'blue')   #plot for the predicted values
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising Polynomial Regression results

plt.scatter(X, y, color = 'red')       #plot for the real values
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')   #plot for the predicted poly reg model
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression

lin_reg.predict([[6.5]])


# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))