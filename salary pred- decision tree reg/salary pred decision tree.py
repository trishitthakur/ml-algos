'''from the machine learning A-Z course
   coded by trishit nath thakur '''

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values   #to select the first column
y = dataset.iloc[:, -1].values    #to select the salary column

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)   

#regressor object created with random_state to get same result in the end neglect random factors

regressor.fit(X, y) #object fitted on the dataset


# Predicting a new result

regressor.predict([[6.5]]) #to input 2D array use [[]]


# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()