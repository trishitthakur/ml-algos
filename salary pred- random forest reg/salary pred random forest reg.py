'''from the machine learning A-Z course
   coded by trishit nath thakur '''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) 

  # n_estimators refer to number of trees to be used while random_state fix seeds to remove randomness in result

regressor.fit(X, y)   #object fitted on the dataset

# Predicting a new result
regressor.predict([[6.5]])   # input is in the form of 2D array


# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))  # reshape to (number_of_rows, number_of_columns)


plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()