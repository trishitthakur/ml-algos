'''from the machine learning A-Z course
   coded by trishit nath thakur '''

#Importing libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  #all columns except the last
y = dataset.iloc[:, -1].values   #only the last column


# Splitting the dataset 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# when random_state set to an integer, train_test_split will return same results for each execution.

# Training the Simple Linear Regression model on Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regressor object created
regressor.fit(X_train, y_train) #object fitted on the dataset


# Predicting the Test set results

y_pred = regressor.predict(X_test)

# Visualising Training set results
plt.scatter(X_train, y_train, color = 'red')    #plot for the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue')   #plot for the predicted values
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising Test set results
plt.scatter(X_test, y_test, color = 'red')   #plot for the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  ##plot for the predicted values
  
  #here again x_train is plotted because line remains samwe if you use train or test

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()