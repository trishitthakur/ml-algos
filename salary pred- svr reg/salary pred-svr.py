'''from the machine learning A-Z course
   coded by trishit nath thakur '''

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values #the first column
y = dataset.iloc[:, -1].values  #only the last column

y = y.reshape(len(y),1) #update shape of y to (no of rows, no of columns)
print(y)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
sc_y = StandardScaler()

  #standard scaler for x and y must be different as it calculates mean and standard deviation for x and y seperately

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)


# Training the SVR model on the whole dataset

from sklearn.svm import SVR  
regressor = SVR(kernel = 'rbf')  #Specifies the kernel type to be used in the algorithm
regressor.fit(X, y)  #fit regressor object to the dataset


# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))) 

 #first apply x_transform to data then change to real world value using inverse y transform

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')   #plot for the real values
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue') ##plot for the predicted values
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
