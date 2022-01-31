'''from machine learning A-Z
   coded by trishit nath thakur'''

# !pip install apyori before we begin

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Data Preprocessing


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []  #list of all transactions one by one

for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #fill the list with all transactions


# Training the Apriori model on the dataset


from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#apriori function returns the rules(variable declared)
'''Support refers to the default popularity of any product. You find the support as a quotient of the division of the
    number of transactions comprising that product by the total number of transactions'''

'''Confidence refers to the possibility that the customers bought both biscuits and chocolates together.
   So, you need to divide the number of transactions that comprise both biscuits and chocolates by the total number of transactions to get the confidence.'''

'''lift refers to the increase in the ratio of the sale of chocolates when you sell biscuits'''


# Visualising the results



## Displaying the first results coming directly from the output of the apriori function


results = list(rules)
results


## Putting the results into a Pandas DataFrame


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] #
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


## Displaying the results non sorted


resultsinDataFrame


## Displaying the results sorted by descending lifts


resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

# n is number of rows to return based on column specified