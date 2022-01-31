'''from machine learning A-Z course
   coded by trishit nath thakur'''

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing Thompson Sampling


import random # random library for different data distributions 

N = 10000 # total number of times ads were shown

d = 10 # total number of ads

ads_selected = [] # full list of ads selected over rounds

numbers_of_rewards_1 = [0] * d  # no of times ad i got reward 1 upto round n

numbers_of_rewards_0 = [0] * d # no of times ad got reward 0 upto round n

total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0

    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # for each ad we take random draw from distributions

        if random_beta > max_random:
            max_random = random_beta
            ad = i                       # we select ad that has highest max_random

    ads_selected.append(ad)
    reward = dataset.values[n, ad]

    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1

    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1

    total_reward = total_reward + reward


# Visualising the results - Histogram


plt.hist(ads_selected)
plt.title('Histogram of ads selections')

plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

plt.show()