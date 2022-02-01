'''from machine learning A-Z
   coded by trishit nath thakur'''


# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 
  
  # delimeter is 't' indicating it is tsv file and not csv quoting to ignore all quotes in file


# Cleaning the texts


import re
import nltk

nltk.download('stopwords') # download all words not useful in prediction ie not relevant

from nltk.corpus import stopwords # import 
from nltk.stem.porter import PorterStemmer # taking only root of word to indicate what it means

corpus = [] # contained all cleaned reviews 

for i in range(0, 1000):

  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # will remove all the punctuations and other non letter text by spaces. The ^ specifies what not to remove

  review = review.lower() # will convert to lower case

  review = review.split() # split review to different words to apply stemming on each word

  ps = PorterStemmer() # call porterstemmer

  all_stopwords = stopwords.words('english') # get all english stopwords

  all_stopwords.remove('not') # since we want 'not' to not be removed since it is needed for sentiment analysis

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #get the root of all words if not in stopwords list

  review = ' '.join(review) # join all words back to make sentences for prediction 

  corpus.append(review) # add all reviews to list corpus one by one 

print(corpus) 


# Creating the Bag of Words model
''' tokenization to create sparse matrix to create all review in different rows and all words from all reviews 
 in dif col where cells will get 1 or 0 depending if it is there or not'''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 

# max_features is max no of most frequent words in col of sparse matrix since corpus still has irrelevant words that we dont want

X = cv.fit_transform(corpus).toarray() #fit method takes all words and transform puts in differnt columns, to array so that matrix of features is 2D array for NB model
y = dataset.iloc[:, -1].values # result of reviews


# Splitting the dataset 


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# when random_state set to an integer, train_test_split will return same results for each execution.

# Training the Naive Bayes model on the Training set


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB() 

classifier.fit(X_train, y_train)


# Predicting the Test set results


y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) #to represent prediction and real values side by side


# Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

 '''A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes
    The target variable has two values: Positive or Negative
     The columns represent the actual values of the target variable
     The rows represent the predicted values of the target variable'''

print(cm)

accuracy_score(y_test, y_pred)