# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
#import sys
import time
import pandas as pd
import numpy  as np
#import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import fbeta_score
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.probability import FreqDist
import seaborn as sns; sns.set()
from textblob import TextBlob
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
#nltk.download('punkt')

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]    
           
           #separate features and target
           drop_col = ['rating']
           features = data.drop(drop_col, axis = 1)
           target = data[drop_col]
          
           # Print the results
           print ("-----------------------------------------------------------------------")
           print ("Total number of records: {}".format(rows))
           print ("Total number of features: {}".format(cols))
           print ("-----------------------------------------------------------------------")
           
           return features,target
           
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def missingValues(data):
    try:
           # Total missing values
           mis_val = data.isnull().sum()
         
           # Percentage of missing values
           mis_val_percent = 100 * mis_val / len(data)
           
           # Make a table with the results
           mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
           
           # Rename the columns
           mis_val_table_ren_columns = mis_val_table.rename(
           columns = {0 : 'Missing Values', 1 : '% of Total Values'})
           mis_val_table_ren_columns.head(4 )
           # Sort the table by percentage of missing descending
           misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                   '% of Total Values', ascending=False).round(1)
                     
           return misVal, mis_val_table_ren_columns

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def tokenString(sen,fdist,stop_w):
    try:
        tokens = word_tokenize(sen)
        sen = [w for w in tokens if not w in stop_w]
        sen = [w for w in sen if w.isalpha()]
        #sen_f = TreebankWordDetokenizer().detokenize(sen)
        wordcnt = 0
        for w in sen:
            wordcnt = wordcnt + fdist[w]
        #print(tokens)
        #print(sen_f)
        #sys.stdout.write('.'); sys.stdout.flush();
        
        return wordcnt 
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def sentimentPolarity(sen):
    try:
        sp = TextBlob(sen)
        if (sp < 0):
            polarity = 0
        else:
            polarity = 1
            
        return polarity 
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
       
def transformData(data):
    try:    
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        #features_final = pd.get_dummies(features_log_minmax_transform)
        enc = LabelEncoder()
        data['genre'] = enc.fit_transform(data['genre'])
         
        return data
        
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#split the data in to train and test data
def splitData(features, testsize):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test = train_test_split(features,
                                           test_size = testsize, 
                                           random_state = 1)

        # Show the results of the split
        print ("Features training set has {} samples.".format(X_train.shape[0]))
        print ("Features testing set has {} samples.".format(X_test.shape[0]))
        print ("-----------------------------------------------------------------------")
        return X_train, X_test
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def kmeans(X_train, X_test):
    try:
       kmeans = KMeans(n_clusters=15, random_state=0).fit(X_train)
       result = kmeans.predict(X_test)
       scr = kmeans.score(X_test)
       return result, scr
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
