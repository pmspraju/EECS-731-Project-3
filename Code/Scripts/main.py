# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
import networkx as nx
import numpy  as np
import pandas as pd
from pandas import compat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, exploreData, missingValues, tokenString, transformData

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 4\HW\Git\EECS-731-Project-2\Data'
filename = "Shakespeare_data.csv"
data = loadData(path,filename)
drop_col = ['Dataline','PlayerLinenumber','ActSceneLine']
data = data.drop(drop_col, axis = 1)
data.rename(columns={'Player':'target'},inplace=True)
print(data.columns)

print ("----------------------Shakespear Play data-----------------------------")
features, target = exploreData(data)
misVal, mis_val_table_ren_columns = missingValues(data)

# Print some summary information
print ("Columns that have missing values:" + str(misVal.shape[0]))
print ("-----------------------------------------------------------------------")
print(mis_val_table_ren_columns.head(20))

#Remove rows with missing target values
ind = data[data['target'].isnull()].index.tolist()
data = data.drop(index=ind, axis=0)

#Compute features to add value
line_count = data.groupby(['Play','target'], as_index=False).count()
line_count.rename(columns={'PlayerLine':'LineCount'},inplace=True)
 
#merge the group by counts to original dataframe
data = pd.merge(data,line_count,on=['Play','target'], how='inner')

#Remove rows with line count <=200
#Remove rows with missing target values
#ind = data[data['LineCount'] <= 350].index.tolist()
#data = data.drop(index=ind, axis=0)

#Find the important players in a play using network
#network graph
g = nx.Graph()
g = nx.from_pandas_edgelist(data,source='Play',target='target')
 
#Get importance by computing Degree of centrality
col = ['Degree','PageRank','Name']
doc = pd.DataFrame(columns = col)
doc['Degree'] = nx.degree_centrality(g).values()
doc['PageRank'] = nx.pagerank(g).values()
doc['Name'] = nx.degree_centrality(g).keys()

#Extract only importance of players
doc_target = doc[doc['Name'].isin(data['target'].unique().tolist())]
doc_target = doc_target.drop(['PageRank'], axis = 1)
doc_target.rename(columns={'Name':'target'},inplace=True)
data = pd.merge(data,doc_target,on=['target'], how='inner')

t = []
data['PlayerLine'].apply(lambda x: t.append(x))
corpus = ' '.join(t)
stop_w = set(stopwords.words('english'))
tokens = word_tokenize(corpus)
sen = [w for w in tokens if not w in stop_w]
corpus = [w for w in sen if w.isalpha()]
fdist=FreqDist(corpus)

#tokenize the strig
#Compute the frequency of words in a sentence 
data['PlayerLine'] = data['PlayerLine'].apply(lambda x: tokenString(x,fdist,stop_w))

features, target = exploreData(data)
features_final, target_final = transformData(features, target)

#Split the data with test size = 30
from projectFunctions import splitData,svmClassifier,decTree,naiveBayes
X_train, X_test, y_train, y_test = splitData(features_final, target_final, 0.3)

#results,learner = svmClassi    fier(X_train, X_test, y_train, y_test)

#print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
#print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
#print "-----------------------------------------------------------------------"

results,learner = decTree(X_train, y_train, X_test, y_test, 'gini', 13)
# 
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

results,learner = naiveBayes(X_train, y_train, X_test, y_test)
 
print "Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time'])     
print "Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test'])     
print "-----------------------------------------------------------------------"

#data.to_csv('test.csv',index=False)