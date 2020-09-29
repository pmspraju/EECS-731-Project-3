# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
#import networkx as nx
import numpy  as np
import pandas as pd
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, sentimentPolarity, exploreData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 5\HW\Git\EECS-731-Project-3\Data'
filename = "links.csv"
data_l = loadData(path,filename)

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 5\HW\Git\EECS-731-Project-3\Data'
filename = "movies.csv"
data_m = loadData(path,filename)
genres = ['Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

d1 = pd.DataFrame(columns = ['movieId','title','genre'])
for ind, row in data_m.iterrows():
    gstr = row['genres']
    glst = gstr.split("|")
    cnt = 0
    for x in glst:
        d1.loc[ind + cnt] = data_m.loc[ind]
        d1.at[ind+cnt,'genre'] = x
        cnt = cnt + 1
#d1.to_csv('test.csv',index=False)   

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 5\HW\Git\EECS-731-Project-3\Data'
filename = "ratings.csv"
data_r = loadData(path,filename)
data_r = data_r.drop(['userId','timestamp'], axis = 1)
data_r = data_r.groupby(['movieId'], as_index=False).max(level=0)
data_r['rating'] = data_r.groupby(['movieId'], as_index=False)['rating'].apply(lambda x: x.value_counts().index[0])
#data_r.to_csv('test.csv',index=False)

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 5\HW\Git\EECS-731-Project-3\Data'
filename = "tags.csv"
data_t = loadData(path,filename)
data_t = data_t.drop(['userId','timestamp'], axis = 1)
data_t['tag'] = data_t['tag'].apply(lambda x: sentimentPolarity(x))
data_t = data_t.groupby(['movieId'], as_index=False).count()
#data_t.to_csv('test.csv',index=False)

data2 = pd.merge(data_r,data_t, on=['movieId'], how='inner')
data = pd.merge(d1,data2, on=['movieId'], how='inner')
data.to_csv('test.csv',index=False)

drop_col = ['title']
data = data.drop(drop_col, axis = 1)

print ("----------------------Shakespear Play data-----------------------------")
features, target = exploreData(data)
misVal, mis_val_table_ren_columns = missingValues(data)

from projectFunctions import splitData, kmeans, transformData
data_tran = transformData(data)
X_train, X_test = splitData(data_tran,  0.3)
result,scr = kmeans(X_train, X_test)
print(result)
print(scr)

#data.to_csv('test.csv',index=False)