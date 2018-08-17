# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 18:00:16 2018

@author: ADMIN
"""

import numpy as np 
import pandas as pd # data processing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('data2csv.csv')
dataset.head()
print('The shape of our dataset is:', dataset.shape)
dataset.describe()
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, :1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)

# Scale the data to be between -1 and 1
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestRegressor(n_jobs=-1)
#plot accuracy for different number of trees
estimators = np.arange(10, 600, 30)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train.ravel())
    scores.append(model.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
print(scores)
