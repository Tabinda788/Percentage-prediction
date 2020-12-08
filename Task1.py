# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:53:00 2020

@author: DELL
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt  

import seaborn as sns

from sklearn import metrics

from sklearn.linear_model import LinearRegression  

from sklearn.model_selection import train_test_split


url = "http://bit.ly/w-data"

df = pd.read_csv(url)


df.head(10)

x=df['Hours']
y=df['Scores']
plt.scatter(x,y)
plt.title('Hours Vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Marks obtained')

Hour=list(df.columns)
print(Hour)
features=list(set(Hour)-set(['Scores']))
print(features)

x=df[features].values
print(x)

y=df['Scores'].values
print(y)


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=1)

reg = LinearRegression()  
reg.fit(train_x, train_y) 
prediction = reg.predict(test_x)


line = reg.coef_*x+reg.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


print(test_y)
print(prediction)

score=reg.score(test_x,test_y)
print(score*100,'%')


own = 9.25
own_pred = reg.predict([[own]])
print(own_pred)


  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, prediction)) 





















