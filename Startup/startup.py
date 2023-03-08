# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:44:46 2022

@author: marah
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X=X[:,1:]


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)


import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
regessor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regessor_ols.summary()


import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,3,4,5]],dtype=float)
regessor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regessor_ols.summary()

import statsmodels.api as sm
X_opt=np.array(X[:,[0,3,5]],dtype=float)
regessor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regessor_ols.summary()

X_train_opt, X_test_opt, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
regressor2=LinearRegression()
regressor2.fit(X_train_opt,y_train)

y_pred2=regressor2.predict(X_test_opt)

