# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:33:40 2022

@author: marah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title("linear reg")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

'''
from sklearn.preprocessing import PolynomialFeatures
polynomialfeatures=PolynomialFeatures(degree=4)
X_poly=polynomialfeatures.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly, y)

plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.scatter(X, y, color='yellow')

y_pred1=lin_reg2.predict([[6.5]])

'''

