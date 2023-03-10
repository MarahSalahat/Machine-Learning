# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 21:03:57 2022

@author: marah
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#training the random forest regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=40,random_state=0)
regressor.fit(X,y)

y_pred= regressor.predict([[6.5]])


X_grid= np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth of Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
