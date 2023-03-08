# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 20:33:30 2022

@author: marah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

y_pred=regressor.predict([[6.5]])

#visualling the decision tree regressor results (higher resolution)

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.title('Truth of Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
