# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:36:40 2022

@author: marah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


y_pred2=regressor.predict(np.array(1.5).reshape(1,1))
 
#visualling the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.TickHelper('salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()