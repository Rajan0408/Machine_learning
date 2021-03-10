#!/usr/bin/env python
# coding: utf-8

# In[243]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[244]:


data = pd.read_csv(r'E:\IIT KGP\ML\P14-Part2-Regression\Section 7 - Multiple Linear Regression\Python\50_Startups.csv')


# In[245]:


x = data.iloc[:,:-1].values
y = data.iloc[:,4].values
x


# In[246]:


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x


# In[247]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain, ytest = train_test_split(x,y, train_size=0.8)
xtrain,xtest,ytrain, ytest


# In[256]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)


# In[257]:


y_pred = lr.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))


# In[258]:


from sklearn.metrics import r2_score
r2_score(ytest, y_pred)


# In[259]:


import statsmodels.formula.api as sm
x = np.append(np.ones((50,1)).astype(int),values=x, axis = 1)


# In[260]:


x_opt = x[:,[0,1,2,3,4,5,6]]
x_opt = np.array(x_opt, dtype=float)
import statsmodels.api as sm1
model = sm1.OLS(y,x_opt).fit()
model.summary()


# In[261]:


x_opt = x[:,[0,1,2,3,4,6]]
x_opt = np.array(x_opt, dtype=float)
import statsmodels.api as sm1
model = sm1.OLS(y,x_opt).fit()
model.summary()


# In[262]:


x_opt = x[:,[0,1,2,3,4]]
x_opt = np.array(x_opt, dtype=float)
import statsmodels.api as sm1
model = sm1.OLS(y,x_opt).fit()
model.summary()


# In[263]:


x_opt


# In[ ]:





# In[ ]:




