#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# In[60]:


data = pd.read_csv(r'E:\IIT KGP\ML\practise_datasets\top50.csv', encoding='latin-1')#since it contain latin words


# In[158]:


data.info()


# In[159]:


data.isnull().sum() #getting sum of NaN


# In[157]:


X = data.drop(['sl no','Artist.Name','Popularity','Genre',], axis = 1)
Y = data.iloc[:,13:].values


# In[160]:


X


# In[65]:


labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])


# In[162]:


xtrain,xtest,ytrain, ytest = train_test_split(x_opt,Y, train_size=0.80)


# In[163]:


lr = LinearRegression()
lr.fit(xtrain,ytrain)


# In[164]:


y_pred = lr.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))


# In[165]:


import statsmodels.formula.api as sm
x = np.append(np.ones((50,1)).astype(int),values=X, axis = 1)


# In[161]:


x_opt = x[:,0:]
x_opt = np.array(x_opt, dtype=float)
import statsmodels.api as sm1
model = sm1.OLS(Y,x_opt).fit()
model.summary()


# In[ ]:





# In[ ]:




