#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv(r'E:\IIT KGP\ML\P14-Part2-Regression\Section 6 - Simple Linear Regression\Python\Salary_Data.csv')
data


# In[3]:


X = data.iloc[:,:-1].values
X


# In[4]:


Y = data.iloc[:,1].values
Y


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3)


# In[23]:


X_train, Y_train, X_test, Y_test


# In[24]:


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# In[25]:


y_pred = regressor.predict(X_train)
y_pred


# In[26]:


plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train, y_pred, color = 'Blue')
plt.title("Salary vs Experience(training)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[27]:


plt.scatter(X_test,Y_test, color = 'red')
plt.plot(X_train, y_pred, color = 'Blue')
plt.title("Salary vs Experience(test)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[ ]:




