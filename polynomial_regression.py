#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset = pd.read_csv(r'E:\IIT KGP\ML\P14-Part2-Regression\Section 8 - Polynomial Regression\Python\Position_Salaries.csv')
dataset


# In[26]:


x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values
x,y


# In[25]:


#from sklearn.model_selection import train_test_split
#xtrain,xtest, ytrain, ytest = train_test_split(x,y, train_size = 0.9)


# In[38]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)


# In[39]:


print(x)
lr.predict(x)


# In[41]:


# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[42]:


# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lr.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# In[43]:


# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[46]:


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lr.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:




