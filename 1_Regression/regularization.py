
# coding: utf-8

# ### Regularization
# 
# Regularization is a very important technique in machine learning to prevent overfitting. 

# it adds a regularization term in order to prevent the coefficients to fit so perfectly to overfit. 
# 
# Regularization: L1 and L2
#     
# L2 is the sum of the square of the weights, while L1 is just the  absolute sum of the weights.

# Regularization is a technique to reduce the complexity of the model. It does this by penalizing the loss function. This helps to solve the overfitting problem.

# ### Lets try to understand what is penalizing the loss function

# Loss function is the sum of squared difference between the actual value and the predicted value
# 
# <img src='images/loss.PNG' height='40%' width='40%'/>

# As the degree of the input features increases the model becomes complex and tries to fit all the 
# data points as shown below

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[2]:


x=np.array([0,1,2,3,4,5])
y=np.array([0,0.8,0.9,0.1,-0.8,-1])


from sklearn.linear_model import Ridge


# In[16]:

'''
def ridge_poly(deg):
    p = PolynomialFeatures(deg)
    new_x=p.fit_transform(x.reshape(-1,1))
    
    lg = Ridge()
    pred=lg.fit(new_x,y).predict(new_x)
    print (lg.coef_)
    plt.scatter(x,y)
    plt.plot(x,pred)
    plt.show()

'''
# In[17]:


#ridge_poly(6)


# In[ ]:
def plot_1(i,pred,d):

    s='alpha={} and degree={}'.format(np.round(i,2),d)
    plt.title(s)
    plt.scatter(x,y)
    ln=plt.plot(x,pred)
    plt.pause(.1)
    ln[0].remove()
'''   
p = PolynomialFeatures(6)
new_x=p.fit_transform(x.reshape(-1,1))
a=np.linspace(0,1,50)
for i in a:
    rid = Ridge(alpha=i)
    rid.fit(new_x,y)
    pred = rid.predict(new_x)
    plot_1(i,pred)
'''
a=np.linspace(0,1,50)
for d in [2,3,4,5,6,7]:
    for i in a:
        p = PolynomialFeatures(d)
        new_x=p.fit_transform(x.reshape(-1,1))


        rid = Ridge(alpha=i)
        rid.fit(new_x,y)
        pred = rid.predict(new_x)
        plot_1(i,pred,d)


# In[ ]:
'''

from sklearn.datasets import load_boston


# In[ ]:


boston = load_boston()


# In[ ]:


X = boston.data
y = boston.target


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=10)


# In[ ]:


x_train.shape


# In[ ]:


p=PolynomialFeatures(degree=4)
X_p_train=p.fit_transform(x_train)
X_p_test=p.transform(x_test)


# In[ ]:


lg = LinearRegression()
lg.fit(X_p_train,y_train)


# In[ ]:


print('train score:',lg.score(X_p_train,y_train))
print('test score:',lg.score(X_p_test,y_test))


# In[ ]:


std = StandardScaler()
train_std = std.fit_transform(x_train)
test_std = std.transform(x_test)


# In[ ]:


poly = PolynomialFeatures()
X_poly = poly.fit_transform(train_std)
X_test_poly = poly.transform(test_std)


# In[ ]:


### Ridge regression/Linear Regression


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.DataFrame(X,columns=boston.feature_names)


# In[ ]:


df['target'] = boston.target


# In[ ]:


df.corr()


# In[ ]:


r = Ridge(max_iter=500,alpha=.01)
r.fit(X_poly,y_train)


# In[ ]:


print('train score:',r.score(X_poly,y_train))
print('test score:',r.score(X_test_poly,y_test))

'''
