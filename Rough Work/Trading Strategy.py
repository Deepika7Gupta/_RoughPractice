
# coding: utf-8

# In[3]:


from pandas_datareader import data as web


# In[4]:


import numpy as np


# In[5]:


from sklearn.linear_model import Lasso


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[9]:


from sklearn.model_selection import RandomizedSearchCV as rcv


# In[11]:


from sklearn.pipeline import Pipeline


# In[12]:


from sklearn.preprocessing import Imputer


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


get_ipython().magic('matplotlib inline')


# In[17]:


df= web.DataReader('SPY', data_source='yahoo', start=2010)


# In[19]:


df=df[['Open','High', 'Low','Close']]


# In[20]:


df.head()


# In[21]:


df.tail()


# In[26]:


df['open']=df['Open'].shift(1)
df['close']=df['Close'].shift(1)
df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)


# In[27]:


df.head()


# In[29]:


imp=Imputer(missing_values='Nan', strategy='mean', axis=0)


# In[34]:


steps= [('imputation', imp),
        ('scaler', StandardScaler()),
        ('lasso', Lasso)]


# In[35]:


pipeline=Pipeline(steps)


# In[37]:


parameters= {'lasso_alpha':np.arange(0.0001,10,.0001),
            'lasso_max_iter':np.random.uniform(100,100000,4)}


# In[39]:


reg=rcv(pipeline, parameters,cv=5)


# In[40]:


reg


# In[42]:


X=df[['open', 'high','low','close']]
Y=df['Close']

