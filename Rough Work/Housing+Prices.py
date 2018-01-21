
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from scipy import sparse
import os,time


# In[2]:


df_train=pd.read_csv('E:\\ML\\Housing price\\train.csv')
df_test=pd.read_csv('E:\\ML\\Housing price\\test.csv')


# In[3]:


df_train.head()


# In[4]:


df_train.tail()


# In[5]:


print(df_train.columns.values)
df_train.info()


# In[6]:


df_train.describe()


# In[7]:


df_test.head()


# In[8]:


print(df_test.columns.values)


# In[9]:


df_train.columns.difference(df_test.columns)


# In[10]:


target=df_train['SalePrice']


# In[11]:


df_train1=df_train[df_test.columns]


# In[12]:


df_train1.columns.difference(df_test.columns)


# In[13]:


preprocessdata=pd.concat([df_train1,df_test])


# In[14]:


preprocessdata.head()


# In[15]:


def ReplaceNA(DataCol,ColValue):
    return DataCol.fillna(value=Colvalue)


# In[16]:


preprocessdata['PoolQC'].unique()


# In[17]:


preprocessdata.isnull().any()


# In[18]:


df_train1.describe().columns


# In[19]:


preprocessdata=preprocessdata.fillna(preprocessdata.mean())


# In[20]:


nullcolumns=preprocessdata.columns[preprocessdata.isnull().any()]


# In[21]:


nullcolumns


# In[ ]:


preprocessdata[nullcolumns]=preprocessdata.replace(preprocessdata[nullcolumns],'other')

