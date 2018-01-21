
# coding: utf-8

# In[1]:


import pandas as pd


# In[48]:


data=pd.read_csv('E:\\ML\\HP\\train.csv')
test=pd.read_csv('E:\\ML\\HP\\test.csv')


# In[49]:


data.describe()


# In[50]:


data.columns


# In[52]:


data.info()


# In[54]:


data.isnull().sum()


# In[55]:


test.isnull().sum()


# In[60]:


data1=data.dropna()


# In[67]:


missing=[col for col in data.columns
        if data[col].isnull().any()]
reduced=data.drop(missing,axis=1)
test1=test.drop(missing,axis=1)


# In[69]:


test1.isnull().sum()


# In[84]:


from sklearn.preprocessing import Imputer
c=Imputer()
data_imp=Imputer(strategy="most_frequent").fit_transform(data)


# In[80]:


missing1=[col for col in data.columns
          if data[c].isnull().any()]


# In[86]:


data.head()


# In[87]:


test.head()


# In[92]:


data_imp=data.copy()
test_imp=test.copy()
miss=[col for col in data.columns
     if data[col].isnull().any()]
for col in miss:
    data_imp[col+'_missing']=data_imp[col].isnull()
    test_imp[col+'_missing']=test_imp[col].isnull()


# In[93]:


c=Imputer()


# In[94]:


data_imp=c.fit_transform(data_imp)
test_imp=c.fit_transform(test_imp)

print('mean absolute error:')
print(score_dataset)


# In[95]:


print(score_dataset(data_imp, test_imp,y_train,y_test))


# In[97]:


data.dtypes


# In[98]:


encoding=pd.get_dummies(data)


# In[99]:


encoding.dtypes

