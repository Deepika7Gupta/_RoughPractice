
# coding: utf-8

# In[4]:


import pandas


# In[5]:


df= pandas.read_csv('C:\\Users\\Deepika\\Downloads\\2016-pydata-carolinas-pandas-master\\2016-pydata-carolinas-pandas-master\\data\\gapminder.tsv',sep='\\t')


# In[10]:


df


# In[11]:


df.head()


# In[12]:


type(df)


# In[14]:


df.shape


# In[15]:


df.columns


# In[16]:


df.dtypes


# In[18]:


df['country']


# In[19]:


df[['country', 'continent','year']]


# In[35]:


subset= df[list([1,2,3])]


# In[28]:


subset.head()


# In[31]:


df[[1]]


# In[37]:


subset= df[list(range(1,3))]


# In[39]:


df.loc[0]


# In[42]:


df.shape[0]


# In[49]:


df.loc[1703]


# In[47]:


df.iloc[0]


# In[6]:


df.iloc[1703]


# In[7]:


df.ix[0]


# In[15]:


df.groupby('year')['lifeExp'].mean()


# In[12]:


df.head()


# In[19]:


df.groupby(['year'])['lifeExp'].mean().plot()


# In[17]:


import matplotlib.pyplot as plt

