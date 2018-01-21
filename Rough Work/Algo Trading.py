
# coding: utf-8

# In[10]:


import pandas as pd


# In[11]:


import numpy as np


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


import seaborn as sns


# In[14]:


df = pd.read_csv('E:\\Programs\\5_Year_NIFTY.csv')


# In[15]:


df.head()


# In[16]:


pnl= np.log(df['CLOSE']/df['CLOSE'].shift(1))


# In[17]:


print(pnl)


# In[18]:


from pandas_datareader import data as pdr


# In[19]:


import datetime


# In[20]:


chn= df['CLOSE'].shift(1)-df['CLOSE'].shift(6)


# In[21]:


print(chn)


# In[22]:


std= pd.rolling_std(chn, window=100)


# In[23]:


std= chn.rolling(window=100, center=False).std()


# In[24]:


print(std)


# In[28]:


df['DateTime'] = df['DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))


# In[29]:


df.head()


# In[30]:


df1= df.drop('DATE', axis=1)


# In[31]:


df1.head()


# In[32]:


import datetime as dt


# In[55]:


df1['weekday'] = df1[['DateTime']].apply(lambda x: dt.datetime.strftime(x['DateTime'], '%A'), axis=1)
df1.head()


# In[48]:


import calendar


# In[54]:


def is_date_the_nth_friday_of_month(nth, date=None):
    if not date:
        date = datetime.datetime.df1['DateTime']
    if date.weekday() == 4:
        if (date.day - 1) // 7 == (nth - 1):
            return True
        return False


# In[49]:


import datetime


# In[47]:


pct= df.CLOSE.pct_change()[1:]


# In[146]:


pct


# In[147]:


df1['pct_change']= df1.CLOSE.pct_change()


# In[148]:


df1.head()


# In[70]:


from math import ceil

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


# In[71]:


week_of_month('DateTime')

