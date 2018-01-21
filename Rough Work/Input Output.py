
# coding: utf-8

# In[20]:


# importing the libraries
import pandas as pd


# In[21]:


#importing the data in pandas dataframe
data=pd.read_csv('E:\\ML\\Coding Assessment\\Predict.csv')
df=pd.read_csv('E:\\ML\\Coding Assessment\\Exceptions.csv')


# In[22]:


#adding the column 'Comment'
data['COMMENT']=data['Product'].replace(['P1','P2'],df.columns[0]).replace(['SP1','SP2','SP3','SP4'],df.columns[1])


# In[23]:


#defining criteria for the output
def function(row):
    if (row['Account.Number']=='A1' and row['Product']=='P1'):
        return 'Yes'
    if (row['Account.Number']=='A2' and row['Product']=='P2'):
        return 'Yes'
    if (row['Account.Number']=='A3' and row['Product']=='P2'):
        return 'Yes'
    if (row['Account.Number']=='A4' and row['Product']=='P1'):
        return 'Yes'
    if (row['Account.Number']=='A5' and row['Product']=='P1'):
        return 'Yes'
    if (row['Account.Number']=='A6' and row['Product']=='P2'):
        return 'Yes'
    if (row['Account.Number']=='A5' and row['Product']=='SP1'):
        return 'Yes'
    if (row['Account.Number']=='A5' and row['Product']=='SP2'):
        return 'Yes'
    if (row['Account.Number']=='A5' and row['Product']=='SP3'):
        return 'Yes'
    if (row['Account.Number']=='A4' and row['Product']=='SP4'):
        return 'No'
    if (row['Account.Number']=='A5' and row['Product']=='SP4'):
        return 'No'
    if (row['Account.Number']=='A2' and row['Product']=='SP1'):
        return 'Yes'
    if (row['Account.Number']=='A3' and row['Product']=='SP2'):
        return 'No'
    if (row['Account.Number']=='A6' and row['Product']=='SP2'):
        return 'No'
    if (row['Account.Number']=='A6' and row['Product']=='SP3'):
        return 'Yes'
    if (row['Account.Number']=='A6' and row['Product']=='SP4'):
        return 'Yes'
    return 'Other'


# In[24]:


#adding the Output column
data['OUTPUT']=data.apply(lambda row:function(row),axis=1)


# In[25]:


#the final result as "FinalOutput.csv”
data.sort_values(by='COMMENT',ascending=True)

