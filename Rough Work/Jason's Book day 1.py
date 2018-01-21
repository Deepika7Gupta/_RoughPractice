
# coding: utf-8

# In[1]:


#strings
data='hello world!'
print(data[0])
print(len(data))
print(data)


# In[2]:


#Number
value=123.5
print(value)
value=10
print(value)


# In[3]:


#Boolean
a=True
b=False
print(a,b)


# In[4]:


#multiple Assignment
a,b,c=1,2,3
print(a,b,c)


# In[5]:


#No value
a=None
print(a)


# In[6]:


#if-then-else conditional
value=99
if value==99:
    print('This is fast')
elif value>200:
    print('This is too fast')
else:
    print('This is safe')


# In[7]:


#for-loop
for i in range(10):
    print(i)


# In[8]:


#while-loop
i=0
while i<10:
    print(i)
    i+=1


# In[9]:


#tuple
a=(1,2,3)
print(a)


# In[38]:


mylist=[1,2,3]
print('Zeroth value: %d' %mylist[0])
mylist.append(4)
print('len of mylist: %d'%len(mylist))
for values in mylist:
    print(values)


# In[45]:


mydict={'a':1, 'b':2,'c':3}
print('A value : %d' %mydict['a'])
mydict['a']=11
print('A value: %d'%mydict['a'])
print('Print Keys: %s'%mydict.keys())
print('Print values: %s'%mydict.values())
for key in mydict.keys():
    print(key)


# In[52]:


#sum function
def mysum(x,y):
    return x+y

result=mysum(1,3)
print(result)


# In[53]:


#numpy array
import numpy as np
mylist=[1,2,3]
myarray=np.array(mylist)
print(myarray)
print(myarray.shape)


# In[54]:


mylist=[[1,2,3],[4,5,6]]
myarray=np.array(mylist)
print(myarray)
print(myarray.shape)
print('First Row: %s'%myarray[0])
print('last row: %s'%myarray[-1])
print('specific row and column %s'%myarray[0,2])
print('While col: %s'%myarray[:,2])


# In[55]:


myarray1=np.array([1,2,3])
myarray2=np.array([2,2,2])
print('Sum ofd arrays: %s'%(myarray1+myarray2))
print('multiplication of arrays: %s'%(myarray1*myarray2))


# In[66]:


#matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
myarray=np.array([1,2,3])
plt.plot(myarray)
plt.show()


# In[68]:


x=np.array([1,2,3])
y=np.array([2,3,4])
plt.scatter(x,y)
plt.show()


# In[71]:


#pandas
import pandas as pd
myarray=np.array([1,2,3])
rownames=['a','b','c']
myseries=pd.Series(myarray,index=rownames)
print(myseries)


# In[72]:


print(myseries[0])
print(myseries['a'])


# In[88]:


myarray=np.array([[1,2,3],[2,3,4]])
rowname=['a','b']
columname=['one','two','three']
myframe=pd.DataFrame(myarray,index=rowname,columns=columname)
print(myframe)


# In[87]:


print('method 1:')
print('one column: \n%s'%myframe['one'])
print('obe column: \n%s'%myframe.one)


# In[99]:


import csv
filename='E:\\ML\\pima-indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pd.read_csv(filename,names=names)


# In[97]:


print(data.head())
print(data.shape)


# In[101]:


url='https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
data1=pd.read_csv(url,names=names)


# In[102]:


data1.head()


# In[103]:


peek=data.head(20)


# In[104]:


peek


# In[105]:


data.shape


# In[106]:


types=data.dtypes


# In[107]:


types


# In[115]:


data.describe()
pd.set_option('display.width',1000)
pd.set_option('display.height',10000)


# In[113]:


data.describe()


# In[118]:


data.groupby('class').size()


# In[121]:


data.corr(method='pearson')


# In[123]:


data.skew()


# In[125]:


data.hist()
plt.show()


# In[132]:


data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()


# In[140]:


data.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
plt.show()


# In[152]:


from matplotlib import pyplot
correlations=data.corr()
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(name)
ax.set_yticklabels(name)
plt.show()


# In[159]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
plt.show()

