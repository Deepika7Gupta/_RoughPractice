
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


# In[10]:


mylist=[1,2,3]
print('Zeroth value: %d' %mylist[0])
mylist.append(4)
print('len of mylist: %d'%len(mylist))
for values in mylist:
    print(values)


# In[11]:


mydict={'a':1, 'b':2,'c':3}
print('A value : %d' %mydict['a'])
mydict['a']=11
print('A value: %d'%mydict['a'])
print('Print Keys: %s'%mydict.keys())
print('Print values: %s'%mydict.values())
for key in mydict.keys():
    print(key)


# In[12]:


#sum function
def mysum(x,y):
    return x+y

result=mysum(1,3)
print(result)


# In[13]:


#numpy array
import numpy as np
mylist=[1,2,3]
myarray=np.array(mylist)
print(myarray)
print(myarray.shape)


# In[14]:


mylist=[[1,2,3],[4,5,6]]
myarray=np.array(mylist)
print(myarray)
print(myarray.shape)
print('First Row: %s'%myarray[0])
print('last row: %s'%myarray[-1])
print('specific row and column %s'%myarray[0,2])
print('While col: %s'%myarray[:,2])


# In[15]:


myarray1=np.array([1,2,3])
myarray2=np.array([2,2,2])
print('Sum ofd arrays: %s'%(myarray1+myarray2))
print('multiplication of arrays: %s'%(myarray1*myarray2))


# In[16]:


#matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
myarray=np.array([1,2,3])
plt.plot(myarray)
plt.show()


# In[17]:


x=np.array([1,2,3])
y=np.array([2,3,4])
plt.scatter(x,y)
plt.show()


# In[18]:


#pandas
import pandas as pd
myarray=np.array([1,2,3])
rownames=['a','b','c']
myseries=pd.Series(myarray,index=rownames)
print(myseries)


# In[19]:


print(myseries[0])
print(myseries['a'])


# In[21]:


myarray=np.array([[1,2,3],[2,3,4]])
rowname=['a','b']
columname=['one','two','three']
myframe=pd.DataFrame(myarray,index=rowname,columns=columname)
print(myframe)


# In[22]:


print('method 1:')
print('one column: \n%s'%myframe['one'])
print('obe column: \n%s'%myframe.one)


# In[23]:


import csv
filename='E:\\ML\\pima-indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pd.read_csv(filename,names=names)


# In[24]:


print(data.head())
print(data.shape)


# In[25]:


url='https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
data1=pd.read_csv(url,names=names)


# In[26]:


data1.head()


# In[27]:


peek=data.head(20)


# In[28]:


peek


# In[29]:


data.shape


# In[30]:


types=data.dtypes


# In[31]:


types


# In[32]:


data.describe()
pd.set_option('display.width',1000)
pd.set_option('display.height',10000)


# In[33]:


data.describe()


# In[34]:


data.groupby('class')


# In[35]:


data.corr(method='pearson')


# In[36]:


data.skew()


# In[37]:


data.hist()
plt.show()


# In[38]:


data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()


# In[39]:


data.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
plt.show()


# In[40]:


from matplotlib import pyplot
correlations=data.corr()
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[46]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
plt.show()


# In[47]:


array=data.values
X=array[:,0:8]
Y=array[:,8]


# In[48]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
rescaler=scaler.fit_transform(X)
from numpy import set_printoptions
set_printoptions(precision=3)
rescaler[0:5,:]


# In[49]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X)
rescaler=scaler.transform(X)
set_printoptions(precision=3)
rescaler[0:5,:]


# In[50]:


from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(X)
rescaler=scaler.transform(X)
set_printoptions(precision=3)
rescaler[0:5,:]


# In[51]:


from sklearn.preprocessing import Binarizer
binarizer=Binarizer(threshold=0.0).fit(X)
rescaler=binarizer.transform(X)
set_printoptions(precision=2)
rescaler[0:5,:]


# In[52]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(X,Y)
print(fit.scores_)
features=fit.transform(X)
features[0:5,:]


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
model=LogisticRegression()
rfe=RFE(model,3)
fit=rfe.fit(X,Y)
print('Num of feature %d' %fit.n_features_)
print('Seclected features %s'%fit.support_)
print('feature ranking %s'%fit.ranking_)


# In[54]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
fit=pca.fit(X)
print('Explained variance is %s'%fit.explained_variance_ratio_)
print(fit.components_)


# In[55]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)


# In[69]:


from sklearn.model_selection import train_test_split
test_size=0.33
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
model=LogisticRegression()
model.fit(X_train,Y_train)
result=model.score(X_test,Y_test)
print('result is: %.3f%%'%(result*100))


# In[79]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_fold=10
seed=7
model=LogisticRegression()
kfold=KFold(n_splits=num_fold,random_state=seed)
results=cross_val_score(model,X,Y,cv=kfold)
print('Acuuracy: %.3f%% (%.3f%%)'%(results.mean()*100,results.std()*100))


# In[85]:


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
model=LogisticRegression()
loocv=LeaveOneOut()
result=cross_val_score(model,X,Y,cv=loocv)
print('Accuracy is: %.3f%% (%.3f%%)'%(result.mean()*100,result.std()*100))


# In[91]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
model=LogisticRegression()
num_fold=10
test_size=0.33
seed=7
kfold=ShuffleSplit(n_splits=num_fold, test_size=test_size, random_state=seed)
result=cross_val_score(model,X,Y,cv=kfold)
print('Accuract is: %.3f%% (%.3f%%)'%(result.mean()*100, result.std()*100))

