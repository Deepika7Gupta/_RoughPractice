
# coding: utf-8

# In[5]:


import csv
import pandas as pd
filename='E:\\ML\\pima-indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pd.read_csv(filename,names=names)
array=data.values
X=array[:,0:8]
Y=array[:,8]


# In[14]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


# In[35]:


models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
results=[]
names=[]
scoring='accuracy'
for name, model in models:
    kfold=KFold(n_splits=10,random_state=7)
    cv_results=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f (%f)'% (name,cv_results.mean(),cv_results.std())
    print(msg)


# In[46]:


from matplotlib import pyplot
fig=pyplot.figure()
fig.suptitle('Algorithm Comparision')
ax=fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

