
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[3]:


from pprint import pprint


# In[131]:


import sklearn


# In[132]:


from sklearn import ensemble, linear_model, svm, tree,naive_bayes,neighbors


# In[133]:


import subprocess


# In[134]:


df_train=pd.read_csv('E:\\ML\\train.csv')
df_test=pd.read_csv('E:\\ML\\test.csv')
combine=[df_train,df_test]


# In[135]:


df_train.head()


# In[136]:


df_train.tail()


# In[137]:


print('#print Features')
print(df_train.columns.values)
print('_'*40)
print('#data types')
df_train.info()
print('_'*40)
df_test.info()


# In[138]:


df_train.describe()


# In[139]:


df_train.describe(include=['O'])


# In[140]:


def chance_to_survive_by_feature(feature):
    return df_train[[feature,'Survived']].groupby([feature]).mean().sort_values(by='Survived',ascending=False)


# In[141]:


chance_to_survive_by_feature('Pclass')


# In[142]:


chance_to_survive_by_feature('Sex')


# In[143]:


chance_to_survive_by_feature('SibSp')


# In[144]:


chance_to_survive_by_feature('Parch')


# In[145]:


g=sns.FacetGrid(df_train,col='Survived')
g.map(plt.hist, 'Age', bins=20 )


# In[146]:


g=sns.FacetGrid(df_train,col='Survived', row='Pclass')
g.map(plt.hist, 'Age', bins=20)


# In[147]:


chance_to_survive_by_feature('Embarked')


# In[148]:


ordered_embarked= df_train.Embarked.value_counts().index


# In[149]:


ordered_embarked


# In[150]:


grid=sns.FacetGrid(df_train, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')
grid.add_legend()


# In[151]:


grid=sns.FacetGrid(df_train, row='Embarked', col= 'Survived')
grid.map(sns.barplot, 'Sex','Fare')


# In[152]:


print('Before', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)
df_train= df_train.drop(['Ticket','PassengerId','Cabin'], axis=1)
df_test=df_test.drop(['Ticket','Cabin'], axis=1)
combine=[df_train, df_test]
print('After', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)


# In[153]:


for dataset in combine:
    dataset['Title']= dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)


# In[154]:


pd.crosstab(df_train['Title'], df_train['Sex'])


# In[155]:


for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Major','Lady','Rev','Sir'],'Rare')
    dataset['Title']=dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')


# In[156]:


df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[157]:


title_mapping={'Master':1, 'Miss':2,'Mr':3,'Mrs':4,'Rare':5}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
    


# In[158]:


df_train.head()


# In[159]:


df_train=df_train.drop(['Name'], axis=1)
df_test=df_test.drop(['Name'],axis=1)
combine=[df_train,df_test]
df_train.shape, df_test.shape


# In[160]:


for dataset in combine:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0})


# In[161]:


df_train.head()


# In[162]:


grid=sns.FacetGrid(df_train, row= 'Pclass', col='Sex')
grid.map(plt.hist, 'Age')
grid.add_legend()


# In[163]:


guess_ages=np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):
        for j in  range(0,3):
            df_guess=dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            age_guess=df_guess.median()
            guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5


# In[164]:


for i in range(0,2):
    for j in range(0,3):
        dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1),'Age']==guess_ages[i,j]
        dataset['Age']=dataset['Age']


# In[165]:


df_train['Ageband']= pd.cut(df_train['Age'],5)


# In[166]:


df_train[['Ageband','Survived']].groupby(['Ageband'],as_index=False).mean()


# In[167]:


for dataset in combine:
    dataset.loc[(dataset['Age']<=16),'Age']==0
    dataset.loc[(dataset['Age'])>16 & (dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']>=64),'Age']=3
    dataset.loc[(dataset['Age']>64),'Age']
    


# In[1]:


combine


# In[168]:


df_train.head()


# In[169]:


df_train=df_train.drop(['Ageband'],axis=1)


# In[170]:


combine=[df_train,df_test]
df_train.head()


# In[171]:


for dataset in combine:
    dataset['Familysize']= dataset['SibSp']+dataset['Parch']+1
    


# In[172]:


df_train[['Familysize','Survived']].groupby(["Familysize"], as_index=False).mean().sort_values(by="Survived")


# In[173]:


for dataset in combine:
    dataset['Isalone']=0
    dataset.loc[dataset["Familysize"]==1,'Isalone']=1


# In[174]:


df_train[['Isalone','Survived']].groupby(['Isalone'], as_index=False).mean()


# In[175]:


df_train.head()


# In[176]:


df_train=df_train.drop(['SibSp','Parch','Familysize'], axis=1)
df_test=df_test.drop(['SibSp','Parch','Familysize'], axis=1)


# In[177]:


combine=[df_train,df_test]


# In[178]:


df_train.head()


# In[179]:


for dataset in combine:
    dataset['Age*class']=dataset.Age*dataset.Pclass


# In[180]:


df_train.loc[:,['Age*class','Age',"Pclass"]].head(10)


# In[181]:


freq_port=df_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset["Embarked"]=dataset['Embarked'].fillna(freq_port)


# In[182]:


df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()


# In[183]:


for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].map({'C':0,'Q':1,'S':2})


# In[184]:


df_train.head()


# In[185]:


df_test['Fare'].fillna(df_test['Fare'].dropna().median(),inplace=True)


# In[186]:


df_test.head()


# In[187]:


df_train['Fareband']=pd.qcut(df_train['Fare'],4)


# In[188]:


df_train[['Fareband','Survived']].groupby(['Fareband'],as_index=False).mean()


# In[189]:


for dataset in combine:
    dataset.loc[dataset['Fare']<=7.91,'Fare']=0
    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454),'Fare' ]=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare']=2
    dataset.loc[(dataset['Fare']>31),'Fare']=3
    dataset['Fare']=dataset['Fare'].astype(int)
    


# In[190]:


df_train=df_train.drop(['Fareband'], axis=1)
combine=[df_train,df_test]


# In[191]:


df_train.head()


# In[194]:


models=[]
models.append({
    'classifier':linear_model.LogisticRegression,
    'name':'Logistic Regression',
})
models.append({
    'classifier':svm.SVC,
    'name':'Support Vector Machine',
})
models.append({
    'classifier':neighbors.KNeighborsClassifier,
    'name':'K Nearest Neighbors',
    'args':{
        "n_neighbors":3,
    },
})
models.append({
    'classifier':naive_bayes.GaussianNB,
    'name':'Gaussian Naive Bayes',
})
models.append({
    'classifier':linear_model.Perceptron,
    'names':'Perceptron',
    'arges':{
        'max_iter':5, 
        'tol':None,
    },
})
models.append({
    'classifier':tree.DecisionTreeClassifier,
    'name':'Decision Tree',
})
models.append({
    'classifier':ensemble.RandomForestClassifier,
    'name':'Random Forest',
    'args':{
        'n_estimators':100,
    },
})


# In[196]:


X_train=df_train.drop('Survived',axis=1)
Y_train=df_train['Survived']


# In[197]:


X_test=df_test.drop('PassengerId',axis=1).copy()


# In[200]:


X_train.shape,Y_train.shape,X_test.shape


# In[202]:


def process_model(model_desc):
    Model=model_desc['classifier']
    model=Model(**model_desc.get('args',{}))
    model.fit(X_train, Y_train)
    Y_pred=model.predict(X_test)
    accuracy=round(model.score(X_train, Y_train)*100,2)
    return {
        'name': model_desc['name'],
        'accuracy':accuracy,
        'model':model,      
    }


# In[205]:


models_result= list(map(process_model,models)).a


# In[206]:


models_result=sorted(models_result, key=lambda res:res['accuracy'], reverse=True)
models_result_df=pd.DataFrame(models_result, columns=['accuracy','name'])
ax=sns.barplot(data=models_result_df, x='accuracy',y='name')
ax.set(xlim=(0,100))
models_result_df

