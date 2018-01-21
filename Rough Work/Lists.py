
# coding: utf-8

# In[1]:


empty_list=[]


# In[2]:


week_days=['Monday','Tuesday','Wednesday']


# In[3]:


birds_name=['emu','ostrich','crow']


# In[4]:


first_name=['sam','ank','deep']


# In[5]:


ano_empty_list=list()


# In[6]:


ano_empty_list


# In[7]:


list('cat')


# In[8]:


a_tuple= ('a','cat','exists')


# In[9]:


list(a_tuple)


# In[13]:


birthday='07/04/1996'


# In[14]:


birthday.split('/')


# In[15]:


splt='a/d//s/d//d'


# In[16]:


splt.split('/')


# In[17]:


splt.split('//')


# In[18]:


birds_name[1]


# In[19]:


birds_name[-1]


# In[20]:


college=['ankita','bdkw','rwdbfk']


# In[21]:


school=['Preeti','jgwdbk']


# In[22]:


pg=['sam','wjbfdkw']


# In[23]:


all_frnds=[college, school, pg, 'jwedbk']


# In[24]:


all_frnds


# In[25]:


all_frnds[0]


# In[26]:


all_frnds[1][0]


# In[27]:


school[1]='navneeta'


# In[28]:


school


# In[29]:


college[0:2]


# In[36]:


college[:2]


# In[37]:


college[::2]


# In[38]:


college[::1]


# In[39]:


college[::-1]


# In[40]:


college[::-2]


# In[43]:


college.append('kansha')


# In[44]:


college


# In[45]:


college.extend(school)


# In[46]:


college


# In[47]:


college+=school


# In[48]:


college


# In[49]:


college.append(school)


# In[50]:


college


# In[55]:


college.insert( 2, 'mriti')


# In[56]:


college


# In[57]:


del college[2]


# In[58]:


college


# In[61]:


del college[2]


# In[63]:


college


# In[65]:


college.remove('Preeti')


# In[66]:


college


# In[67]:


college.pop()


# In[68]:


college


# In[69]:


college.pop(1)


# In[71]:


college


# In[72]:


college.index('ankita')


# In[73]:


'kansha' in college


# In[76]:


college.count('ankita')


# In[77]:


','.join(college)

