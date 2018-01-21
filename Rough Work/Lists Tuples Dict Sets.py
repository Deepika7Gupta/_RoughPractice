
# coding: utf-8

# In[1]:


frns=['sam','ank','row']


# In[2]:


sep='*'


# In[3]:


fr=sep.join(frns)


# In[4]:


fr


# In[7]:


se=fr.split(sep)


# In[8]:


se


# In[10]:


sort=sorted(frns)


# In[11]:


sort


# In[12]:


frns.sort()


# In[13]:


frns


# In[14]:


num=[1,5,9.0,4.4]


# In[15]:


pi=sorted(num)


# In[16]:


pi


# In[17]:


num


# In[18]:


num.sort()


# In[19]:


num


# In[20]:


num.sort(reverse=True)


# In[21]:


num


# In[22]:


len(num)


# In[23]:


a=[1,2,3]


# In[25]:


b=a


# In[26]:


b


# In[27]:


a[0]='bdek'


# In[28]:


b


# In[29]:


a


# In[30]:


b[1]='bwkjr'


# In[31]:


a


# In[32]:


b


# In[35]:


c=list(a)


# In[36]:


c


# In[37]:


d=a[:]


# In[38]:


d


# In[39]:


a[2]='erkjng'


# In[40]:


a


# In[41]:


b


# In[42]:


c


# In[43]:


empy= ()


# In[44]:


empy


# In[56]:


ma='sam',


# In[57]:


ma


# In[52]:


gf=('wrbf','nwfkr','nqrl')


# In[53]:


gf


# In[58]:


a,b,c=gf


# In[59]:


a


# In[60]:


b


# In[61]:


c


# In[62]:


a,b,c,d=num


# In[63]:


a


# In[64]:


b


# In[66]:


tuple(num)


# In[69]:


bierce = {"day": "A period of twenty-four hours, mostly misspent","positive": "Mistaken at the top of one's voice","misfortune": "The kind of fortune that never misses",}


# In[70]:


bierce


# In[71]:


di=[['sam','hit'],['viv','sur']]


# In[72]:


dict(di)


# In[75]:


num=[('h','d'),('j','q')]


# In[76]:


dict(num)


# In[77]:


put={'sam':'gaba', 'ank':'sani','awi':'sona'}


# In[78]:


put['deeep']='gupta'


# In[79]:


put


# In[80]:


put['deeep']='gup'


# In[81]:


put


# In[82]:


other={'auy':'dg', 'eqe2':'et'}


# In[84]:


put.update(other)


# In[85]:


put


# In[86]:


del put['auy']


# In[87]:


put


# In[88]:


put.clear()


# In[89]:


put


# In[90]:


other={'auy':'dg', 'eqe2':'et'}


# In[92]:


'auy'in other


# In[93]:


other['eqe2']


# In[94]:


other['ed']


# In[95]:


other.get('eqe2')


# In[96]:


other.keys()


# In[97]:


list(other.keys())


# In[98]:


list(other.values())


# In[99]:


other.values()


# In[100]:


other.items()


# In[101]:


other


# In[102]:


colo={'red':'bad', 'green':'kreen', 'yellow':'mellow'}


# In[103]:


orig=colo.copy()


# In[104]:


colo['voilet']='roilet'


# In[105]:


colo


# In[106]:


orig


# In[107]:


save=colo


# In[108]:


save


# In[110]:


set()


# In[111]:


even={1,2,3,4}


# In[112]:


even


# In[116]:


set(['gjkb','gbk'])


# In[117]:


set('gbk')


# In[118]:


set(save)


# In[120]:


drinks = {
'martini': {'vodka', 'vermouth'},
'black russian': {'vodka', 'kahlua'},
'white russian': {'cream', 'kahlua', 'vodka'},
'manhattan': {'rye', 'vermouth', 'bitters'},
'screwdriver': {'orange juice', 'vodka'}
}


# In[123]:


for name, contents in drinks.items():
    if 'vodka' in contents:
        print(name)


# In[124]:


for name, contents in drinks.items():
    if 'vodka' in contents and not ('vermouth' in contents or 'cream' in contents):
        print(name)


# In[127]:


for name, contents in drinks.items():
    if contents & {'vermouth','orange juice'}:
        print(name)


# In[134]:


for name, contents in drinks.items():
    if 'vodka' in contents and not contents & {'vermouth','cream'}:
        print(name)


# In[137]:


bruss= drinks['black russian']


# In[138]:


bruss


# In[141]:


wruss=drinks['white russian']


# In[142]:


wruss


# In[143]:


a={1,2}
b={2,3}
a&b


# In[144]:


a.intersection(b)


# In[145]:


bruss & wruss


# In[146]:


a|b


# In[147]:


bruss|wruss


# In[148]:


a.union(b)


# In[149]:


a-b


# In[150]:


bruss-wruss


# In[152]:


wruss-bruss


# In[153]:


a^b


# In[155]:


a.symmetric_difference(b)


# In[156]:


a<=b


# In[157]:


a.issubset(b)


# In[158]:


bruss<=wruss


# In[159]:


a<=a


# In[160]:


a.issubset(a)


# In[161]:


a<a


# In[163]:


a>=b


# In[164]:


wruss>=bruss


# In[165]:


a.issuperset(b)


# In[166]:


a>=a


# In[167]:


a.issuperset(a)


# In[168]:


a>b


# In[169]:


max_list=['gup','ghj','gfd']
max_tuple='gup','ghj','gfd'
max_dict={'gup':'ey','ghj':'rfdw','gfd':'wed'}


# In[170]:


max_list[1]


# In[173]:


max_tuple[1]


# In[175]:


max_dict['gup']


# In[176]:


frnds=['sam','mil','ank']
college=['hji','wasf','w']
school=['qe','qeeee','wdf']


# In[177]:


tuple= frnds, college, school


# In[178]:


tuple


# In[179]:


lisr=[frnds,college,school]


# In[180]:


lisr


# In[181]:


dict={'frnds':frnds, 'college':college, 'school':school}


# In[182]:


dict


# In[185]:


year_list=[1996,1997,1998,1999,2000,2001]


# In[186]:


year_list[0]


# In[187]:


year_list[5]


# In[2]:


things=['mozzarella','cindrella','salmonella']


# In[6]:


man=things[0].capitalize()


# In[7]:


sam=things[1].capitalize()


# In[9]:


df=things[2].capitalize()


# In[10]:


things=[man, sam,df]


# In[20]:


things


# In[19]:


things[0]=things[0].upper()


# In[18]:


things


# In[21]:


del things[2]


# In[22]:


things


# In[23]:


surprise=['Groucho','Chico','Harpo']


# In[24]:


surprise[2].lower()


# In[26]:


surprise.sort(reverse=True)


# In[27]:


surprise


# In[29]:


surprise[0].capitalize()


# In[30]:


e2f={'dog':'chien', 'cat':'chat', 'walrus':'morse'}


# In[31]:


e2f


# In[32]:


e2f['walrus']


# In[41]:


f2e=e2f.items()


# In[42]:


f2e


# In[45]:


set(e2f.keys())


# In[47]:


print('No comments! Quoyes make # harmless')


# In[48]:


alpha='rwnkf'+'ref'+'fref'


# In[49]:


alpha


# In[54]:


1+2+3


# In[64]:


disater=True
if disaster:
    print('dq')
else:
    print('rf')


# In[61]:


disaster = True
if disaster:print("Woe!")
else:print("Whee!")


# In[66]:


furry=True
small=True
if furry:
    if small:
        print('its a cat')
    else:
        print('nitnd')
else:
    if small:
        print('wbrnkf')
    else:
        print('wnfclk')


# In[70]:


color='puce'
if color=='red':
    print('it is a tomato')
elif color=='green':
    print('jwedbfk')
elif color=='efv':
    print('fcs')
else:
    print("I dont know",color)


# In[71]:


x=7


# In[72]:


x==5


# In[73]:


x>4


# In[74]:


x>=6


# In[75]:


x<5


# In[76]:


x<=7


# In[77]:


x!=8


# In[78]:


x=6


# In[79]:


2<x<8<89


# In[81]:


sf=[]
if sf:
    print('something')
else:
    print('nothing')


# In[ ]:


count=1
while count<=5:
    print(count)
    count+=1


# In[ ]:


count=1
while count<=5:
    print(count)
    count+=1

