
# coding: utf-8

# In[2]:


a= [1,2,3,4]


# In[4]:


import numpy as np


# In[5]:


np.array(a)


# In[6]:


s= [[1,2,3,4],[5,6,7,9]]


# In[9]:


d=np.array(s)


# In[10]:


d.ndim


# In[11]:


d.dtype


# In[12]:


d.shape


# In[14]:


np.zeros(10)


# In[15]:


np.zeros((4,2))


# In[17]:


np.empty(2)


# In[18]:


np.empty(2)


# In[19]:


np.arange(15)


# In[20]:


np.asarray(s)


# In[22]:


np.asarray(d)


# In[23]:


np.ones(2)


# In[28]:


np.eye((4))


# In[30]:


x=np.array([1,2,3], dtype=np.float64)


# In[31]:


x


# In[32]:


c=np.array([1,2,3], dtype=np.int32)


# In[33]:


c


# In[34]:


c_f=c.astype(np.float64)


# In[35]:


c_f


# In[36]:


r=np.array([1.2,5.6,4,8])


# In[37]:


r


# In[38]:


r.dtype


# In[39]:


r_f=r.astype(np.int32)


# In[40]:


r_f


# In[43]:


num=np.array(['1','2','4'], dtype=np.string_)


# In[45]:


num.dtype


# In[49]:


num.astype(np.float64)


# In[50]:


no=np.array([1,2,3,4])


# In[51]:


yes=np.array([2,4,6], dtype=np.float)


# In[52]:


no.astype(yes.dtype)


# In[75]:


arr=np.arange(10,dtype=np.float)


# In[76]:


arr[5]


# In[77]:


arr[5:8]


# In[78]:


arr[5:8]=12


# In[79]:


arr


# In[80]:


arr_slice=arr[5:8]


# In[81]:


arr_slice


# In[82]:


arr_slice[1]=12345


# In[83]:


arr


# In[86]:


arr_slice[:]=64


# In[87]:


arr


# In[88]:


arr.astype(int)


# In[92]:


aj=np.array([[1,2,3],[4,5,6],[6,7,8]])


# In[93]:


aj


# In[94]:


aj[1]


# In[96]:


aj[1][2]


# In[97]:


aj[1,2]


# In[99]:


ak=np.array([[[1,2,3],[23,5,4]],[[2,5,3],[6,9,4]]])


# In[100]:


ak


# In[101]:


ak.shape


# In[102]:


ak[0]


# In[119]:


old=ak[0].copy()


# In[104]:


ak[0]=42


# In[111]:


ak


# In[112]:


old


# In[117]:


ak[0]=3


# In[118]:


ak


# In[120]:


ak[0]=old


# In[121]:


ak


# In[123]:


ak[1,0]


# In[129]:


q=np.array([[1,2,3],[3,4,6]])


# In[130]:


q[:2]


# In[134]:


q[1,0]


# In[135]:


q[:,:1]


# In[140]:


names=np.array(['bob','billy','joe','bob','billy','joe'], dtype=np.string_)


# In[141]:


names


# In[149]:


data=np.random(7,4)


# In[143]:


data


# In[150]:


from numpy import randn


# In[151]:


names=='bob'


# In[153]:


data=np.random.randn(7,4)


# In[154]:


data


# In[156]:


data[names=='bob']


# In[158]:


mask=(names=='bob')|(names=='joe')


# In[159]:


data[mask]


# In[160]:


mask


# In[161]:


data


# In[163]:


data[data<0]==0


# In[164]:


data


# In[166]:


data[data<0]=0


# In[167]:


data


# In[168]:


data[names!='Joe']=7


# In[169]:


data


# In[171]:


ar=np.empty((8,4))


# In[172]:


ar


# In[173]:


for i in range(8):
    ar[i]=i


# In[174]:


ar


# In[176]:


ar[[5,2,0,6]]


# In[177]:


av=np.arange(32).reshape(8,4)


# In[178]:


av


# In[185]:


av[[1,5,3],[3,3,2]]


# In[189]:


av[[1,5,3]][:,[3,3,2]]


# In[191]:


av[np.ix_([1,5,3],[3,3,2])]


# In[195]:


we=np.arange(15).reshape((3,5))


# In[196]:


we


# In[197]:


we.T


# In[200]:


v=np.random.randn(2,6)


# In[201]:


np.dot(v.T,v)


# In[215]:


b=np.array([[2,2],[1,1
                  ]])


# In[216]:


b*b


# In[217]:


b


# In[218]:


np.dot(b,b)


# In[220]:


mn= np.arange(16).reshape(2,2,4)


# In[221]:


mn


# In[224]:


u=mn.T


# In[230]:


mn.swapaxes(1,2)


# In[231]:


b=np.arange(4)


# In[232]:


np.sqrt(b)


# In[233]:


b


# In[236]:


np.sqrt(3)


# In[237]:


np.exp(b)


# In[240]:


np.exp(1)


# In[243]:


x=np.random.randn(8)


# In[245]:


y=np.random.randn(8)


# In[246]:


x


# In[247]:


y


# In[248]:


np.maximum(x,y)


# In[249]:


am=np.random.randn(7)*5


# In[250]:


am


# In[251]:


np.modf(am)


# In[252]:


np.abs(am)


# In[253]:


np.square(am)


# In[254]:


np.log(am)


# In[255]:


np.log10(am)


# In[256]:


np.sign(am)


# In[257]:


np.ceil(am)


# In[258]:


np.floor(am)


# In[259]:


am


# In[260]:


np.rint(am)


# In[261]:


np.isnan(am)


# In[262]:


np.isinf(am)


# In[263]:


np.isfinite(am)


# In[264]:


sin(1)


# In[265]:


np.sin(1)


# In[267]:


np.arcsin(1)


# In[272]:


np.logical_not(10)


# In[273]:


x=np.array([1,2,5])


# In[276]:


y=np.array([6,3,8])


# In[277]:


np.add(x,y)


# In[278]:


np.subtract(x,y)


# In[279]:


np.divide(x,y)


# In[280]:


np.multiply(x,y)


# In[281]:


np.power(x,y)


# In[282]:


np.maximum(x,y)


# In[283]:


x


# In[284]:


y


# In[285]:


np.fmax(x,y)


# In[286]:


np.fmin(x,y)


# In[287]:


np.mod(x,y)


# In[288]:


w=[1,2,3]
t=[-5,8,-4]


# In[289]:


np.copysign(w,t)


# In[290]:


np.greater(w,t)


# In[291]:


sd=np.arange(-5,5,0.01)


# In[303]:


xs, ys=np.meshgrid(sd,sd)


# In[304]:


ys


# In[305]:


xs


# In[312]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[308]:


z=np.sqrt(xs**2+ys**2)


# In[309]:


z


# In[318]:


plt.imshow(z, cmap=plt.cm.gray);plt.colorbar();plt.title('Image')


# In[322]:


xarr=[2.4,6.5,7.3]
yarr=[5.2,7.4,6.9]
boo=[True,False,True]


# In[323]:


result=[(x if c else y) for x,y,c in zip(xarr,yarr,boo)]


# In[324]:


result


# In[325]:


np.where(boo,xarr,yarr)


# In[327]:


np.where(boo,6,4)


# In[329]:


rt=np.random.randn(4,4)


# In[330]:


rt


# In[333]:


np.where(rt>0,-1,1)


# In[336]:


np.where(rt>0,2,rt)


# In[344]:


cond1=[True,False,False]
cond2=[False,True,True]


# In[340]:


result = []
for i in range(n):
if cond1[i] and cond2[i]:
result.append(0)
elif cond1[i]:
result.append(1)
elif cond2[i]:
result.append(2)
else:
result.append(3)


# In[342]:


np.where(cond1&cond2,0,np.where(cond1,1,np.where(cond2,2,3)))


# In[345]:


result=cond1*1+cond2*2+3*-(cond1|cond2)


# In[346]:


array=np.random.randn(3,4)


# In[347]:


array


# In[348]:


array.mean()


# In[349]:


d=np.array([[1,2,3],[2,3,4]])


# In[350]:


d


# In[351]:


d.mean()


# In[352]:


d.sum()


# In[353]:


np.mean(d)


# In[356]:


d.mean(axis=0)


# In[355]:


d.sum(0)


# In[357]:


d.cumsum(0)


# In[358]:


d


# In[359]:


d.cumsum(1)


# In[360]:


d.cumprod(0)


# In[362]:


d.cumprod(1)


# In[363]:


np.zeros(4).mean()


# In[367]:


d.std(axis=0)


# In[368]:


d.var(axis=1)


# In[369]:


d.min()


# In[371]:


d.max()


# In[372]:


d.argmin()


# In[373]:


d.argmax()


# In[374]:


ace=np.random.randn(8)


# In[376]:


(ace>0).sum()


# In[377]:


bools=np.array([True,True,False,False])


# In[378]:


bools.any()


# In[379]:


bools.all()


# In[380]:


a.sort()


# In[381]:


ds=np.random.randn(3,8)


# In[382]:


ds


# In[387]:


ds.sort()


# In[388]:


ds


# In[389]:


ds.sort(1)


# In[390]:


ds


# In[391]:


gf=np.arange(90)


# In[396]:


gf[int(0.05*len(gf))]


# In[397]:


names=np.array(['bob','bob','willi','willi','rob','rob'])


# In[398]:


np.unique(names)


# In[399]:


sorted(list(names))


# In[400]:


sorted(set(names))


# In[401]:


x


# In[403]:


x=np.array([2,8,6,9,6])


# In[407]:


y=np.array([3,8,9,5,0,0])


# In[406]:


np.unique(x)


# In[410]:


np.intersect1d(x,y)


# In[413]:


np.union1d(x,y)


# In[415]:


np.setdiff1d(x,y)


# In[417]:


np.setxor1d(x,y)


# In[418]:


nb=np.arange(10)


# In[419]:


np.save('some',nb)


# In[421]:


np.load('some.npy')


# In[423]:


get_ipython().system('cat array_ex.txt')


# In[424]:


x=np.array([[1,2,3],[3,4,5]])


# In[425]:


y=np.array([[1,3],[4,6],[7,1]])


# In[426]:


x


# In[427]:


y


# In[428]:


np.dot(x,y)


# In[429]:


x.dot(y)


# In[430]:


np.ones(3)


# In[431]:


np.dot(x,np.ones(3))


# In[434]:


x=np.random.randn(5,5)


# In[437]:


inv(x)


# In[457]:


from numpy.linalg import inv,qr,det


# In[438]:


mat=x.T


# In[439]:


mat


# In[440]:


x


# In[445]:


mat1=x.T.dot(x)


# In[446]:


mat


# In[447]:


inv(mat1)


# In[448]:


mat1.dot(inv(mat1))


# In[450]:


q,r=qr(mat1)


# In[451]:


r


# In[452]:


q


# In[454]:


np.diag(x)


# In[455]:


np.trace(x)


# In[461]:


det(x)


# In[464]:


eig(x)


# In[465]:


inv(x)


# In[475]:


from numpy.linalg import eig,pinv,svd


# In[469]:


np.trace(x)


# In[472]:


pinv(x)


# In[474]:


qr(x)


# In[476]:


svd(x)


# In[479]:


solve(x*a==0)


# In[478]:


from numpy.linalg import solve


# In[483]:


samples=np.random.normal(size=(4,4))


# In[484]:


samples

