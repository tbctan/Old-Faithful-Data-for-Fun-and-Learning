#!/usr/bin/env python
# coding: utf-8

# In[42]:



import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,mixture
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


df=[]
df = pd.read_csv('data.tsv',sep="\t")
print(df)

plt.ylabel('Waiting time')
plt.xlabel('Eruption time')
plt.plot(df['eruptions'],df['waiting'],linestyle='',marker='o')
print(df)

#Resources: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# In[43]:



## Initialize: Reset variables

kmfit = None
gmfit = None
dfn=pd.DataFrame()
dfn2=pd.DataFrame()
GM = None
KM = None
gmcenters = None
kmcenters = None



# In[44]:


####### Run 
iter =1
GM = GaussianMixture(n_components=2, init_params='random',max_iter=iter,n_init=1)
KM = KMeans(n_clusters=2,init='random',max_iter=iter,n_init=1)


kmfit = KM.fit(df)
kmcenters=kmfit.cluster_centers_
dfn['kmlabels']=kmfit.labels_
print('')
print('Iter=1')
print('K-Means')
#print(kmfit.labels_)
print('Centers')
print(kmcenters)

print('')

gmfit = GM.fit(df)
gmcenters = gmfit.means_
dfn2['gmlabels']=gmfit.predict(df)
#df['gmlabels'] = gmfit.labels_
#gmcenters=gmfit.cluster_centers_

print('Gaussian Mixture')
print('Weights')
print(gmfit.weights_)
print('Centers')
print(gmcenters)
######## 


colors = ["navy","darkorange"]


plt.rcParams["figure.figsize"] = (15,18)
plt.subplot(2,2,1)

for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[dfn['kmlabels'][i]])
plt.plot(kmcenters[0,0],kmcenters[0,1],linestyle='',marker='x',color='red',markersize=16)
plt.plot(kmcenters[1,0],kmcenters[1,1],linestyle='',marker='x',color='red',markersize=16)
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, K-Means clustering, %s iter' % iter)

plt.subplot(2,2,2)
for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[dfn2['gmlabels'][i]])

plt.plot(gmcenters[0,0],gmcenters[0,1],linestyle='',marker='x',color='red',markersize=16)
plt.plot(gmcenters[1,0],gmcenters[1,1],linestyle='',marker='x',color='red',markersize=16)
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, GM/EM clustering, %s iter' % iter)

print('')


## Initialize for more iterations
iter =75
kmfit = None
gmfit = None
dfn=pd.DataFrame()
dfn2=pd.DataFrame()
GM = None
KM = None
gmcenters = None
kmcenters = None

GM = GaussianMixture(n_components=2, init_params='random',max_iter=iter,n_init=1)
KM = KMeans(n_clusters=2,init='random',max_iter=iter,n_init=1)



####### Run
kmfit = KM.fit(df)
kmcenters=kmfit.cluster_centers_
dfn['kmlabels']=kmfit.labels_
print('')
print('Iter=%s' % iter)
print('K-Means')
#print(kmfit.labels_)
print('Centers')
print(kmcenters)

print('')

gmfit = GM.fit(df)
gmcenters = gmfit.means_
dfn2['gmlabels']=gmfit.predict(df)
#df['gmlabels'] = gmfit.labels_
#gmcenters=gmfit.cluster_centers_

print('Gaussian Mixture')
print('Weights')
print(gmfit.weights_)
print('Centers')
print(gmcenters)
######## 

plt.subplot(2,2,3)

for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[dfn['kmlabels'][i]])
plt.plot(kmcenters[0,0],kmcenters[0,1],linestyle='',marker='x',color='red',markersize=16)
plt.plot(kmcenters[1,0],kmcenters[1,1],linestyle='',marker='x',color='red',markersize=16)
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, K-Means clustering, %s iter' % iter)

plt.subplot(2,2,4)
for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[dfn2['gmlabels'][i]])

plt.plot(gmcenters[0,0],gmcenters[0,1],linestyle='',marker='x',color='red',markersize=16)
plt.plot(gmcenters[1,0],gmcenters[1,1],linestyle='',marker='x',color='red',markersize=16)
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, GM/EM clustering, %s iter' % iter)


print('')


# In[ ]:





# In[ ]:




