#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt

from sklearn import datasets,mixture
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

df = pd.read_csv('data.tsv',sep="\t")
print(df)

plt.ylabel('Waiting time')
plt.xlabel('Eruption time')
plt.plot(df['eruptions'],df['waiting'],linestyle='',marker='o')


#Resources: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# In[64]:


k_iter=5
g_iter=5


GM = GaussianMixture(n_components=2, init_params='random',max_iter=g_iter)
KM = KMeans(n_clusters=2,max_iter=k_iter)

kmfit = KM.fit(df)

kmcenters=kmfit.cluster_centers_
df['kmlabels']=kmfit.labels_

print('K-Means')
#print(kmfit.labels_)
print(kmcenters)

print('')

gmfit = GM.fit(df)
gmcenters = gmfit.means_
df['gmlabels']=gmfit.predict(df)
#df['gmlabels'] = gmfit.labels_
#gmcenters=gmfit.cluster_centers_

print('Gaussian Mixture')
print(gmfit.weights_)
print(gmcenters)

colors = ["navy","darkorange"]


plt.rcParams["figure.figsize"] = (15,18)
plt.subplot(2,1,1)
for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[df['kmlabels'][i]])
#plt.plot(kmcenters[0],kmcenters[1],linestyle='',marker='+',color='red')
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, K-Means clustering, %s iter' % k_iter)

plt.subplot(2,1,2)
for i in range(0,len(df)):
    plt.plot(df['eruptions'][i],df['waiting'][i],linestyle='',
             marker='o',color=colors[df['gmlabels'][i]])
#plt.plot(kmcenters[0],kmcenters[1],linestyle='',marker='+',color='red')
plt.ylabel('Waiting time')
plt.xlabel('Eruption time, GM/EM clustering, %s iter' % g_iter)


# In[ ]:





# In[ ]:




