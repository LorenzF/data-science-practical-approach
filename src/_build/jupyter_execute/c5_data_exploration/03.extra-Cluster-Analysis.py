#!/usr/bin/env python
# coding: utf-8

# # Cluster analysis
# Before starting this notebook I would like to state that what is explained here will be elaborated later in the course and might look complicated at this point. If you do not feel familiar with these concepts that is perfectly fine, everything will become more clear later.

# In[1]:


import pandas as pd
import seaborn as sns


# We will load a digits dataset from sklearn, the machine learning library, these are 8x8 pixel images showing handwritten digits with the correct answer.  
# In the dataset there are 1797 images giving the dataset a dimension of (1797, 8*8)

# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# Before we start, let's print out a few of them, the following cell will do that.
# Again, plotting is not yet seen, so the following cells might be overwhelming.

# In[3]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


# In cluster analysis we will try to figure out clusters within the dataset, keep in mind that these cluster are constructed without knowning the correct answer.
# Here we use the Isomap algorithm to create clusters, by using fit and transform methods we can create the clusters

# In[4]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape


# Now that the algorithm seperated the dataset into 10 clusters, we can ask it to print the center of each cluster.  
# This gives us an idea how the average digit in that cluster looks like.

# In[5]:


fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# Those look similar to the actual numbers, confirming that arabic numbers have good visual seperation inbetween.  
# Aside from the centers we can also print a few examples from the clusters.

# In[6]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[clusters==i%10][int(i/10)], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(i%10),
            transform=ax.transAxes, color='green')


# You can see that the cluster number does not match the actual number, that's because our algorithm does not understand which numbers there are.  
# It does however understand the differences between the numbers!
# This technique can also be used for other datasets where no outcome is given, but we would like to separate our dataset into clusters.

# In[ ]:




