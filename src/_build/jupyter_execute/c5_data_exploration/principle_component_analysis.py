#!/usr/bin/env python
# coding: utf-8

# # Principle Component Analysis
# In this notebook we will not try to remove data from our dataset, but transform the variation in our features (columns) into less features.  
# We will do this using the concept of PCA (principle component analysis).
# The dataset we will be using here is about the dimensions of iris flowers, in total 150 flowers were measured of 3 species.

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
iris = sns.load_dataset('iris')


# you can see that we imported a function PCA from sklearn, this will do the calculations for us, but we still need to specify some parameters.  
# Before we do that, let us use the first 2 columns of the dataset to plot a scatter and see if we can distinguish the different species of flowers.

# In[2]:


iris.head()


# In[3]:


sns.scatterplot(x=iris['sepal_length'], y=iris['sepal_width'], hue=iris['species'])


# That already looks pretty good, but versicolor and virginica are still hard to differentiate. Let's see if we can compress the variation of all 4 columns into 2 axi.  
# We do this by creating a PCA transformer and specifying we want only 2 output components

# In[4]:


pca = PCA(n_components=2)


# We also need to prepare our dataframe, we do this by only dropping our outcome (that which we do not need for the transform)

# In[5]:


X = iris.drop(columns='species')
X.head()


# In[6]:


iris_pca = pca.fit_transform(X)
pd.DataFrame(iris_pca, columns=['PC1', 'PC2'])


# Running it through the PCA transformer using the fit_transform function gives us a numpy 2 dimensional array (which is similar to a pandas dataframe) with 2 columns.  
# When inserted into a scatter plot they show us (nearly) all variance of 4 columns compressed into a 2 dimensional plot. 

# In[7]:


sns.scatterplot(x=iris_pca[:,0], y=iris_pca[:,1], hue=iris['species'])


# ## TODO
# it is clear that this function is very potent concerning data visualisation, do you think you can improve on the mpg dataset?
# - experiment with the PCA transformer using the mpg dataset

# In[8]:


mpg = sns.load_dataset('mpg')
mpg.head()


# In[ ]:




