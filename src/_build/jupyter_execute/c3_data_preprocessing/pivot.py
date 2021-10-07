#!/usr/bin/env python
# coding: utf-8

# ## Pivot

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('./data/cuisine/rating_final.csv')


# In[3]:


rating_df


# In[4]:


geo_df = pd.read_csv('./data/cuisine/geoplaces2.csv').set_index('placeID')


# In[5]:


merged_rating_df = pd.merge(rating_df, geo_df, on='placeID')
merged_rating_df


# In[6]:


merged_rating_df.pivot(index='alcohol', columns='smoking_area', values='rating')


# In[14]:


merged_rating_df.groupby(['alcohol','smoking_area']).mean().reset_index().pivot(index='alcohol', columns='smoking_area', values='rating')


# In[15]:


merged_rating_df.groupby(['alcohol','smoking_area']).count().reset_index().pivot(index='alcohol', columns='smoking_area', values='rating')


# In[16]:


geo_df.columns


# In[20]:


merged_rating_df.groupby(['city','smoking_area']).mean().reset_index().pivot(index='city', columns='smoking_area', values='rating')


# In[ ]:




