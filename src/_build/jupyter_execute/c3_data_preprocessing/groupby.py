#!/usr/bin/env python
# coding: utf-8

# ## Groupby

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('./data/cuisine/rating_final.csv')


# In[3]:


grouped_rating_df = rating_df.groupby('placeID').mean().sort_values('rating')
grouped_rating_df


# In[4]:


geo_df = pd.read_csv('./data/cuisine/geoplaces2.csv').set_index('placeID')


# In[5]:


merged_rating_df = pd.merge(grouped_rating_df, geo_df, left_index=True, right_index=True)
merged_rating_df


# In[6]:


merged_rating_df.columns


# In[7]:


merged_rating_df.groupby('country')[['rating', 'food_rating', 'service_rating']].mean()


# In[8]:


get_ipython().set_next_input('can you fix this string problem');get_ipython().run_line_magic('pinfo', 'problem')


# In[9]:


merged_rating_df.groupby('alcohol')[['rating', 'food_rating', 'service_rating']].mean()


# In[10]:


merged_rating_df.groupby('accessibility')[['rating', 'food_rating', 'service_rating']].mean()


# In[11]:


merged_rating_df.accessibility.value_counts()


# In[12]:


merged_rating_df.groupby('price')[['rating', 'food_rating', 'service_rating']].mean()


# In[13]:


get_ipython().set_next_input('can you solve the mean-mean problem');get_ipython().run_line_magic('pinfo', 'problem')


# In[14]:


get_ipython().set_next_input('in the merge example we added the cuisine type, could you perform a groupby analysis on this');get_ipython().run_line_magic('pinfo', 'this')

