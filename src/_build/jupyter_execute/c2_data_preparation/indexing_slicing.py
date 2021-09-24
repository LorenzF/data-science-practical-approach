#!/usr/bin/env python
# coding: utf-8

# ## Indexing and slicing

# In[1]:


import pandas as pd


# In[2]:


min_temp_df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
min_temp_df


# In[3]:


min_temp_df.Date = pd.to_datetime(min_temp_df.Date)


# In[4]:


min_temp_df = min_temp_df.set_index('Date')


# In[5]:


min_temp_df.loc['1989-06-01':'1989-06-30']


# In[6]:


min_temp_df.loc['1989-06-01':'1989-06-30'].mean()


# In[7]:


import seaborn as sns


# In[8]:


tip_df = sns.load_dataset('tips')
tip_df.head()


# In[9]:


tip_index_df = tip_df.set_index('day')


# In[10]:


tip_index_df.loc['Sun']


# In[11]:


tip_index_df = tip_df.set_index(['day','time'])


# In[12]:


tip_index_df.loc[('Thur','Lunch')].tip.mean()


# In[13]:


pd.pivot_table(tip_df, values='total_bill', index='day', columns='time', aggfunc='median')


# In[ ]:




