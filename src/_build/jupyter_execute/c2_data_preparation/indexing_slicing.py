#!/usr/bin/env python
# coding: utf-8

# ## Indexing and slicing

# In[1]:


get_ipython().system('pip install yfinance')
import yfinance as yf
import pandas as pd


# In[2]:


pd.Timestamp.now()-pd.Timedelta(days=100)


# In[3]:


df = yf.download('TSLA', '2020-01-01', '2021-01-01')


# In[4]:


df


# In[5]:


df.set_index('Open')


# In[6]:


df.loc['2020-06-01':'2020-06-30']


# In[7]:


df.loc['2020-05-01':'2020-05-31'].Volume.sum()


# In[8]:


get_ipython().system('pip install seaborn')
import seaborn as sns


# In[9]:


tip_df = sns.load_dataset('tips')
tip_df.head()


# In[10]:


tip_index_df = tip_df.set_index('day')


# In[11]:


tip_index_df.loc['Sun']


# In[12]:


tip_index_df = tip_df.set_index(['day','time'])


# In[13]:


tip_index_df.loc[('Thur','Lunch')].tip.mean()


# In[ ]:




