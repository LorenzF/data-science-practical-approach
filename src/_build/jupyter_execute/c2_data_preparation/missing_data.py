#!/usr/bin/env python
# coding: utf-8

# ## Missing Data
# 
# this is a notebook about missing data

# In[1]:


variable = 'test'


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('https://openmv.net/file/kamyr-digester.csv')
df.head()


# In[4]:


df.isna().sum()


# In[32]:





# In[5]:


df.ffill()['SulphidityL-4 ']


# In[6]:


df = pd.read_csv('https://openmv.net/file/travel-times.csv')
df


# In[7]:


df.isna().sum()


# In[8]:


df[~df.Comments.isna()]


# In[9]:


df.loc[df.Comments.isna(),'Comments'] = ''


# In[10]:


df.Comments


# In[11]:


df[~df.FuelEconomy.isna()]


# In[12]:


df.info()


# In[13]:


df.FuelEconomy = pd.to_numeric(df.FuelEconomy, errors='coerce')


# In[14]:


df.info()


# In[15]:


df[~df.FuelEconomy.isna()]


# In[16]:


df = pd.read_csv('http://openmv.net/file/raw-material-properties.csv')
df.head()


# In[17]:


get_ipython().system('pip install sklearn')
from sklearn.impute import KNNImputer


# In[18]:


imputer = KNNImputer(n_neighbors=5, weights="distance")


# In[19]:


pd.DataFrame(
    imputer.fit_transform(df.drop(columns=['Sample'])), 
    columns=df.columns.drop('Sample')
)


# In[ ]:




