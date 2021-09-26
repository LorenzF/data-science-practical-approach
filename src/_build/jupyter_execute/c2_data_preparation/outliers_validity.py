#!/usr/bin/env python
# coding: utf-8

# ## Outliers and validity

# In[1]:


import pandas as pd


# In[2]:


wafer_df = pd.read_csv('https://openmv.net/file/silicon-wafer-thickness.csv')
wafer_df.head()


# In[3]:


iqr = wafer_df.quantile(0.75)-wafer_df.quantile(0.25)


# In[4]:


range_df = (wafer_df-wafer_df.quantile(0.5))/iqr


# In[5]:


range_df[(range_df>2).any(axis='columns')]


# In[6]:


range_df[(range_df<-2).any(axis='columns')]


# In[7]:


from sklearn.ensemble import IsolationForest


# In[8]:


clf = IsolationForest(random_state=0).fit(wafer_df)
wafer_df[clf.predict(wafer_df)==-1]


# In[ ]:




