#!/usr/bin/env python
# coding: utf-8

# ## Scaling and Normalization
# 
# In this notebook we are going to look into 2 rather mathematical concepts, Scaling and Normalization.
# The idea is that we can scale the values and shape the distribution of feature in our dataset.
# 
# As an example we take a dataset containing samples from a low density polyethylene production process, containing several numerical features such as temperatures, Forces, Pressure,...
# 
# The idea is that by using Scaling and normalization, the 'range of motion' for these sensors is equal and we can compare the fluxtuations not only inbetween records, but also inbetween sensors.

# In[1]:


import pandas as pd


# In[2]:


ldpe_df = pd.read_csv('https://openmv.net/file/LDPE.csv').drop(columns=['Unnamed: 0'])
ldpe_df.head()


# We can see that our features clearly have different ranges, but lets try to visualise these ranges using a density plot

# In[3]:


ldpe_df.plot(kind='density')


# Ouch, this is clearly not working! Because the 'Mw' feature is in the range of 150k-175k our plot is so dilluted the rest are pinned to 0.
# We can use the sklearn library to perform a min max scaling, this scaling will shift the distribution of each feature between 0 and 1, but that can also be adjusted.

# In[4]:


from sklearn.preprocessing import MinMaxScaler


# In[5]:


scaler = MinMaxScaler()
scaler.fit(ldpe_df)
pd.DataFrame(scaler.transform(ldpe_df), columns=ldpe_df.columns).plot(kind='density')


# That makes a lot more sense, you can now see all of the distribution at once.
# Also there seems to be one (yellow) feature that has some outliers perhaps something weird is going on there...
# 
# Taking it a step further we could also alter the distributions by using a standard scaler instead of a min max scaler, redistributing the values mathematically into a normal distribution.

# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()
scaler.fit(ldpe_df)
pd.DataFrame(scaler.transform(ldpe_df), columns=ldpe_df.columns).plot(kind='density')


# You can see it had some trouble fitting our special feature into the normal distribution but it did work out in the end.
# With this we are ready to perform machine learning algorithms on this data, but first why not try and figure out where those outliers are I mentioned earlier?

# In[ ]:




