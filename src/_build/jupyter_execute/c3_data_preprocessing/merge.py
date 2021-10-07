#!/usr/bin/env python
# coding: utf-8

# ## Merge
# 
# https://www.kaggle.com/uciml/restaurant-data-with-consumer-ratings

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('./data/cuisine/rating_final.csv')


# In[3]:


rating_df


# In[4]:


cuisine_df = pd.read_csv('./data/cuisine/chefmozcuisine.csv')
cuisine_df


# In[5]:


merged_df = pd.merge(rating_df, cuisine_df, on='placeID', how='inner')
merged_df


# In[6]:


merged_df[merged_df.Rcuisine.isna()]


# In[7]:


merged_df.Rcuisine.unique()


# In[8]:


merged_df[merged_df.Rcuisine=='Bar_Pub_Brewery'][['rating', 'food_rating', 'service_rating']].mean()


# In[9]:


merged_df[merged_df.Rcuisine=='Bar'][['rating', 'food_rating', 'service_rating']].mean()


# In[10]:


merged_df[merged_df.Rcuisine=='Cafeteria'][['rating', 'food_rating', 'service_rating']].mean()


# In[11]:


merged_df[merged_df.Rcuisine=='Cafe-Coffee_Shop'][['rating', 'food_rating', 'service_rating']].mean()


# In[12]:


user_payment_df = pd.read_csv('./data/cuisine/userpayment.csv')


# In[13]:


payment_df = pd.merge(rating_df, user_payment_df, how='inner')


# In[14]:


payment_df.Upayment.unique()


# In[15]:


payment_df[payment_df.Upayment=='cash'][['rating', 'food_rating', 'service_rating']].mean()


# In[16]:


payment_df[payment_df.Upayment=='bank_debit_cards'][['rating', 'food_rating', 'service_rating']].mean()


# In[ ]:




