#!/usr/bin/env python
# coding: utf-8

# ## Merge
# 
# When data becomes multi-dimensional - covering multiple aspects of information - it usually happens that a lot of information is redundant.
# Take for example the next dataset, we have collected ratings of restaurants from users, when a single user rates 2 restaurants the information of the user relates to both rows, yet it would be wasteful to keep this info twice.
# The same can happen when we have a restaurant with 2 ratings, the location of the restaurant is kept twice in our data, which is not scalable.
# 
# We solve this problem using relational data, the idea is that we have a common key column in 2 of our tables which we can use to join the data for further processing.
# 
# In our example we use a dataset with consumers, restaurants and ratings between those, you can find more information [here](https://www.kaggle.com/uciml/restaurant-data-with-consumer-ratings).

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/rating_final.csv')
rating_df


# this first table we read contains the userID from whom the rating came, the placeID is the restaurant he/she rated and the numerical values of the 3 different ratings.
# 
# Perhaps you can find out what the min and max values for the ratings are?

# In[ ]:





# to know the type of restaurant, we can not read another table

# In[3]:


cuisine_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/chefmozcuisine.csv')
cuisine_df


# This table also contains the placeID, so we should be able to merge/join these 2 tables and create a new table with info of both.
# Notice how we specify the 'on' parameter where we denote placeID as our common key.

# In[4]:


merged_df = pd.merge(rating_df, cuisine_df, on='placeID', how='inner')
merged_df


# Great! now we have more info about the rating that were given, being the type of cuisine that they rated.
# We could figure out which cuisines are available in our dataset and do a comparison, let us count the occurences of each cuisine.

# In[5]:


merged_df.Rcuisine.value_counts()


# A lot of mexican, which is not surpising as this dataset comes from Mexico.
# I wonder if there is a difference between 'Bar' and 'Bar_Pub_Brewery', we can see if the average rating for those 2 differ.

# In[6]:


for cuisine in ['Bar', 'Bar_Pub_Brewery']:
    print(cuisine)
    print(merged_df[merged_df.Rcuisine==cuisine][['rating', 'food_rating', 'service_rating']].mean())
    print()


# just looking at the averages we can deduces that while food ratings do not change a lot, the service seems a lot better at the Brewery.
# 

# In[7]:


merged_df[merged_df.Rcuisine=='Cafeteria'][['rating', 'food_rating', 'service_rating']].mean()


# In[8]:


merged_df[merged_df.Rcuisine=='Cafe-Coffee_Shop'][['rating', 'food_rating', 'service_rating']].mean()


# As easy as it looks, we can now merge information of different tables in our dataset and perform some simple comparisons, in later sections we will see how we can improve on those.
# 
# As an exercise I already read in the table containing the info about which type of payment the user has opted for.
# Could you find out if the type of payment could have an influence on the rating?

# In[9]:


user_payment_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/userpayment.csv')
user_payment_df


# In[ ]:




