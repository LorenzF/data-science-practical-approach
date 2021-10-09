#!/usr/bin/env python
# coding: utf-8

# ## Groupby
# 
# In the previous section we saw how to combine information of multiple tables from our dataset.
# Here we are going to build further on that by using the merged information to group on categorical variables.

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/rating_final.csv')
rating_df


# Again we have our rating data containing the users, places and ratings they gave.
# As a simple example we could just group by the placeID column and take the mean, this would give us the mean rating for each restaurant

# In[3]:


grouped_rating_df = rating_df.groupby('placeID').mean().sort_values('rating')
grouped_rating_df


# Keep in mind that this might be tricky, as we do not always have as much records per group, we could count the amount per records using a groupby operation and count.

# In[4]:


rating_df.groupby('placeID').rating.count()


# Taking an average of 4 ratings might not be ideal, so we should keep in mind that our groups have a good sample size.
# 
# Let's make things more interesting and insert some location data.

# In[5]:


geo_df = pd.read_csv('./data/cuisine/geoplaces2.csv').set_index('placeID')
geo_df


# Here we have for each restaurant information about its location, I mentioned earlier that grouping per restaurant might be dangerous as some restaurants have nearly no reviews.
# By adding information such as city, state and country we have other categorical variables to group by.
# Notice how we use the merge operation from previous section, but this time specify our common key is the index.

# In[6]:


geo_rating_df = pd.merge(grouped_rating_df, geo_df, left_index=True, right_index=True)
geo_rating_df


# By adding this amount of data, things are getting a bit cluttered, thankfully we can use pandas to get a list of all our columns.

# In[7]:


geo_rating_df.columns


# How about we try and see if we can find a difference between countries for the ratings?

# In[8]:


geo_rating_df.groupby('country')[['rating', 'food_rating', 'service_rating']].mean()


# Ah, it seems we forgot to do some data cleaning here, perhaps you could jump in and fix this string problem, might as well tackle the missing value while we are at it.
# Aside from that, we can see that lower-case Mexico is not doing very well, perhaps the food was so bad they forgot how to write Mexico?

# In[ ]:





# Jokes aside, do you see the ressemblance between this and our rudimentary approach of comparing different categories?
# We are slowly getting more and more efficient using these operations, how about the difference between alcohol consumption?

# In[9]:


geo_rating_df.groupby('alcohol')[['rating', 'food_rating', 'service_rating']].mean()


# Something we can remark here is that the food rating for no alcohol locations seems to be holding up, whilst the general rating and service rating fall behind.
# This would suggest that the food rating indeed is for the food, where the type of drinks served have no influence.
# 
# As a last we look at the difference between accessibility, does that influences our ratings?

# In[10]:


geo_rating_df.groupby('accessibility')[['rating', 'food_rating', 'service_rating']].mean()


# It seems having partial accessibility is the way to go here, performing better than complete accessibility.
# We can however find that is due to a low sample size of 9 restaurants, making it prone to variation.

# In[11]:


geo_rating_df.accessibility.value_counts()


# You should get the hang of it by now, perhaps you can play some more with the other categories.
# 
# There is one thing I still would like to address, you perhaps have notices that in the beginning I first took the average rating per restaurant and later again took the average per category.
# This is a bad practice as a bad restaurant with one review has equal influence as a good restaurant with 100 reviews, perhaps you can think of a way to group all reviews from a category instead of the average for each restaurant?

# In[ ]:





# In the previous section we added the cuisine type, perhaps you could do some groupby operations on that too here?

# In[ ]:




