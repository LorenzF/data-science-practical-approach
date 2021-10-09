#!/usr/bin/env python
# coding: utf-8

# ## Pivot
# 
# When using the groupby operation we used 1 categorical variable to seperate/group our data into those categories.
# Here we go a step further and use 2 categories to aggregate our data, resulting in a comparison matrix.
# 
# Aside from that, the pivot operation can in general be used to go from a long data format, to a wide data format.
# To keep things uniform we stick with the same cuisine dataset.

# In[1]:


import pandas as pd


# In[2]:


rating_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/rating_final.csv')
rating_df


# And again we merge with the geolocations data, I feel that it becomse obvious here how these operations are very related to eachother.

# In[3]:


geo_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/geoplaces2.csv')


# A subtle difference between last time is that I did not first group per restaurant, however this leads to a dataframe that has a lot of redundant information!
# Try to look in the merged dataframe and spot the copies of data.

# In[4]:


geo_rating_df = pd.merge(rating_df, geo_df, on='placeID')
geo_rating_df


# Now that we have our workable data, we can choose 2 categories and create a comparison matrix using the pivot operation.
# Yet there might be a problem that we still have to resolve, can you figure out the problem reading the error at the end of the stack trace below?

# In[5]:


geo_rating_df.pivot(index='alcohol', columns='smoking_area', values='rating')


# It says: 'Index contains duplicate entries, cannot reshape' meaning that some combinations of our 2 categories, alcohol and smoking area have duplicates, which is understandable.
# I opted to solve this by grouping over the 2 categories and taking the mean for each combination, then i take this grouped data and pivot by setting the alcohol consumption as index and the smoking are as columns.

# In[6]:


grouped_geo_rating_df = geo_rating_df.groupby(['alcohol','smoking_area'])[['rating','food_rating', 'service_rating']].mean().reset_index()
grouped_geo_rating_df.pivot(index='alcohol', columns='smoking_area', values='rating')


# Wonderful! Now we have for each combination an average rating, notice however that not every combination has the same sample size, so comparing might be tricky if you only have a few ratings.
# 
# To figure that out I counted the ratings per combination.

# In[7]:


geo_rating_df.groupby(['alcohol','smoking_area']).count().reset_index().pivot(index='alcohol', columns='smoking_area', values='rating')


# It seems that there might e a correlation between the 2 categories, as a lot of place where smoking is not permitted/none, there is no alcohol served, which makes sense.
# Comparing the ratings with alcohol allowance for places where smoking is not permitted is not a good idea, the counts are 7, 209 and 9, very unbalanced.

# In[8]:


geo_df.columns


# I printed the columns above, perhaps you could figure out a relation between the price category and the (R)ambience of the restaurant?
# Perhaps there are other combinations of which I did not think of, try some out!

# In[ ]:




