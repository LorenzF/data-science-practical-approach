#!/usr/bin/env python
# coding: utf-8

# ## Concatenation and deduplication
# 
# In this notebook we are going to investigate the concepts of stitching data files (concatenation) and verifying the integrity of our data concercing duplicates
# 

# ### Concatenation
# 
# When dealing with large amounts of data, fractioning is often the only solution.
# Not only does this tidy up your data space, but it also benefits computation.
# Aside from that, appending new data to your data lake is independent of the historical data.
# However if you want to perform historical analysis this means you will need to perform additional operations.
# 
# In this notebook we have a setup of a very small data lake containing daily minimal temperatures. 
# If you would look closely in the url you would see the following structure.
# 
# 
# > data/temperature/australia/melbourne/1981.csv
# 
# This is a straight-forward but perfect example on how fragmentation works, in our data lake we have:
# temperatures data fractioned by country, city and year. As we are working with daily temperatures further fractioning would not be interesting, but you could fraction e.g. per month.
# 
# In the cells below, we read our both 1981 and 1982 data and concatenate them using python.

# In[1]:


import pandas as pd


# In[2]:


melbourne_1981_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/temperatures/australia/melbourne/1981.csv')


# In[3]:


melbourne_1982_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/temperatures/australia/melbourne/1982.csv')


# In[4]:


df = pd.concat(
    [
        melbourne_1981_df,
        melbourne_1982_df,
    ]
)


# In[5]:


df


# And there you have it! we now have a dataframe containing both data from 1981 as 1982.
# Can you figure out what I calculated in the next cell? Do you think there might be a more 'clean' solution?

# In[6]:


df[df.Date.str[5:7]== '01'].Temp.mean()


# As an exercise I would ask you now to create a small python script that given a begin and end year (between 1981 and 1990) can automatically concatenate all the necessary data

# In[7]:


for i in range(1982,1987):
    print(i)


# ### Deduplication
# 
# Another important aspect of data cleaning is the removal of duplicates.
# Here we fragment of a dataset from activity on a popular games platform.
# We can see which user has either bought or played specific games and how often.
# Unfortunately for some reason, entries might have duplicates which we have to deal with as otherwise users might have e.g. bought a game twice.

# In[8]:


df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/steam.csv')
df


# We have a dataframe with 1839 interactions, you can see that the freq either notes the amount they bought (which always 1 as there is not use in buying it more) or the amount in hours they played.
# 
# Let us straightforward ask pandas to remove all rows that have an exact duplicate

# In[9]:


df.drop_duplicates()


# Alright! this seemed to have dropped 707 rows from our dataset, but we would like to know more about those.
# Let's ask which rows the algorithm has dropped:

# In[10]:


df[df.duplicated()]


# Here we can see the duplicates, no particular pattern seems to be present, we could just for curiosity count the games that are duplicated

# In[11]:


df[df.duplicated()].game.value_counts()


# It seems there are some games which are very prone to being duplicated, at this point we could go and ask the IT department why these games are acting weird.
# 
# Another thing im interested about is the perspective of a single gamer, here we took a single user_id and printed all his games

# In[12]:


df[df.user_id == 11373749]


# Ah, you can see all of his three games are somehow duplicated in purchase, also it seems he only played one of them for only 0.1 hours. 
# Looks like he fell to the bait of a tempting summer sale but didn't realise he had no time to actually play it.
# 
# Another thing I would like to mention here is that this dataset would make a fine recommender system as it contains user ids and hours played.
# Add game metadata (description) and reviews to the mix and your data preparation is done!
# 
# We can remove all duplicates now by overwriting our dataframe

# In[13]:


df = df.drop_duplicates()


# One thing still bothers me, as hours played can change over time it might be that different snapshots have produced different values, therefore more duplicates might be present with different hours_played.
# 
# Time to investigate this by using a subset of columns in the drop_duplicates algorithm

# In[14]:


df.drop_duplicates(subset=['user_id', 'game', 'action'])


# Seems we have shaved off another 12 records, so our intuition was right, again lets see which the duplicates are:

# In[15]:


df[df.duplicated(subset=['user_id', 'game', 'action'])]


# As expected the duplicates are all in the 'play' action, to complete our view we extract the data of a single user

# In[16]:


df[df.user_id==118664413]


# It looks like we have a problem now, we know these are duplicates and should be removed, but which one?
# Personally I would argue here that we keep the highest value, as it is impossible to 'unplay' hours on the game.
# I will leave this as an exercise for you, but the solution is pretty tricky so i'll give a hint:
# 
# The algorithm always keeps the first record in case of duplicates, so you could sort the rows making sure the higher value is always encountered first, good luck!

# In[ ]:




