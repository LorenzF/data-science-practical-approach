#!/usr/bin/env python
# coding: utf-8

# # Case Study: Olympic medals
# 
# In this case study we explore the history of medals in the summer and winter olympics
# 
# The case study is divided into several parts:
# - Goals
# - Parsing
# - Preparation (cleaning)
# - Processing
# - Exploration
# - Visualization
# - Conclusion

# ## Goals
# 
# In this section we define questions that will be our guideline througout the case study
# 
# - Which countries are over-/underperforming?
# - Are some countries exceptional in some sports?
# - Do physical traits have an influence on some sports?
# 
# We'll (try to) keep these question in mind when performing the case study.

# ## Parsing
# 
# we start out by importing all necessary libraries

# In[1]:


import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('svg')


# in order to download datasets from kaggle, we need an API key to access their API, we'll make that here

# In[2]:


if not os.path.exists("/root/.kaggle"):
    os.mkdir("/root/.kaggle")

with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(
        {
            "username":"lorenzf",
            "key":"7a44a9e99b27e796177d793a3d85b8cf"
        }
        , f)


# now we can import kaggle too and download the datasets

# In[3]:


import kaggle
kaggle.api.dataset_download_files(dataset='heesoo37/120-years-of-olympic-history-athletes-and-results', path='./data', unzip=True)


# the csv files are now in the './data' folder, we can now read them using pandas, here is the list of all csv files in our folder

# In[4]:


os.listdir('./data')


# 
# The file of our interest is 'athlete_events.csv', it contains every contestant in every sport since 1896. Let's print out the top 5 events.

# In[5]:


athlete_events = pd.read_csv('./data/athlete_events.csv')
print('shape: ' + str(athlete_events.shape))
athlete_events.head()


# Seems we have a name, gender, age, height and weight of the contestant, as wel as the country they represent, the games they attended located in which city. The last 3 columns specify the sport, event within the sport and a possible medal. Presumably the keeping of their score would have been difficult as different sports use different score metrics which would be hard to compare.

# In[6]:


noc_regions = pd.read_csv('./data/noc_regions.csv')
print('shape: ' + str(noc_regions.shape))
noc_regions.head()


# ## Preparation
# 
# here we perform tasks to prepare the data in a more pleasing format.

# ### Data Types
# 
# Before we do anything with our data, it is good to see if our data types are in order

# In[7]:


athlete_events.info()


# In[8]:


athlete_events[['Sex', 'Team', 'Season', 'City', 'Sport', 'Event']] = athlete_events[['Sex', 'Team', 'Season', 'City', 'Sport', 'Event']].astype('category')
athlete_events.info()


# ### Missing values
# 
# for each dataframe we apply a few checks in order to see the quality of data

# In[9]:


print(100*athlete_events.isna().sum()/athlete_events.shape[0])


# Age, 3.5% missing: 
# 
# Here we can't do much about it, we could impute using mean or median by looking at other contestants from the same sport/event, however I  have a feeling that missing ages might be prevalent in the same sports.
# 

# In[10]:


athlete_events.groupby('Year')['Age'].apply(lambda x: x.isna().sum()).sort_values(ascending=False).head(25)


# In[11]:


athlete_events.groupby('Sport')['Age'].apply(lambda x: x.isna().sum()).sort_values(ascending=False).head(25)


# Although some sports and years are more problematic, we cannot pinpoint a specific group where ages are missing. Imputing with mean or median would drasticly influence the distribution and standard deviation later on. I opt to leave the missing values as is and drop rows with NaN's when using age in calculations. 

# Height & Weight, 22 & 23 % missing:
# 
# Similar to the Age, yet much more are missing, to a point where dropping would become problematic. Let's see if we can find a hotspot of missing data.

# In[12]:


athlete_events.groupby('Year')[['Height', 'Weight']].apply(lambda x: x.isna().sum()).sort_values(by='Height', ascending=False).head(25)


# In[13]:


athlete_events.groupby('Sport')[['Height', 'Weight']].apply(lambda x: x.isna().sum()).sort_values(by='Height', ascending=False).head(25)


# Again, no hotspots. For the same reason (distribution) we will not be imputing values, although for machine learning reasons this might be useful to increase the training pool. We will drop the rows with missing values whenever we use the height/weight columns. It would be wise here to inform our audience that conclusions on this data might be skewed by a possible bias - there might be a reason the data is missing - which might in turn cause us to make a wrongful conclusion!

# Medal, 85% Missing:
# 
# Lastly we see that most are missing the medal, this is obviously that they did not win one. We could boldly assume that since each event has 3 medals, there must be an average of 20 contestants (17/20 = 85%). But this might be deviating over time and sport.

# ### Duplicates
# 
# For any reason, our dataset might be containing duplicates that would be counted twice and will introduce a bias we would not want. On the other hand, duplicates can be subjected to interpretation, here we would say that if 2 records share a name, gender, NOC, Games and event, the rows would be identical.
# This would mean that the person would have performed twice in the same event for the same games under the same flag. The illustration below demonstrates a duplicate.

# In[14]:


athlete_events[athlete_events.Name == 'Jacques Doucet']


# We can se that Jacques for some reason is listed twice for the Sailing Mixed 2-3 Ton event. He won silver, but coming in second is no excused to be listed a second time! Perhaps we can find out where things went wrong by investigating in which year the duplicates appear.

# In[15]:


duplicate_events = athlete_events[athlete_events.duplicated(['Name', 'Sex', 'NOC', 'Games', 'Event'])]
duplicate_events.groupby(['Year'])['Name'].count()


# Seems most of them happen before 1948, perhaps due to errors in manual entries, it feels safe to delete them.

# In[16]:


athlete_events = athlete_events.drop_duplicates(['Name', 'Sex', 'NOC', 'Games', 'Event'])


# ### Indexing
# 
# It is more convenient to work with an index, our dataset already contains an id which we can use as index

# In[17]:


athlete_events = athlete_events.set_index('ID')
athlete_events.head()


# ## Processing

# ### Medals per country per sport
# To find out which country (NOC) performs the best, we would like to have a dataframe with 3 columns ['Gold', 'Silver', 'Bronze'] containing the count of each, as row index, we would have the games and the NOC, thus a multiindex.
# An important detail is that team sports are given multiple medals, as indicated by the exampe below. Be careful as bias might not always as visible.

# In[18]:


athlete_events[(athlete_events.Event == "Basketball Men's Basketball")&(athlete_events.Games=='1992 Summer')&(athlete_events.Medal=='Gold')]


# The preprocessing for this dataframe seem complex but is combination of several operations:
# 
# - drop all records with no medals
# - drop duplicates based on 'Games', 'NOC' , 'Event', 'Medal' to correct for team sports
# - group per 'Games', 'NOC' , 'Medal'
# - aggregate groups by calculating their size
# 
# At this point, we have a single column containing the amount of medals and 3 indices: 'Games' , 'NOC' and 'Medal'
# 
# - unstack the 'Medal' column to obtain 3 columns 'Gold', 'Silver', 'Bronze'
# - make sure the order of columns is 'Gold', 'Silver', 'Bronze'
# - drop rows where no medals are won, as we do not need those rows
# 
# This operation looks like the following:

# In[19]:


medals_country_df = athlete_events.dropna(subset=['Medal']).drop_duplicates(['Games', 'NOC', 'Event']).groupby(['Games', 'NOC', 'Medal', 'Sport']).size().unstack('Medal')[['Gold', 'Silver', 'Bronze']]#.dropna(how='all')#.fillna(0)
medals_country_df = medals_country_df[medals_country_df.sum(axis='columns')>0]
medals_country_df


# ### average statistics per year, country and sport

# In[20]:


avg_stats_df = athlete_events.groupby(['Sex', 'NOC', 'Games', 'Sport'])[['Age', 'Height', 'Weight']].mean().dropna()
avg_stats_df


# ## Exploration

# ## Visualization

# ## Summary

# In[20]:




