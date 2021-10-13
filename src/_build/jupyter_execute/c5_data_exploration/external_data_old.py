#!/usr/bin/env python
# coding: utf-8

# # New Data Sources
# In this notebook we are going to look into adding new data to your dataset.
# We start out with a taxi dataset describing all pickup points from taxis in a specific date interval, notice that the dataset is divided up into months. Each month has their specific csv file saved in an AWS location.
# 
# 
# 

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_style()
sns.set(rc={'figure.figsize':(16,12)})


# In[2]:


taxi_df = sns.load_dataset('taxis')
taxi_df.head()


# We would like to find out how many taxi rides were performed each day so we:
# - parse the date string to a datetime format
# - set the date as index
# - resample to '1D' or one day (and chose count as aggregation)

# In[3]:


taxi_df.pickup = pd.to_datetime(taxi_df.pickup)
taxi_df = taxi_df.set_index('pickup')


# In[4]:


taxi_sum_df = taxi_df.resample('D').sum()
taxi_sum_df.head()


# great! now we have an idea on how many ubers were taken each day, let us use a simple line plot to show the results.

# In[5]:


sns.lineplot(data=taxi_sum_df.passengers[1:])


# This dataset is nice, but we want to know which factors might influence the taxi habits of NYC.
# What we could do is add weather information to it.
# 
# We found a dataset online of the wether in and around central park, which should be relevant to NYC.

# In[6]:


weather_df = pd.read_csv('https://raw.githubusercontent.com/toddwschneider/nyc-taxi-data/master/data/central_park_weather.csv')
weather_df.head()


# you can see a variaty of information, more info on the column names can be found [here](https://docs.opendata.aws/noaa-ghcn-pds/readme.html)  
# again we need to:
# - parse the date
# - set it to the index and drop useless columns
# - resampling is not needed as it is already in day-to-day intervals
# 

# In[7]:


weather_df.DATE =  pd.to_datetime(weather_df.DATE)
weather_df = weather_df.set_index('DATE').drop(columns=['STATION', 'NAME'])
weather_df.head()


# Having 2 dataset, now we need to merge them. Since we already prepared the date as index, this should be easy.

# In[8]:


merged_df = pd.merge(taxi_sum_df, weather_df, left_index=True, right_index=True, how='left')
merged_df.head()


# In[9]:


merged_df.head()


# One would assume that when it is a rainy day, people would use more cabs. so let us seperate based on precipitation.

# In[10]:


rained = merged_df[merged_df['PRCP']>0]
no_rain = merged_df[merged_df['PRCP']==0]


# In[11]:


print('average passengers on a rainy day')
print(rained['passengers'].mean())
print('average passengers on a dry day')
print(no_rain['passengers'].mean())


# In[12]:


no_rain.shape


# ouch! it looks like the average new yorker doesn't mind getting wet, or they take a cab any day...  
# using a regression plot we can see it more clear

# In[13]:


sns.regplot(data=merged_df, x='PRCP', y='passengers')


# Ok, here we see that it might just be because a lot of days are dry and the dataset is skewed. Not reliable info.  
# What about temperatures, can we see a difference if the lowest temperature changes?

# In[14]:


sns.regplot(data=merged_df, x='SNWD', y='passengers')


# Appearantly when the temperature lowers, yorkers seem to be taking more cab rides. So global warming might be disastrous for capitalism after all?
