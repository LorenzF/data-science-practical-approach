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
from urllib.request import urlopen


# In[2]:


data_url_files = urlopen('https://raw.githubusercontent.com/toddwschneider/nyc-taxi-data/master/setup_files/raw_data_urls.txt')
data_urls = data_url_files.read().decode('utf-8').split('\n')
data_urls[0:12]


# Due to slow parsing of data we will here only parse the uber data from jan-mar 2015

# In[3]:


datasets = [pd.read_csv(url) for url in data_urls[0:3]]


# In[4]:


cab_df = pd.concat(datasets)


# In[5]:


print('shape: ' + str(cab_df.shape))
cab_df.head()


# We would like to find out how many uber rides were performed each day so we:
# - parse the date string to a datetime format
# - set the date as index
# - resample to '1D' or one day (and chose count as aggregation)

# In[6]:


cab_df['datetime'] = pd.to_datetime(cab_df['Pickup_date'], format="%Y/%m/%d %H:%M:%S")


# In[7]:


cab_df = cab_df.set_index('datetime')


# In[8]:


cab_df.head()


# In[9]:


cabs_taken = cab_df['Dispatching_base_num'].resample('1D').count().rename('cabs_taken')
cabs_taken.head()


# great! now we have an idea on how many ubers were taken each day, let us use a simple line plot to show the results.

# In[10]:


sns.lineplot(data=cabs_taken)


# This dataset is nice, but by itself pretty useless, why don't we look up some weather information to see if this influences our traffic.

# In[11]:


url = 'https://raw.githubusercontent.com/toddwschneider/nyc-taxi-data/master/data/central_park_weather.csv'
weather = pd.read_csv(url)


# In[12]:


weather.head()


# you can see a variaty of information, more info on the column names can be found [here](https://docs.opendata.aws/noaa-ghcn-pds/readme.html)  
# again we need to:
# - parse the date
# - set it to the index
# - resampling is not needed as it is already in day-to-day intervals
# 

# In[13]:


weather['DATE'] =  pd.to_datetime(weather['DATE'], format="%Y/%m/%d")
weather = weather.set_index('DATE')


# In[14]:


weather.head()


# Having 2 dataset, now we need to merge them. Since we already prepared the date as index, this should be easy.

# In[15]:


merged_df = pd.merge(cabs_taken, weather, left_index=True, right_index=True)


# In[16]:


merged_df.head()


# One would assume that when it is a rainy day, people would use more cabs. so let us seperate based on precipitation.

# In[17]:


rained = merged_df[merged_df['PRCP']>0]
no_rain = merged_df[merged_df['PRCP']==0]


# In[18]:


print('average uber rides on a rainy day')
print(rained['cabs_taken'].mean())
print('average uber rides on a dry day')
print(no_rain['cabs_taken'].mean())


# ouch! it looks like the average new yorker doesn't mind getting wet, or they take a cab any day...  
# using a regression plot we can see it more clear

# In[19]:


sns.regplot(data=merged_df, x='PRCP', y='cabs_taken')


# Ok, here we see that it might just be because a lot of days are dry and the dataset is skewed. Not reliable info.  
# What about temperatures, can we see a difference if the lowest temperature changes?

# In[20]:


sns.regplot(data=merged_df, x='TMIN', y='cabs_taken')


# Appearantly when the temperature lowers, yorkers seem to be taking more cab rides. So global warming might be disastrous for capitalism after all?
