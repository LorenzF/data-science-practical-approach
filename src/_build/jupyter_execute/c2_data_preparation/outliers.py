#!/usr/bin/env python
# coding: utf-8

# ## Outliers and validity
# 
# When preparing data we have to be cautious with the accuracy of our set.
# Outliers and invalid data points are difficult to detect but should be handled with caution.
# 
# we start out by importing our most important library.

# In[1]:


import pandas as pd


# ### Silicon wafer thickness
# 
# Our first dataset contains information about the production of silicon wafers, each wafers thickness is measure on 9 different spots. 
# More information on the dataset can be found [here](https://openmv.net/info/silicon-wafer-thickness).

# In[2]:


wafer_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/silicon-wafer-thickness.csv')
wafer_df.head()


# we would like to investigate the distribution of measurements here, as we are early in this course using visualisation techniques would be too soon.
# This does not mean we can't use simple mathematics, introducing the InterQuartile Range.
# A reason for using IQR over standard deviation is that with IQR we do not assume a normal distribution.
# The IQR calculates the range between the bottom 'quart' or 25% and the top 25%, giving us an indication of the spread of our results, we calculate this IQR for each of the 9 measurements independently.
# For more info about IQR you can visit [wikipedia](https://en.wikipedia.org/wiki/Interquartile_range).

# In[3]:


iqr = wafer_df.quantile(0.75)-wafer_df.quantile(0.25)
iqr


# you can see that the IQR spread for each measurement lays between 0.5 and 1 unit indicating that the 9 measurements of the wafer have a similar spread.
# With these IQR's we could calculate for each point relative to the spread of the measurement how far it is from the median.

# In[4]:


relative_spread_df = (wafer_df-wafer_df.median())/iqr
relative_spread_df.head()


# You can now see that some points are close to the median, whilst others are much higher, both positive as negative.
# By defining a threshold, we quantify what deviation has to be there to flag a reading as an outlier.
# The high outliers are seperated, note that only a single measurement of the 9 can trigger and render the total measurement as an outlier.
# Yet judging from the setup where we would want to find wafers with varying thickness that approach is desirable.

# In[5]:


relative_spread_df[(relative_spread_df>2).any(axis='columns')]


# seems we have a few high outliers, you can clearly see the measurements are mostly all across the board high, but in some cases (e.g. id 154) only one measurement was an outlier.
# We can do the same for the low outliers.

# In[6]:


relative_spread_df[(relative_spread_df<-2).any(axis='columns')]


# For a simple mathematical equation these result look promising, yet it can always be more sophisticated.
# Not going to deep into the subject we could perform some Machine Learning, using a unsupervised method.
# Here we use the sklearn library which contains the Isolation forest algorithm.
# More info about the algorithm [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

# In[7]:


from sklearn.ensemble import IsolationForest


# We first create the classifier and train (fit) it with the generic wafer data.
# Then for each record of the wafer data we make a prediction, if it thinks its an outlier, we keep them

# In[8]:


clf = IsolationForest(random_state=0).fit(wafer_df)
wafer_df[clf.predict(wafer_df)==-1]


# Comparing the results with our IQR approach we see a lot of similarities, here the id 154 record did not show up as we already realised this was perhaps not a strong enough outlier.
# You could enhance our IQR technique by checking the amount of measurements that are above the threshold and respond accordingly, I will leave you a little hint.

# In[9]:


(relative_spread_df>2).sum()


# ### Distillation column
# 
# As an exercise you can try the same technique to this dataset and see what you would find, good luck!
# Be mindful that you do not incorporate the date as a variable in your outlier algorithm.

# In[10]:


distil_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/distillation-tower.csv')
distil_df


# In[ ]:




