#!/usr/bin/env python
# coding: utf-8

# ## Box plot
# 
# In the previous section we looked into visualising the distributions of 1 dimensional data.
# We used histograms for this, but there is a second more statistical option for this, the Boxplot.
# 
# To be brief, the boxplot shows a box containing the InterQuartile Data that we already talked about and also has 2 whiskers, showing the threshold for outliers.
# Actual outliers are then printed seperately, making this plot ideal for outlier detection aswel as distributions.
# 
# I personally think this option is more suited for multiple categories compared to histograms, yet your mileage may vary.

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(16,12)})


# For this section we will look into the discovery of extrasolar planets, or planets that are ourside our own solar system.
# For each planet they listed the method of discovery, orbital period, mass, distance and year of discovery.

# In[2]:


planet_df = sns.load_dataset('planets')
planet_df.head()


# Let's say we would like to show the distances of each discovery method, if we would use a bar plot, the results might be hard to interpret.

# In[3]:


ax = sns.barplot(data=planet_df, x='distance', y='method')
ax.set(xscale="log")


# Whilst bar plots can be a good idea, here they are not.
# 
# Only use bar plots when visualising singular data points who are related to zero, not aggregations of multiple data points.
# Bar plots do not work if:
# - your datapoints have no relation to zero
# - your categories are related with different intervals
# - you are dealing with groups of datapoints, not single datapoints (this case)
# 
# anyway, we could use a histogram similar to previous section, let's see how that turns out.

# In[4]:


ax = sns.histplot(data=planet_df, x='distance', hue='method', multiple='stack', log_scale=True)


# The histogram seems to be working, yet the methods with lower count are suppressed.
# A boxplot can overcome this and we can also compare medians of each method with eachother.
# 
# Take a few minutes to understand the next plot, at first it is very confusing, yet when adapted this is the most powerful visualisation of data exploration.

# In[5]:


ax = sns.boxplot(data=planet_df, x='distance', y='method')
ax.set(xscale="log")


# Can you see now why the bar plot here is a bad idea?
# Some methods have a broader distribution and relating our data to zero makes no real sense.
# With financial data this is different as budgets always start with 0.
# 
# Here we can conclude that some methods of detecting a planet requires a further or closer distance.
# You could say that if you want to discover a far extrasolar plant pick one of the last methods
# 
# An addition to the boxplot, where we focus more on distribution instead of statistics, would be the violin plot.
# Can you see why they would call it like that?

# In[6]:


sns.violinplot(data=planet_df, x='year', y='method')


# As an exercise calculate the median, Q1 and Q3 of the distance per method and see if you come to the same conclusion as the boxplot

# In[ ]:




