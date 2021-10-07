#!/usr/bin/env python
# coding: utf-8

# ## Binning and ranking
# 
# When dealing with numerical data the trouble can sometimes be that numbers can have a wide variety.
# 
# Here we apply 2 methods to deal with that, binning and ranking.
# With binning we change the numerical feature into a categorical/ordinal feature.
# Ranking is used when our numerical feature contains a non normal distribution that fails to be normalized.
# 
# For this example we use a food consumption dataset, where european countries are listed and the relative percentage of each country is given that consumes the type of food, e.g. a value of 67 means that 67% of that country eats that type of food.

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[2]:


food_df = pd.read_csv('https://openmv.net/file/food-consumption.csv')
food_df


# Here you could do some data validity, where we check if all values are between 0 and 100, or we check for missing values. 
# I will leave that up to you

# In[ ]:





# ### Binning
# 
# the first thing we want to do is seperate the countries based on their coffee consumption, instead of creating arbitrary values we can perform a quantitative cut.
# This means we create a number of equally sized groups using the qcut function, we give them the labels low, medium and high.

# In[3]:


food_df['bin_coffee'] = pd.qcut(food_df['Real coffee'], q=3, labels=['low', 'medium', 'high'])
food_df


# a new column has appeared at the end of our dataframe, containing the labels of our binning, countries with low coffee consumption are put in the low category and vice versa.
# Now we can seperate the countries with low coffee consumption from the rest

# In[4]:


food_df[food_df.bin_coffee == 'low']


# You can already see the England/Ireland stereotype here, note that those are the only 2 with really low coffee consumption, the others are only in this low binning because we requested equally spaced bins in our qcut function. using the cut function would result in a different outcome.
# Perhaps you could try that out?

# In[ ]:





# I tried to think of some metric to quantify the status of coffee drinkers, since we also have the instant coffee consumption we could create a metric where we subtract the amount of instant coffe drinkers from the amount of real coffee drinkers.
# This way we can measure that difference between them, I already went ahead and made equal quantity bins for them with labels low, medium and high 'quality coffee'.

# In[5]:


food_df['bin_qual_coffee'] = pd.qcut(food_df['Real coffee'] - food_df['Instant coffee'], q=3, labels=['low', 'medium', 'high'])


# In[6]:


food_df[food_df.bin_qual_coffee=='high']


# Aha! you can see here which countries prefer the real coffee over the instant version. 
# It seems the scandinavian countries together with obviously Italy are the true Caffeine connoisseur of Europe. 
# Another intersting thing we can do now is take the mean for each product for both group high and low and take the difference for high - low. 
# We can see the result below

# In[7]:


food_df[food_df.bin_qual_coffee=='high'].mean()-food_df[food_df.bin_qual_coffee=='low'].mean()


# It seems a preference for quality coffee also pairs with crisp bread, who knew?
# Do you think scaling/normalization might be interesting here? why (not)?

# In[ ]:





# ### Ranking
# 
# In case normalization fails us and we are for some reason not able to get a normal distribution out of a feature, we can still resort to ranking.
# Note that non linear machine learning techniques often use a ranking functionality under the hood, therefore this technique is often not required, yet for educational purposes we are going to use it here anyway.
# Let's see how the distribution for Real coffee consumption looks like.

# In[8]:


food_df.sort_values('Real coffee')


# Ah yes, about half of the values are 90 or higher, not really optimal as the range is between 0 and 100!
# We can also view this in a visual way using a density plot.

# In[9]:


food_df['Real coffee'].plot(kind='density', title='Real coffee (raw)')


# For larger datasets this would be more useful as we cannot see our whole dataset, it is clear we have to do something about this, now imagine we can not use regular normalization techniques.
# The rank method now comes in handy!

# In[10]:


food_df['rank_coffee'] = food_df['Real coffee'].rank()
food_df


# At the end of our data a new column was appended, containing the ranking of each country with the lowest being 1 and the highest equal to the amount of countries.
# When we visualise this distribution we get a uniform distribution, not normal but still better than before!

# In[11]:


food_df['rank_coffee'].plot(kind='density', title='Real coffee (ranked)')


# In[ ]:




