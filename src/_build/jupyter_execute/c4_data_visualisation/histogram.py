#!/usr/bin/env python
# coding: utf-8

# ## Histogram plot
# 
# When visualising one dimensional data without relating it to other information an option would be histograms.
# Histograms are used when describing distributions in your data, it is not the values itself you are visualising, rather the counts/frequencies of each value.
# 
# We again start with importing our libraries

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(16,8)})


# For this example we will be using the prepared dataset from seaborn containing mileages of several cars.
# Information about the cars is also given.

# In[2]:


mpg_df = sns.load_dataset('mpg')
mpg_df.head()


# We start of simple by plotting the distribution of horsepower in our dataset.

# In[3]:


sns.histplot(data=mpg_df, x='horsepower')


# A first thing that is visible is that our feature is not normally distributed, we have a long tail to the higer end.
# 
# For histograms we can specify the amount of bins in which we seperate the counts, seaborn selects a suitable number yet we can change this.

# In[4]:


sns.histplot(data=mpg_df, x='horsepower', bins=100)


# As you can see, the previous option looks a lot better.
# Taking the right amount of bins is important.
# 
# In order to add more information to our plot, we can use categorical data to split our data into multiple histograms.
# Here we used the origin of the cars to split into 3 categories, notice how each of them has their own area, japan and europe are on the lower end whilst usa is centered in higher horsepower.

# In[5]:


sns.histplot(data=mpg_df, x='horsepower', hue='origin', bins=20, multiple='stack')


# A neat feature of seaborn is that it can join histograms and scatter plots (in the next section) together.
# 
# Here we see how the visualisations of 2 one dimensional histograms perfectly combine together into a scatter plot, where 2 dimensional data is shown (both mileage and horsepower).

# In[6]:


sns.jointplot(data=mpg_df, x='mpg', y='horsepower')


# Histograms are a really powerfull tool when it comes to validating your data, we can easily the distribution of each feature, see if they are normally distributed and visualise distributions of subgroups.
# 
# Yet for final visualisations they are often not interesting enough.

# In[ ]:




