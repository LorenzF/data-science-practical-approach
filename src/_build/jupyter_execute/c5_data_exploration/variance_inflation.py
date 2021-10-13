#!/usr/bin/env python
# coding: utf-8

# # VIF: Variance Inflation Factor
# in this notebook we will investigate the variance inflation which can occur in a dataset. As an example here, we will use the 'Mile Per Gallon' dataset contianing a set of cars and their fuel efficiency. Some columns in the dataset might 

# In[1]:


import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
mpg = sns.load_dataset('mpg')


# In[2]:


mpg.head()


# as you can see, we also imported a function 'variance_inflation_factor' which will help us calculate this, more information can be found on [wikipedia](https://en.wikipedia.org/wiki/Variance_inflation_factor).

# to use the function, we refer to the [documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html). The function is a bit stubborn and requires the following:
# - only numerical values (so we to drop the categories)
# - no nan values (dropping nans)
# - as a numpy array instead of a pandas dataframe

# In[3]:


cols_to_keep = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
vif_compatible_df = mpg[cols_to_keep]
vif_compatible_df = vif_compatible_df.dropna(axis='index')
vif_compatible_df = vif_compatible_df.values
vif_compatible_df


# this looks a lot different! we don't know anymore what all of that means, but the computer does, now we run it through the function.
# Notice how we have to specify a specific column, the resulting inflation factor is that for the chosen column

# In[4]:


# we pick column 0 which is 'cylinders' according to cols_to_keep
variance_inflation_factor(vif_compatible_df, 0)


# In[5]:


for idx, col in enumerate(cols_to_keep):
  print(col + ": \t" + str(variance_inflation_factor(vif_compatible_df, idx)))


# ## TODO
# The variance inflation gives a numerical value to how little variation there is between one column and the others in a dataset, you will see how the numbers will gradually go down as you remove more and more columns.  
# This way we have a quantifyable method of removing data from our dataset in case there is too much 'duplicate' information.  
# There is no real cut-off value that specifies of a column should or should not be removed, so make sure you can argument your decision.
# 
# - experiment with removing columns in the cols_to_keep list
# - What do you think would be the ideal dataset here? we would like to predict the fuel economy (mpg) of a car.

# In[ ]:




