#!/usr/bin/env python
# coding: utf-8

# ## Line plot
# 
# The most straight-forward yet very useful plotting graph is the line plot.
# With the line plot we achieve the visualisation of a single feature organized in a usually time based reference.
# 
# The line plot is ideal if you want to achieve a time critical pattern residing within your data.
# In this example we use the prepared taxi dataframe that comes with our plotting library seaborn.
# 
# From all possible plotting libraries in Python we opted for the seaborn as it has an optimal combination of simplicity and beaty, yet other libraries are equally powerful.
# 
# We begin by importing our neccesary libraries

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_theme()


# For aestetic reasons we change the figure size to something a bit larger

# In[2]:


sns.set(rc={'figure.figsize':(16,12)})


# We load our dataset, this dataset contains the trip of taxi's in regions of New York City with timestamps of pickup and dropoff.

# In[3]:


taxi_df = sns.load_dataset('taxis')
taxi_df.head()


# As we saw earlier, it is important to prepare the data, due to storage specification they did not parse the dates into a datetime format, which we do here.

# In[4]:


taxi_df.pickup = pd.to_datetime(taxi_df.pickup)
taxi_df.dropoff = pd.to_datetime(taxi_df.dropoff)


# Before we can do anything with this dataset, we need to format it into a proper format, for our first graph I would like to view the total amount of passengers per day.
# This means we have to take our data and resample on the pickup date, taking the sum.

# In[5]:


pass_df = taxi_df.set_index('pickup').resample('D').sum()
pass_df.head()


# You can almost see the plot here, we have an index of dates and a feature 'passengers', these two will make the backbone of our visualisation.

# In[6]:


sns.lineplot(x=pass_df.index, y=pass_df.passengers)


# Looks about right, however I don't like the start of it, the data started late on that first day, resampling shows we only have 1 passenger for that day.
# This is not representable, so we remove that record.

# In[7]:


pass_df = pass_df.loc['2019-03-01':]
ax = sns.lineplot(x=pass_df.index, y=pass_df.passengers)


# Much better, however the plot feels like there is a lot of fluctuations, so it would be practical to apply a rolling sum or mean.
# This rolling operation takes the last x values and applies an operation (sum, mean,...) to it, creating a smoother graph and is visually more sensitive to trends.

# In[8]:


rolling_pass_df = pass_df.rolling(7).mean()
ax = sns.lineplot(x=rolling_pass_df.index, y=rolling_pass_df.passengers)


# By applying a rolling mean, we can see that the average amount of passengers per day is decreasing.
# I feel there is no need to panick, as this is only 1 month of data and seasonal fluxtuations do happen.
# 
# Something else that triggers my curiosity is the amount these passengers paid, can we perhaps see a trend there?
# It would be ideal to plot these together so the comparison is simple.

# In[9]:


ax = sns.lineplot(x=rolling_pass_df.index, y=rolling_pass_df.passengers)
ax = sns.lineplot(x=rolling_pass_df.index, y=rolling_pass_df.fare, ax=ax)


# As we only have a few passengers per trip, yet trips can be costly the ranges of these 2 features are completely different.
# Before we think about scaling, we actually do want to know the scale here, we just cant fit them in the same graph.
# 
# A first approach would be to use a secondary axis, where the right side of the y-axis is used to show the fare scale.
# You can see that the graph is already getting more complicated code-wise, this is where using the right library is key as they usually have built in features for that.

# In[10]:


ax = sns.lineplot(x=rolling_pass_df.index, y=rolling_pass_df.passengers, label='passengers', legend=False)
ax2 = ax.twinx()
ax = sns.lineplot(x=rolling_pass_df.index, y=rolling_pass_df.fare, ax=ax2, color='r', label='fare', legend=False)
ax.figure.legend()


# Interesting! It shows that there was a period where they did not follow eachother perfect, yet the trend is almost exact for these features.
# 
# Another method where you can compare them would require feature engineering, where we calculate the fare per passenger per day, apply the rolling window and plot.
# Perhaps you could figure that out? create a new feature that divides the fare by the passengers, recreate the rolling dataframe and use seaborn to plot the results.

# In[ ]:





# At the start we used the sum of passengers per day, however we could also visualise the average amount of passengers per ride.
# The reason why I would like to do this is because earlier I saw a difference in trend for the fare and the amount of passengers, an explanation for this could be that the average amount of passengers dropped, resulting in lower passengers, yet the total expenditure of fares would remain constant.
# 
# Let us figure this out, we here calculate the average (mean) of the passengers per day.

# In[11]:


avg_pass_df = taxi_df.set_index('pickup').resample('D').mean()
avg_pass_df.head()


# Doing more or less exactly the same we can create a simple plot with the average amount of passengers in a taxi.

# In[12]:


avg_pass_df = avg_pass_df[1:]
ax = sns.lineplot(x=avg_pass_df.index, y=avg_pass_df.passengers)


# For the same reasons, this plot is not suitable as it has too much variance.
# We apply a rolling mean of 7 days and re-evaluate.

# In[13]:


rolling_avg_pass_df = avg_pass_df.rolling(7).mean()
ax = sns.lineplot(x=rolling_avg_pass_df.index, y=rolling_avg_pass_df.passengers)


# We find a dip in passengers per ride that looks to be in the same time interval, therefore we could conclude here that fares did not get more expensive, rather the sharing of cabs was less.
# You could try and find a method to add the data of these two graphs together, yet this is already advanced visualisation.
# 
# Another question that I have for you, do you think that the dip is relevant? Not specifically from a business point of view, rather from a statistical view, Perhaps if you look at the range of the y-axis you might feel that our plot is a bit magnified. This is a good example of how you can use ranges of your axi to make data more dramatic. Be weary of these malpractices!

# In[ ]:





# We are not done yet, as our dataset contains much more information.
# Harnessing the powers of the preprocessing we learned, we could include other (mostly categorical) feature into our line plot.
# 
# Here we take the payment option (either cash or card) and use it to create 2 time series in long format (2 datasets below each other).

# In[14]:


pass_payment_df = taxi_df.groupby('payment').apply(
    lambda x: x.set_index('pickup').resample('D').sum()
)
pass_payment_df


# Seaborn does not like this long format type, therefore we unstack the first index and create a wide format.
# For those wo are punctilious, you can notice we created a missing value, with wat should we fill it? (Our luck that seaborn can handle missing values!)

# In[15]:


pass_payment_df.unstack(0).head()


# Same data, different structure, now seaborn understands the format and we can go back to visualisation.
# 
# For simplicity we start with a simple passengers line plot

# In[16]:


ax = sns.lineplot(data=pass_payment_df.passengers.unstack(0)[1:])


# You can see that there are generally more people paying by card, which is more convenient in such an occasion.
# Note that here we should not use a seperate y-axis as we are comparing 2 sets of data that are similar by origin.
# 
# We do the same for fares.

# In[17]:


ax = sns.lineplot(data=pass_payment_df.fare.unstack(0)[1:])


# This is more or less a no-brainer, as more people pay by card, the fares by card are also more.
# So we can't really compare fares with this plot, we have to be creative.
# 
# I opted to go for an average fare per passenger, as this is in my opinion more relevant than the amount of rides

# In[18]:


pass_payment_df['fare_pass'] = pass_payment_df.fare/pass_payment_df.passengers
pass_payment_df.head()


# We created a new feature both containing info of fares and passengers, using this we create a new visualisations.
# 
# In this visualisation we show for both payment options the average fare amount per passenger in the cab.

# In[19]:


ax = sns.lineplot(data=pass_payment_df.fare_pass.unstack(0).rolling(7, min_periods=3).mean())


# We can conclude that the average amount that has to be paid per person is lower for cash, indicating that people jump to their debit card as soon as the amount gets too high.
# 
# As a last I would like to emphasise that the x-axis, being time does not have to be linear.
# To illustrate this we create a weekly passenger rate and impose each week over the others.

# In[20]:


pass_df.groupby(pd.Grouper(freq='W')).apply(
    lambda x: sns.lineplot(x=x.index.day_name(), y=x.passengers)
)


# Here we can see there is a weekly trend occuring, where Sundays and Mondays are usually less busy days.
# The origin of this is hard to argue, as it might be less traffic, less taxi drivers working,...
# 
# Perhaps you could complete this visualisation by investigating the distance and/or tips?

# In[ ]:




