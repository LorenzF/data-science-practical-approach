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
#set_matplotlib_formats('svg')
plt.rcParams['figure.figsize'] = [10, 10]


# in order to download datasets from kaggle, we need an API key to access their API, we'll make that here

# In[2]:


if not os.path.exists(os.path.expanduser('~/.kaggle')):
    os.mkdir(os.path.expanduser('~/.kaggle'))

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
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
# 
# At first we would like to know which countries are performing well, we could simply do a sum of all medals for each country as shown below

# In[21]:


medals_agg_df = medals_country_df.groupby('NOC').sum().sort_values(by='Gold', ascending=False)
medals_agg_df.head(20)


# As expected, USA leads the charts, interestingly although disbanded over 30 years ago, the soviet are still second in amount of medals, this leads me to several questions:
# - does every country have the same resources? 
# - are some sports easier to obtain medals?
# - is the type of medal important?
# 
# To create a simple answer on the last one, we could for each country calculate the percentage of gold/silver/bronze medals they obtained, meaning that not the amount but the ratio is important.

# In[22]:


medals_perc_df = medals_agg_df[medals_agg_df.sum(axis='columns')>20].apply(lambda x: x/x.sum(), axis='columns').sort_values(by='Gold', ascending=False)
medals_perc_df.head(20)


# In[23]:


medals_agg_df.loc['ETH']


# Out of nowhere Ethiopia seems to be the highest achiever when it comes to gold medals, but this might be an anomaly as their total medal count is rather low, but still impressive! Also China steps up showing that they don't take second best.
# 
# I also mentioned resources, some countries are not as big as USA an China and therefore send less athletes.
# We could have checked for the amount of athlete's yet opted to go for each countries population.
# If a country has a bigger population it means it has a bigger pool of genetically favored persons for a sport.
# 
# To investigate this I searched for a dataset containing the data, coming from the worldbank API, in the next section we download the data.

# In[24]:


from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


# In[25]:


resp = urlopen("https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv")
zipfile = ZipFile(BytesIO(resp.read()))
print(f"found files: {zipfile.namelist()}")


# In[26]:


file_name = 'API_SP.POP.TOTL_DS2_en_csv_v2_3358390.csv'
zipfile.extract(file_name, './data')
pop_df = pd.read_csv('./data/'+file_name, encoding='', skiprows=4)
pop_df.head()


# You can see that for each year from 1960 the population for each country is given, we first have to stack/unpivot the data to obtain a view that is useful for our purpose.

# In[27]:


pop_df = pop_df.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code'] + pop_df.columns[pop_df.columns.str.contains('Unnamed')].tolist()).set_index('Country Code').stack()
pop_df = pop_df.rename('population')
pop_df.head(5)


# Now we have to match this with our medals dataset we created earlier

# In[28]:


medals_country_df.head()


# There seems to be a problem, our medals dataset does not indicate the year, we can solve this by adding a column

# In[29]:


medals_country_df['year'] = medals_country_df.index.get_level_values('Games').str[:4]
medals_country_df.head()


# Great! now we can merge the population data with our medals data

# In[30]:


medals_country_pop_df = pd.merge(medals_country_df, pop_df, left_on=[medals_country_df.index.get_level_values('NOC'), 'year'], right_index=True, how='left')
medals_country_pop_df


# As our population data only contained data from 1960 onwards, we need to discard some of our rows, we do this with the dropna method

# In[31]:


medals_country_pop_df = medals_country_pop_df.dropna()
medals_country_pop_df


# In order to use our population information, we need to be creative, I decided to keep things simple and for each type of medal divide the amount with the population, therefore the value is changed from:
# 
# - the amount of medals earned for a country
# 
# to
# 
# - the amount of medals earned per person for a country
# 
# Which will be much lower for countries with a higher population

# In[32]:


medals_pop_df = medals_country_pop_df[['Gold', 'Silver', 'Bronze']].div(medals_country_pop_df.population,axis='index')
medals_pop_df


# You can see that these values are much lower as populations are very high. Now we can do exactly the same as before and sort per highest total amount.

# In[33]:


medals_pop_df.groupby('NOC').sum().sort_values(by='Gold', ascending=False).head(20)


# Our data is now completely different, for the reason that Liechtenstein is very small it scores very high.
# You could argue that being small is an advantage here, yet it also means you have less chance to have highly athletic persons. Just to make sure that they did not by accident get a gold medal let's get all of their medals.

# In[34]:


athlete_events[(athlete_events.NOC=='LIE') & ~(athlete_events.Medal.isna())]


# In my opinion this looks about right, 2 gold medals, 2 silver and 5 bronze is impressive for a country with less than 40k inhabitants.
# 
# Also a lot of scandinavian countries seem to have taken the lead, this might be indicating that there is less competition in winter sports as they are known to excel there.
# 
# Most remarkable is the fall of the USA, which falls to the 20th place, indicating that if we correct for the amount of persons in the country it does not perform that well.
# 
# In a same method we could also account for the Gross Domestic Product per Capita, indicating the wealth of a country, again we download data from worldbank

# In[35]:


resp = urlopen("https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=csv")
zipfile = ZipFile(BytesIO(resp.read()))
print(f"found files: {zipfile.namelist()}")


# In[36]:


file_name = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3358201.csv'
zipfile.extract(file_name, './data')
gdp_cap_df = pd.read_csv('./data/'+file_name, encoding='', skiprows=4)
gdp_cap_df.head()


# In[37]:


gdp_cap_df = gdp_cap_df.drop(columns=['Country Name', 'Indicator Name', 'Indicator Code'] + gdp_cap_df.columns[gdp_cap_df.columns.str.contains('Unnamed')].tolist()).set_index('Country Code').stack()
gdp_cap_df = gdp_cap_df.rename('gdp')
gdp_cap_df.head(5)


# Again data from 1960 untill recent that we can use, we merge this with our original medals data.

# In[38]:


medals_country_gdp_df = pd.merge(medals_country_df, gdp_cap_df, left_on=[medals_country_df.index.get_level_values('NOC'), 'year'], right_index=True, how='left').dropna()
medals_country_gdp_df


# And again we recompute our metric, by dividing the amount of medals by the GDP, indicating not how many medals but how many medals per dollar of weight per person obtained

# In[39]:


medals_country_gdp_df = medals_country_gdp_df[['Gold', 'Silver', 'Bronze']].div(medals_country_gdp_df.gdp,axis='index')
medals_country_gdp_df


# In order to compare we calulcate again the total medal/wealth metric for each country

# In[40]:


medals_country_gdp_df.groupby('NOC').sum().sort_values(by='Gold', ascending=False).head(20)


# As expected China performs well, but also Etheopia again scores high together with Kenia, I'm assuming a lot of runners come from this region. Remarkable is that countries such as USA and Japan, which are known to have a high GDP are still performing outstanding.
# 
# Now that we have 3 versions of the same analysis it debatable which one is 'more accurate', I personally believe that good athlete's depend more on the countries population than wealth, as talent will always emerge from a pool and GDP is not a great indicator if the country has the resources to support an athlete.

# In[ ]:





# In[ ]:





# ### Medals per group (season, sport,...)
# 
# I mentioned earlier that Scandinavian countries are good at winter sports, let's prove it, we divide our dataset in 'Summer' and 'Winter'.

# In[41]:


medals_country_df['season'] = medals_country_df.index.get_level_values('Games').str[5:]
medals_country_df.head()


# By grouping per season and country and counting the total amount of medals (here gold, silver or bronze does not matter) we get 2 values for each country. We first sum all types of medals, then group by season and country and last pivot the season feature to create columns for each season

# In[42]:


medals_season_df = medals_country_df.set_index('season', append=True)[['Gold', 'Silver', 'Bronze']].sum(axis='columns').unstack('NOC').groupby('season').sum().T
medals_season_df


# Using our contingengy table chi squared test we can easily find out if for certain rows the distribution of our 2 columns (Summer and Winter) is skewed.

# In[43]:


F, p, df, exp = scipy.stats.chi2_contingency(medals_season_df)
F, p


# with a p-value of 0.0 we know there is a definite shift for certain countries, using the expected values we calculate the diff and sort it by descending order on summer

# In[44]:


medals_season_diff_df = medals_season_df-exp
medals_season_diff_df.sort_values(by='Summer', ascending=False)


# Although having bad weather, the british do not fancy some snow at all, similar for the United States.
# In contrast, countries as Norway, Austria, Canada, Finland, Switzerland, ... really excel in winter sports!
# 
# Similarly to this analysis we can to the same for countries and types of sports, we do the same manipulation and obtain the next view of our data.

# In[45]:


medals_sport_df = medals_country_df.sum(axis='columns').unstack('Sport').groupby('NOC').sum()
medals_sport_df.index = medals_sport_df.index.astype('str')
medals_sport_df


# So instead of knowing which countries are performing different on summer and winter games, we can not figure out which sports are excelled by a nation.

# In[46]:


F, p, df, exp = scipy.stats.chi2_contingency(medals_sport_df)
F, p


# Again a p-value of 0.0 indicate the correlation is not a coincidence, so we should investigate with the differences, as we have a lot of sports and countries, it would be wise to select a single country or sport.

# In[47]:


medals_sport_diff_df = medals_sport_df-exp
medals_sport_diff_df.loc['NED'].sort_values(ascending=False).head(10)


# As you can see I took the Dutch which clearly have a favorite. Speed Skating and Hockey where 2 sports where I thought they would be scoring well, but they also perform well on cycling and swimming!
# 
# It also works the other way around if we select a sport and see which countries are good, I wanted to known which countries are good at sailing.

# In[48]:


medals_sport_diff_df['Sailing'].sort_values(ascending=False).head()


# Looks like Great Britain is good at sailing, all those years of colonialism still seem to pay of...
# 
# ### Athlete attributes
# 
# In this section we will be looking at attributes from athletes, age, height and weight are all given in the dataset, yet with a lot of missing values. To make our life easier I created 2 functions that retrieves groups of athletes based on a grouping and the mean of each groups for the grouping, also you can set if we only take athletes that received a medal.

# In[49]:


def group_athletes(grouping=['Sex'], agg=False, medals=False):
    df = athlete_events.drop_duplicates(subset=['Name', 'Age', 'NOC'])
    df = df.dropna(subset=['Age', 'Height', 'Weight'])
    if medals:
        df = df[~df.Medal.isna()]
    return [x[1][['Age', 'Height', 'Weight']] for x in df.groupby(grouping) if len(x[1])>5]


# In[50]:


def median_athletes(grouping=['Sex'], medals=False):
    df = athlete_events.dropna(subset=['Age', 'Height', 'Weight'])
    if medals:
        df = df[~df.Medal.isna()]
    return df.groupby(grouping)[['Age', 'Height', 'Weight']].median()


# To give an example, here is the result of the mean for athletes grouped per gender. I want to remark here that I did not perform a non-normal test as a fact that I always know data such as this is not normal distributed. A mean is not the perfect indicator for this!

# In[51]:


median_athletes(['Sex'])


# Now for each attribute we would like to perform an ANOVA with the initial values, we can do this with the scipy library, where we supply the data from the (in this case) 2 groups.

# In[52]:


F, p = scipy.stats.f_oneway(*group_athletes(['Sex']))
print(f'F: {F}')
print(f'p: {p}')


# You can See that the p-values are all less that 0.05 indicating no chance this happend by accident, so there is a clear difference for Age, Height and Weight for Male and Female Athletes. Which was also visible in the earlier table we created, yet we know it is not by random coincidence.
# 
# How about we only take athletes that have obtained a medal? do we see a difference then?

# In[53]:


F, p = scipy.stats.f_oneway(*group_athletes(['Sex'], medals=True))
print(f'F: {F}')
print(f'p: {p}')


# Again the results are very clear, yet we can see that the F-Values are much lower, indicating the difference is much lower, let's look at medians

# In[54]:


median_athletes(['Sex'], medals=True)


# Although no big differences most values have shifted upwards indicating being taller and heavier gives you more chance on a medal?
# 
# Instead of focussing on gender, let's look at sports, as I assume not every sports prefers the same athlete.

# In[55]:


F, p = scipy.stats.f_oneway(*group_athletes(['Sport'], medals=True))
print(f'F: {F}')
print(f'p: {p}')


# F values are much less, yet we should not compare as we changed our grouping, the p-values as usually are so low there is no chance of randomness.
# 
# As we have too many sports, I decided to sort them by Height and only show the shortest.

# In[56]:


median_sport_df = median_athletes(['Sport']).dropna().sort_values(by='Height')
median_sport_df.head()


# Clearly there are some sports that favor being small, there are probably numerous arguments why that would be, but I'm not going to go there. 
# 
# Now that we are here, let's look at the sports with the heaviest athletes.

# In[57]:


median_sport_df.sort_values(by='Weight', ascending=False).head()


# Although not a sport anymore, Tug-Of-War still has the heaviest contestants, which indicates that weight sure is a way to win an old-fashioned tug of war.
# 
# To give it some more insight, we could divide each row with it's mean, this would give a differential compared to the mean.

# In[58]:


median_sport_df.apply(lambda x: x-x.mean())


# This way you can see that the median basketball player is 15.5 cm taller than an average athlete.
# 
# Aside from grouping on 1 attribute (Gender or Sport) we can also combine them, but this makes things more complicated. Here we group on Gender and Sport type and only select medal wining athletes.

# In[59]:


sport_gender_df = median_athletes(['Sex', 'Sport'], medals=True).dropna().unstack('Sex')
sport_gender_df.head()


# The options of comparison grow exponentially with every grouping level, therefore I selected one which I thought might be interesting, we are comparing per sport the height of males and females. so a negative value means females are higher than males. 

# In[60]:


(sport_gender_df['Height']['M']-sport_gender_df['Height']['F']).rename('height_difference').sort_values(ascending=False).dropna()


# Here you can read that e.g. basketbalplayers in general have a taller height, yet difference between male and female is also 15cms so the height advantage is not that appearant in female basketball. On the other side, Boxing has a lower height difference, yet boxing already was a sport that benefits smaller athletes than average.
# 
# To end this section I would like to take a grouping where the difference is not that obvious, by grouping per medal.

# In[61]:


F, p = scipy.stats.f_oneway(*group_athletes(['Medal']))
print(f'F: {F}')
print(f'p: {p}')


# You can see that for age we have a p-value of 0.67, indicating no difference in age for athletes that have obtained different types of medals, yet for height and weight the p-value is significant.
# However if we look at the median values we see nearly no difference.

# In[62]:


median_athletes(['Medal'])


# This is a great example of how significance does not imply relevance, the differences here are so small they are irrelevant.

# ## Visualization
# 
# Before we start creating graphics, a little recall we started out with a view of our data for each games, NOC and sport the amount of medals

# In[63]:


medals_country_df.head()


# What I would be interested in is the evoluation of amount of medals for the highest achieving countries, therefore we need a list of the best countries, I selected the top 10 countries with most medals.

# In[64]:


most_medals = medals_country_df.groupby('NOC')[['Gold','Silver','Bronze']].sum().sum(axis='columns').sort_values(ascending=False).head(10).index.values
most_medals


# Now for those countries we create a new view on our data that contains the won medals for each of those countries.

# In[65]:


medals_country_wide_df = medals_country_df.reset_index().groupby(['year','NOC'])[['Gold', 'Silver', 'Bronze']].sum().sum(axis='columns').unstack()
medals_country_wide_df = medals_country_wide_df[most_medals].fillna(0)
medals_country_wide_df.tail()


# We can create a simple line plot for this, where the x-axis is the chronological years of each games and y is the amount of medals

# In[66]:


sns.lineplot(data=medals_country_wide_df)
plt.xticks(rotation=45)
plt.show()


# Looks like we forgot something, we are plotting the amount of medals per year and not cumulative, fortunately a builtin method can solve this

# In[67]:


sns.lineplot(data=medals_country_wide_df.cumsum())
plt.xticks(rotation=45)
plt.show()


# I did the same for the population corrected data, creating a line plot, this is in my opinion more interesting as it gives a more honest take on the competition.

# In[68]:


most_medals_pop = medals_pop_df.groupby('NOC')[['Gold','Silver','Bronze']].sum().sum(axis='columns').sort_values(ascending=False).head(10).index.values
medals_pop_df['year'] = medals_pop_df.index.get_level_values('Games').str[:4].astype('int')
medals_country_wide_pop_df = medals_pop_df.reset_index().groupby(['year','NOC'])[['Gold', 'Silver', 'Bronze']].sum().sum(axis='columns').unstack()
medals_country_wide_pop_df = medals_country_wide_pop_df[most_medals_pop].fillna(0)
sns.lineplot(data=medals_country_wide_pop_df.cumsum())
plt.xticks(rotation=45)
plt.show()


# There seems to have been a golden age for Liechtenstein, as they are taking up a lot of space I opted to remove them and plot again

# In[69]:


sns.lineplot(data=medals_country_wide_pop_df.drop(columns=['LIE']).cumsum())
plt.xticks(rotation=45)
plt.show()


# Great! a lot of other interesting countries performances, note that CHI stands for Chile which catches up phenomenally.
# 
# Another take would be a pie chart, although not my favorite it would make a good option in this situation, as we want to compare the relative portions of countries. When we use the regular data we obtain the following.

# In[70]:


medals_country_df.groupby(level='NOC').sum().sum(axis='columns').sort_values(ascending=False).plot.pie()


# Verry messy, as most countries are not visible on the pie chart, a good option would be to only take the top 20 countries and put the others in a 'other' category.

# In[71]:


medals_country_vis_df = medals_country_df.groupby(level='NOC').sum().sum(axis='columns').sort_values(ascending=False)[:19]
medals_country_vis_df['other'] = medals_country_df.groupby(level='NOC').sum().sum(axis='columns').sort_values(ascending=False)[19:].sum()
medals_country_vis_df.plot.pie()


# Much better, with this pie plot we can see that 10 countries obtained about half of all medals and the next 10 have about 25%, the other 130 countries are in the botton quarter.
# 
# Now to add more depth we can divide our dataset, something we mentioned earlier is the dominance in winter sports, here we create the same pie chart but only take events from winter games.

# In[72]:


medals_winter_df = medals_country_df[medals_country_df.season=='Winter'].groupby(level='NOC').sum().sum(axis='columns').sort_values(ascending=False)[:19]
medals_winter_df['other'] = medals_country_df[medals_country_df.season=='Winter'].groupby(level='NOC').sum().sum(axis='columns').sort_values(ascending=False)[19:].sum()
medals_winter_df.plot.pie()


# You can compare them and see that some countries fall and some rise, indicating that countries definitely have a preference.
# 
# Again we can do the same with population corrected data.

# In[73]:


medals_pop_vis_df = medals_pop_df.groupby(level='NOC').sum()[['Gold','Silver','Bronze']].sum(axis='columns').sort_values(ascending=False)[:19]
medals_pop_vis_df['other'] = medals_pop_df.groupby(level='NOC').sum()[['Gold','Silver','Bronze']].sum(axis='columns').sort_values(ascending=False)[19:].sum()
(medals_pop_vis_df*1200).plot.pie()


# Or GDP corrected data

# In[74]:


medals_gdp_vis_df = medals_country_gdp_df.groupby(level='NOC').sum()[['Gold','Silver','Bronze']].sum(axis='columns').sort_values(ascending=False)[:19]
medals_gdp_vis_df['other'] = medals_country_gdp_df.groupby(level='NOC').sum()[['Gold','Silver','Bronze']].sum(axis='columns').sort_values(ascending=False)[19:].sum()
medals_gdp_vis_df.plot.pie()


# ### best performing per sport
# 
# To visualise the best performing country per sport we first need the country that won the most medals per sport. we do this with the following code

# In[75]:


best_country_sport_df = pd.concat(
    [
        medals_country_df.groupby(level=['NOC', 'Sport']).sum().sum(axis='columns').groupby(level='Sport').apply(lambda x: x.idxmax()[0]),
        medals_country_df.groupby(level=['NOC', 'Sport']).sum().sum(axis='columns').groupby(level='Sport').apply(lambda x: x.max())
    ], axis='columns', keys=['country', 'medals']
)

best_country_sport_df.head()


# As there are to many sports, I opted to only visualise the top 20 most popular sports, by the amount of medals

# In[76]:


total_medals_sport = medals_country_df.groupby(level='Sport').sum().sum(axis='columns').rename('medals').sort_values(ascending=False).reset_index().head(20)
popular_sports = list(total_medals_sport.Sport)
best_country_sport_df.loc[popular_sports].medals


# Now we can create a bar plot, where the portion of each best performing country is shown together with the region name.

# In[77]:


sns.barplot(x=total_medals_sport.Sport.astype('str'), y=total_medals_sport.medals, color='b')
sns.barplot(x=popular_sports, y=best_country_sport_df.loc[popular_sports].medals, color='r')

for idx, sport in enumerate(popular_sports):
    plt.text(idx, best_country_sport_df.loc[sport].medals+10, best_country_sport_df.loc[sport].country, horizontalalignment='center', size='medium', color='white', rotation=90)

plt.xticks(rotation=90)
plt.show()


# This both indicates the popularity of the sport (by amount of total medals) and the amount of medals won by the best performing country.
# 
# Another approach would be to use the difference between truth and expected values, we calculated the difference earlier.

# In[78]:


medals_sport_diff_df.head()


# By sorting on the values in this matrix, we find the combination of region and sport that are most extreme, meaning either much more medals then expected, or much less medals than expected.

# In[79]:


medals_diff_df = medals_sport_diff_df.stack().sort_values(ascending=False)
medals_diff_df.head()


# So now we know that USA has aboutn 224 more medals in Swimming than expected, we could put this in a bar chart

# In[80]:


sns.barplot(x=medals_diff_df.head(), y=medals_diff_df.head().index.values, color='b')
plt.show()


# This reveals that USA seems to be investing a lot in Swimming or Athletics sports, which are by coincidence sports that have the most medals.
# You could argue that due to the cold war show-off they have fallen prey to the cobra effect where they used the amount of medals they could get as a target instead of a measure of performance, shifting them towards sports where more medals can be obtained.
# 
# Anyway, the same analysis can be done for the worst combinations.

# In[81]:


sns.barplot(x=medals_diff_df.tail(), y=medals_diff_df.tail().index.values, color='r')
plt.show()


# This analysis can also be performed on Country level, here we see that Belgium is good at

# In[82]:


medals_sport_diff_df.loc['BEL']


# And we can put this in the same type of barchart to make it comparible with the previous chart

# In[83]:


sns.barplot(x=medals_sport_diff_df.loc['BEL'].sort_values(ascending=False).head(10), y=medals_sport_diff_df.loc['BEL'].sort_values(ascending=False).head(10).index.astype('str').values, color='b')


# ### Athlete attributes
# 
# We also investigated athlete specific attributes, to refresh our memory a printout of how the dataset looks

# In[84]:


df = athlete_events.drop_duplicates(subset=['Name', 'Age', 'NOC'])
df = df.dropna(subset=['Age', 'Height', 'Weight'])[['Sex', 'Sport', 'Medal', 'Age', 'Height', 'Weight']].reset_index(drop=True)
df.head()


# I kept features such as gender, Sport, ... as these were attributes on which the physical appearance was different, we can use these features to group our athletes and visualise the distribution with a histogram.

# In[85]:


sns.histplot(data = df, x='Age', hue='Sex', bins=20, kde=True)


# In[86]:


sns.histplot(data = df, x='Height', hue='Sex', bins=20, kde=True)


# For gender, the difference in age is not that appearent, yet the shift in height is, women are in general less tall as men.
# 
# When grouping per sport we saw significant differences.

# In[87]:


ax = sns.kdeplot(data=df, x='Age', hue='Sport')
plt.legend().remove()


# If we put all sports like this in a distribution plot, it becomes a big mess, I had to remove the legend as there are a lot of sports and the bins all overlap. It seems not a good idea to make such a plot.
# 
# For medals we only have 3 different groups.

# In[88]:


ax = sns.histplot(data=df, x='Weight', hue='Medal', bins=20, kde=True)


# Obviously for each ceremony we have 1 gold, 1 silver and 1 bronze, so distributions are equal in size.
# We saw erlier that the groups do not have significant differences and this is confirmed with the histogram, although you can see some small differences that perhaps show a pattern?
# 
# Lastly I would like to add another dimension to the plots by using scatterplots, it will be messy but creates a new perspective.
# For the scatter plot I would first plot all athlete's height and weight (you could add lines of equal BMI here) and superpose in other colors subgroups of athletes based on groups.
# Here I use the sport to show all athletes, gymnastics and weightlifting.

# In[89]:


sns.scatterplot(data=df, x='Weight', y='Height', label='all')
sns.scatterplot(data=df[df.Sport=='Gymnastics'], x='Weight', y='Height', label='Gymnastics')
sns.scatterplot(data=df[df.Sport=='Weightlifting'], x='Weight', y='Height', label='Weightlifting')


# you can clearly see how gymnastics are the smallest athletes and whilst weightlifting are also fairly small, they have a much higher weight, as they need muscles to perform their sport.

# In[90]:


sns.scatterplot(data=df[df.Medal.isna()], x='Weight', y='Height', label='all', color='blue')
sns.scatterplot(data=df[df.Medal=='Bronze'], x='Weight', y='Height', label='Bronze', color='brown', alpha=0.4)
sns.scatterplot(data=df[df.Medal=='Silver'], x='Weight', y='Height', label='Silver', color='grey', alpha=0.4)
sns.scatterplot(data=df[df.Medal=='Gold'], x='Weight', y='Height', label='Gold', color='yellow', alpha=0.4)


# Looking at this graph we can see that while there is no difference for athlete that achieves different types of medals, there is a clear area in which you should be in order to be a medal winner, outside that area clearly dimishishes your chances.
# 
# Also there seems to be an athlete that is more than 200kgs?

# In[91]:


athlete_events[athlete_events.Weight==athlete_events.Weight.max()]


# ## Summary

# 
# 
# - Best performing depends on metric
# - Some countries focus on different sports due to multiple reasons (# medals, heritage, ...)
# - Your sport and physical attributes are related, there is a ideal weight and height

# In[ ]:




