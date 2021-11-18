#!/usr/bin/env python
# coding: utf-8

# ## Using SQL
# 
# In this notebook we are going to do things different, instead of using python and pandas for data wrangling/processing we outsource them to the SQL, a language used for databases.
# 
# As it would be complicated to setup a complete SQL server, I opted to create a local database using SQLite which is built-in the sqlalchemy library used by python to interact with a database.
# 
# We start by importing or necessary libraries

# In[1]:


import pandas as pd
import sqlalchemy


# As mentioned we are going to create a local SQL database and dump it to a .db file. In order to do that we first have to read our data from comma seperated value (CSV) files that were provided within the repository.
# 
# We use pandas to read them and collect them into an object data

# In[2]:


data = {
    'ratings': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/rating_final.csv'),
    'cuisine': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/chefmozcuisine.csv'),
    'parking': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/chefmozparking.csv'),
    'user_cuisine': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/usercuisine.csv'),
    'user_payment': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/userpayment.csv'),
    'user_profile': pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c3_data_preprocessing/data/cuisine/userprofile.csv', na_values='?'),
}


# Before we can act with our database we need to create an engine by setting up the connection. In a more complex situation this would need an url to the server running the database, a userid and password for loging and some other configurations.
# 
# In this case we only need the location of our .db file, which i will put in the same location as the notebook.

# In[3]:


engine = sqlalchemy.create_engine('sqlite:///ratings.db')


# Great! we now have an engine that can run our SQL queries, yet for now our database is empty, let us fill it with all the data we collected earlier!
# 
# We use the .to_sql method of pandas to easily convert the pandas dataframe to a table in our database, each name in our data object will be a table with the corresponding data.

# In[4]:


for table_name, df in data.items():
    df.to_sql(table_name, engine, if_exists='replace')


# And with this our migration to SQL has been completed, we now have a SQL server running locally that has several tables containing data.
# Instead of using python to do the processing we can instruct our server to handle this, usually resulting is faster compute times, yet results may vary!
# 
# Let's start with a simple example, I saw that we have a table with ratings, to see how it looks by selecting all columns.

# In[5]:


df = pd.read_sql(
    """
    SELECT * FROM ratings
    """,
    engine
)
df


# It looks that an index has been copied too, we skipped preparation and it already shows. For now we are going to ignore these steps yet we should clean that later.
# If you want to save some time, you can LIMIT your search to a number of rows, next I put a limit of 5 to only retrieve the first 5 results

# In[6]:


df = pd.read_sql(
    """
    SELECT * FROM ratings
    LIMIT 5
    """,
    engine
)
df


# Great! Here it does not matter as our database is local and not at all large in size, but this trick might save you a lot of time when exploring.
# 
# Next we would like to only select specific columns, by changing the asterisk to the wanted columns the server knowns which columns to retrieve.

# In[7]:


df = pd.read_sql(
    """
    SELECT userID, rating FROM ratings
    """,
    engine
)
df.head()


# Aside from less traffic, this tidies up your data as usually most columns are not needed.
# 
# Just like columns, entries can also be filtered, in the next example we use an equation to filter only the ratings with a general rating of 2.

# In[8]:


df = pd.read_sql(
    """
    SELECT userID, rating FROM ratings
    WHERE ratings.rating = 2
    """,
    engine
)
df.head()


# Similarly you can also filter based on text fields, for this example I retrieve data from another table, cuisine.
# No particular columns are selected yet we want to only retrieve the entries where the column Rcuisine contains a text ending on 'food' the percent sign is a wildcard indicating that any text can be present here.

# In[9]:


df = pd.read_sql(
    """
    SELECT * FROM cuisine
    WHERE Rcuisine LIKE '%food'
    """,
    engine
)
df.head()


# It looks that the server has found 2 types of entries that satisfy my filter, both 'Fast_Food' and 'Seafood' were results as they both end in 'food', the percent sign in this case filled for 'Fast_' and 'Sea'.
# 
# A third method of filtering entries can be a range of numbers, using the BETWEEN and AND statements.

# In[10]:


df = pd.read_sql(
    """
    SELECT userID, placeID, rating FROM ratings
    WHERE placeID BETWEEN 132000 AND 135000
    """,
    engine
)
df.head()


# Another method would be to use the IN statement and supply a list/tuple of possible entries, in the example we filter on 2 users that placed ratings.

# In[11]:


df = pd.read_sql(
    """
    SELECT userID, placeID, rating FROM ratings
    WHERE userID IN ('U1077', 'U1103')
    """,
    engine
)
df.head()


# It is also possible to filter on NULL values (NaN or missing values in SQL), this way we can easily see we again forgot to do our data preparation.

# In[12]:


df = pd.read_sql(
    """
    SELECT * FROM user_profile
    WHERE smoker is NULL
    """,
    engine
)
df


# We can quickly fix this by just removing all users that have missing values for smoker, as there are only 3.
# The syntax is a bit different as we are not using pandas, but the idea is the same, we just dont parse the result into pandas.

# In[13]:


conn = engine.connect()
conn.execute(
    """
    DELETE FROM user_profile
    WHERE smoker is NULL
    """
)


# Before we check if they are removed, think about the impact of removing users, do you think we can just do this without consequences? what about the ratings they gave? Perhaps you could remove them too here? Is it still possible?

# In[ ]:





# We do a quick check to see if the users with missing values are gone.

# In[14]:


df = pd.read_sql(
    """
    SELECT * FROM user_profile
    WHERE smoker is NULL
    """,
    engine
)
df


# Thus far we used 2 tables, ratings and cuisine, yet always seperate.
# Here we combine the information of both by joining them on a common column; the placeID.
# 
# Using the JOIN keyword together with the ON keyword we here perform an inner join.

# In[15]:


df = pd.read_sql(
    """
    SELECT ratings.placeID, cuisine.Rcuisine, ratings.rating
    FROM ratings JOIN cuisine
    ON ratings.placeID == cuisine.placeID
    """,
    engine
)
df.head()


# Now we can see per rating, not only which placeID is related but also the cuisine of that place.
# This way we can create new views on our data without having overly complicated structures with redundant data.
# 
# Next to joining we can also aggregate data, here I created a query that counts the ratings in the ratings table, giving us the total amount of ratings.

# In[16]:


count_df = pd.read_sql(
    """
    SELECT COUNT(rating) FROM ratings
    """,
    engine
)
count_df


# The strengh of aggregation becomes useful when using the GROUP BY keyword, where we can group our data based on columns.
# The next query calculates the average rating from the rating table grouped on the placeID, note when using grouping all other selected columns need to have an aggregation function in order to work.

# In[17]:


avg_df = pd.read_sql(
    """
    SELECT placeID, AVG(rating) FROM ratings
    GROUP BY placeID
    """,
    engine
)
avg_df.head()


# We can go further and combine joining and grouping, with this we can join the cuisine type from the cuisine table and group on that column, we then take both average and count of ratings.

# In[18]:


cuisine_df = pd.read_sql(
    """
    SELECT cuisine.Rcuisine, AVG(ratings.rating), COUNT(ratings.rating)
    FROM ratings JOIN cuisine
    ON ratings.placeID == cuisine.placeID
    GROUP BY cuisine.Rcuisine
    """,
    engine
)
cuisine_df


# For an American cuisine we have an average rating of 1.15 and a count of 39 ratings. 
# Keeping track of the count makes sure we known how many ratings are behind the average score.
# 
# Let's say we want to know the type with the highest average rating, we could use the ORDER BY keyword to order our results.

# In[19]:


cuisine_df = pd.read_sql(
    """
    SELECT cuisine.Rcuisine, AVG(ratings.rating), COUNT(ratings.rating)
    FROM ratings JOIN cuisine
    ON ratings.placeID == cuisine.placeID
    GROUP BY cuisine.Rcuisine
    ORDER BY AVG(ratings.rating) DESC
    """,
    engine
)
cuisine_df


# So, mediterranean cuisine has the highest rating, yet only 4 ratings are present, not a representable amount.
# What we could do is create a query that filters all the places with 5 or more ratings, we can use the HAVING keyword to filter groups whilst performing a GROUP BY operation.

# In[20]:


place_df = pd.read_sql(
    """
    SELECT placeID, COUNT(rating)
    FROM ratings 
    GROUP BY ratings.placeID
    HAVING COUNT(rating) > 4
    """,
    engine
)
place_df.head()


# With this query qe only keep the places with 5 or more ratings, as we chosen 5 as an arbitrary value of statistical significance here.
# 
# As a last query I would like to combine the last 2, where we use the filter as a subquery in our query to find the average of each cuisine type.
# This means that we take the average of each cuisine type, but only take into account places with 5 or more reviews.

# In[21]:


cuisine_df = pd.read_sql(
    """
    SELECT cuisine.Rcuisine, AVG(ratings.rating), COUNT(ratings.rating)
    FROM ratings JOIN cuisine
    ON ratings.placeID == cuisine.placeID
    WHERE ratings.placeID in (
        SELECT placeID
        FROM ratings 
        GROUP BY ratings.placeID
        HAVING COUNT(rating) > 4
    )
    GROUP BY cuisine.Rcuisine
    ORDER BY AVG(ratings.rating) DESC
    """,
    engine
)
cuisine_df


# You can see that Mediterranean now is missing as it only had 4 ratings, yet the Family cuisine still has 10 out of 12 reviews and it's average even increased. 
# 
# Although we used SQL which can only perform simple mathematics we were able to manipulate our dataset before even going into the data exploration phase.
# When dealing with larger datasets using SQL can drastically improve your data analytical experience and is therefore an essential tool for a data scientist
# 
# I'll leave a blank cell here for you to experiment, for more inspiration you could also check out this [cheat sheet](https://learnsql.com/blog/sql-basics-cheat-sheet/sql-basics-cheat-sheet-a4.pdf)

# In[ ]:




