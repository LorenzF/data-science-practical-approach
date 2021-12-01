#!/usr/bin/env python
# coding: utf-8

# # Case study: Olist
# 
# In this case study we will create an overview on how a generic Data Analysis study on a dataset works.
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
# In this section we will state the goals we try to obtain by analyzing this dataset. Here are the questions that our customer had:
# 
# - Can we predict prices for products?
# - Do customers behave predictable, can we recommend specific items to specific customers?
# - Sellers with more/better reviews seem to do better, can you quantify this?
# - Are there items with a specific time pattern?
# - Are products related to geographical information?
# - Is there anything else remarkable in our data?
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
set_matplotlib_formats('svg')


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
kaggle.api.dataset_download_files(dataset='olistbr/brazilian-ecommerce', path='./data', unzip=True)
kaggle.api.dataset_download_files(dataset='olistbr/marketing-funnel-olist', path='./data', unzip=True)


# the csv files are now in the './data' folder, we can now read them using pandas, here is the list of all csv files in our folder

# In[4]:


os.listdir('./data')


# we will now parse interesting dataframes.

# In[5]:


customers = pd.read_csv('./data/olist_customers_dataset.csv')
print('shape: ' + str(customers.shape))
customers.head()


# In[6]:


sellers = pd.read_csv('./data/olist_sellers_dataset.csv')
print('shape: ' + str(sellers.shape))
sellers.head()


# In[7]:


products = pd.read_csv('./data/olist_products_dataset.csv')
print('shape: ' + str(products.shape))
products.head()


# In[8]:


translation = pd.read_csv('./data/product_category_name_translation.csv')
print('shape: ' + str(translation.shape))
translation.head()


# In[9]:


orders = pd.read_csv('./data/olist_order_items_dataset.csv')
print('shape: ' + str(orders.shape))
orders.head()


# In[10]:


order_reviews = pd.read_csv('./data/olist_order_reviews_dataset.csv')
print('shape: ' + str(order_reviews.shape))
order_reviews.head()


# ## Preparation
# 
# here we perform tasks to prepare the data in a more pleasing format.

# ### Data Types
# 
# Before we do anything with our data, it is good to see if our data types are in order

# In[11]:


customers.info()


# In[12]:


customers['customer_city'] = customers['customer_city'].astype('category')
customers['customer_state'] = customers['customer_state'].astype('category')
customers.info()


# In[13]:


sellers.info()


# In[14]:


sellers['seller_city'] = sellers['seller_city'].astype('category')
sellers['seller_state'] = sellers['seller_state'].astype('category')
sellers.info()


# In[15]:


products.info()


# In[16]:


products['product_category_name'] = products['product_category_name'].astype('category')
products.info()


# In[17]:


orders.info()


# In[18]:


orders['shipping_limit_date']= pd.to_datetime(orders['shipping_limit_date'])
orders.info()


# ### Missing values
# 
# for each dataframe we apply a few checks in order to see the quality of data

# In[19]:


print('customer missing values: ')
print(customers.isna().any())


# In[20]:


print('sellers missing values: ')
print(sellers.isna().any())


# In[21]:


print('products missing values: ')
print(products.isna().any())


# we can see that there are missing values for products, let's see how many!

# In[22]:


products.isna().sum()


# as there are not 'that many' products with missing information, I opted to drop them out. But maybe later i'll come back to that decision if these products seem crucial.

# In[23]:


products = products.dropna()


# In[24]:


print('orders missing values: ')
print(orders.isna().any())


# ### Duplicates

# In[25]:


print('customer duplicates: ')
print(customers.duplicated().any())


# In[26]:


print('seller duplicates: ')
print(sellers.duplicated().any())


# In[27]:


print('products duplicates: ')
print(products.duplicated().any())


# In[28]:


print('orders duplicates: ')
print(orders.duplicated().any())


# No duplicates, that's a good sign, it means that each customer, seller and product is unique!

# ### Indexing
# 
# It is more convenient to work with an index, usually we can use ids as index

# In[29]:


customers = customers.set_index('customer_id')
customers.head()


# In[30]:


sellers = sellers.set_index('seller_id')
sellers.head()


# In[31]:


products = products.set_index('product_id')
products.head()


# In[32]:


orders = orders.set_index('order_id')
orders.head()


# ### Translation
# for the products we have a specific dataset that contains the translations, we can apply that to the products dataframe

# In[33]:


translation_dict = translation.set_index('product_category_name')['product_category_name_english'].to_dict()
products['product_category_name'] = products['product_category_name'].cat.rename_categories(translation_dict)
products.head()


# ## Processing

# ### Product pricing
# if we want to find out if there is a correlation between pricing and products, we need to match each product with a price, let's see what happens when we merge orders and products

# In[34]:


orders.head()


# it seems that we only have prices of complete orders, which makes things more complicated. Below you can see that some orders contain multiple unique products, therefore we cannot easily deduce the price of a single item...

# In[35]:


orders.groupby(level=0).apply(lambda x: x.product_id.nunique()).value_counts()


# well, let us see if we can find all orders with one item, these prices should agree with the price of the product

# In[36]:


multi_item_orders = orders[orders['order_item_id']!=1].index.unique().values
single_item_orders = orders.drop(index=multi_item_orders)


# In[37]:


products_w_price = products.merge(single_item_orders[['product_id', 'price', 'freight_value']], how='left', left_index=True, right_on='product_id').drop(columns='product_id')


# In[38]:


products_w_price


# ### grouped per category
# It would be interesting to have the averages of each feature grouped per category.

# In[39]:


avg_category_product = products_w_price.groupby('product_category_name').mean()
avg_category_product


# ### seller reviews
# Another thing that says a lot about sales is the seller rating, we combine orders with order reviews for this

# In[40]:


seller_review_df = pd.merge(
    orders,
    order_reviews,
    left_index=True,
    right_on='order_id'
).merge(
    sellers, 
    left_on='seller_id', 
    right_index=True
)
seller_review_df.head()


# We can do a lot of things with this, an option is to get the average review per seller

# In[41]:


seller_review_df.groupby('seller_id')['review_score'].mean().sort_values()


# or the average review per seller state

# In[42]:


seller_review_df.groupby('seller_state')['review_score'].mean().sort_values()


# ## Exploration

# ### Product pricing
# 
# for the product pricing we created a dataframe that contained the single item price for most products, lets review the dataframe

# In[43]:


products_w_price.info()
products_w_price.head()


# In[44]:


products_w_price.describe()


# #### normal distribution
# 
# When we would want to predict the price of an item, it means the the other information of that item should correlate with said price. we can do that for all numerical values with a correlation plot. Before we do that let us use shapiro wilk to test normality 

# In[45]:


for name, col in products_w_price.loc[:,(products_w_price.dtypes == float).values].iteritems():
    print(name)
    print(scipy.stats.shapiro(col.dropna()))


# #### Numerical correlation
# 
# hmm it seems that we are dealing with very non normal data, which is usually the case if human behaviour is involved. We should be careful when using linear or parametric methods, so instead of calculating the pearson correlation coefficients, I opt to go for spearman rank correlations

# In[46]:


pricing_corr = products_w_price.loc[:,(products_w_price.dtypes == float).values].corr(method='spearman')
pricing_corr


# #### Variance inflation
# 
# it looks like there seem to be some interesting correlations, the price is (slightly) correlated with things as product description, weight, length, height, width and freight value, indicating that bigger items are priced higher.
# We have to take into account that freight value is on itself correlating with the latter and therefore might be inflating our results, lets use VIF to check this

# In[47]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[48]:


cols_to_keep = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'freight_value']
vif_compatible_price = products_w_price[cols_to_keep]
vif_compatible_price = vif_compatible_price.dropna(axis='index')
vif_compatible_price = vif_compatible_price.values
vif_price = {}
for idx, col in enumerate(cols_to_keep):
    vif_price[col] = variance_inflation_factor(vif_compatible_price, idx)
    print(col + ": \t" + str(variance_inflation_factor(vif_compatible_price, idx)))


# As mentioned earlier, the values here are hard to interpret, however the values seem to be lower than my experience expected. If infinite values arise we know that we need to do things different. Let's assume the collinearity between these columns is ok and they don't interfere with eachother enough to make a difference in the outcome.

# #### Categorical correlation
# 
# Something interesting we haven't looked into yet is the product category, we could try an ANOVA, but knowing at least one category is different is just a beginning.

# In[49]:


str(list(products_w_price.dtypes[(products_w_price.dtypes == float)].index))


# In[50]:


products_w_price_p_category = [products_w_price.loc[products_w_price['product_category_name']==category,(products_w_price.dtypes == float).values].dropna() for category in products_w_price['product_category_name'].unique()]
result = scipy.stats.f_oneway(*products_w_price_p_category)

anova_price = {}
for name, test, p in zip(list(products_w_price.dtypes[(products_w_price.dtypes == float)].index), result[0], result[1]):
    anova_price[name] = [test, p]

anova_price = pd.DataFrame.from_dict(anova_price, columns=['test', 'p'], orient='index')
anova_price


# it seems that every continuous column has at least one category that differs from the rest, aside from order item id, which is always 1.

# #### Grouping by category
# 
# Now comes the tricky part, we would like to know if specific categories perform better on the correlations, but this is impossible to do by hand! However python gives us the opportunity to automate this. To do this properly we have to set a rule:
# - correlations should be better than the original one without separation of categories
# 
# Look closely how we do almost exactly the same, however we aggregate (groupby) based on the category name

# In[51]:


pricing_corr


# In[52]:


pricing_rel_corr = products_w_price.groupby('product_category_name').apply(
    lambda x: x.loc[:,(x.dtypes == float).values].corr(method='spearman') - pricing_corr
    )
pricing_rel_corr


# for those who are already proficient with python can read that I opted to take the absolute correlation (meaning negatives become positives), this way both negative and positive correlations mean the same thing. Then I subtracted with the overall absolute correlation and divided that whole with the overall correlation giving me a relative change. When this relative change is positive, that category has an increased correlation

# In[53]:


pricing_corr_stacked = pricing_rel_corr.stack()
pricing_corr_stacked.sort_values(ascending=False)


# wow! we seem to be having very strong correlation increases up to 99%!? Is this possible? We should be very suspicious about these results, lets us find out why there are this high increases by calculating the initial correlation of 'security_and_services'

# In[54]:


pricing_p_cat_corr = products_w_price.groupby('product_category_name').apply(
    lambda x: x.loc[:,(x.dtypes == float).values].corr(method='spearman')
    )


# In[55]:


pricing_p_cat_corr.loc[('security_and_services','price')]


# In[56]:


pricing_corr.loc['price']


# This is not normal, a perfect correlation might indicate a category with only one record, let us print the subset of data belonging to this category

# In[57]:


products_w_price[products_w_price['product_category_name']=='security_and_services']


# #### Dealing with small subsets in data
# 
# as expected, we only have 2 item here making things a lot more complicated. We can solve this by making a compromise, since predicting prices for categories (of there is a difference in categories) with little to no examples is inaccurate, we can choose to drop all small categories. This means that our prediction is not capable for certain items however.

# In[58]:


category_sizes = products_w_price.groupby('product_category_name').size().sort_values()
small_categories = list(category_sizes[category_sizes<50].index.values)
small_categories


# We opted for a minimum of 50 items per category, let's see how that improves our relative correlations:

# In[59]:


pricing_corr_stacked.drop(index=small_categories).sort_values(ascending=False)


# Now we filtered out smaller categories that might have high fluctuations, however we are not interested into correlations between any 2 columns (keep your goals in mind!) so we are going to filter only the price. I even found a method (xs) which I never use myself, google is your friend!

# In[60]:


pricing_corr_stacked.drop(index=small_categories).xs('price', level=1, drop_level=False).sort_values(ascending=False)


# Ok, here I personally believe we have something we can work with! We can clearly see a relative change for correlation with certain columns. One thing that still remains is to filter per category the most important change compared to the average correlation

# In[61]:


pricing_most_important = pricing_corr_stacked.drop(index=small_categories).xs('price', level=1, drop_level=True).sort_values(ascending=False).reset_index().drop_duplicates(subset=['product_category_name']).set_index('product_category_name')
pricing_most_important.columns = ['parameter', 'relative_correlation']
pricing_most_important.head(10)


# In[62]:


pricing_least_important = pricing_corr_stacked.drop(index=small_categories).xs('price', level=1, drop_level=True).sort_values(ascending=False).reset_index().drop_duplicates(subset=['product_category_name'], keep='last').set_index('product_category_name')
pricing_least_important.columns = ['parameter', 'relative_correlation']
pricing_least_important.tail(10)


# What we can distill here:
# - the quantity of photo's is important for small applicances, computers, furniture,... which is to be expected because you are willing to pay more if you are sure it looks like you want it to look
# - the weight of fasion accessories and 'industry commerce' is not as important compared to other categories, as these things are always light, expensive or not
# 
# Anyway, now it is up to you to further interpret these values, but I think this should already give a nice idea on how we can estimate prices and how this changes per category.

# ## Visualization

# ### Product pricing
# 
# Now that we done the exploration, we can back our hypothesi up with some visual representations, many plots you will make will not end up in the final product but are meant to give you a more clear view on the situation itself

# #### Normal distribution
# 
# In the exploration we talked about the non normal distribution of our dataset, let us plot the numerical columns into histograms to verify this. Fortunately, pandas has a built-in hist method that works perfect.

# In[63]:


products_w_price.hist(figsize=(16,8), layout=(2,5));


# ouch! this doesn't look normally distributed at all, we can also put it into a boxplot and compare with a bar plot

# In[64]:


ax = sns.boxplot(data=products_w_price.loc[:,(products_w_price.dtypes == float).values].stack().reset_index(), x='level_1', y=0)
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(),rotation=-20,horizontalalignment='left');


# In[65]:


ax = sns.barplot(data=products_w_price.loc[:,(products_w_price.dtypes == float).values].stack().reset_index(), x='level_1', y=0)
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(),rotation=-20,horizontalalignment='left');


# These 2 plots look alike, but in my opinion the first clearly shows that the peak consists out of outliers, hence the non normal distribution. Can you find the column responsible for this peak using the histograms?

# #### Numerical correlation
# 
# We saw there were some numerical correlations within the dataset, let us try to visualize these, the first thing that pops into my mind is the pairplot.

# In[66]:


#sns.pairplot(data=products_w_price.loc[:,(products_w_price.dtypes == float).values].dropna())


# hmm it seems that in this case the pairplot doesn't seem to be that conclusive, but we already knew that the correlations werent that appearent. Let us keep it simple and make a heatmap of the correlation statistic!

# In[67]:


sns.heatmap(products_w_price.loc[:,(products_w_price.dtypes == float).values].corr(method='spearman'), annot=True)


# ok this is basically the same as in the exploration but with colors, these colors however give us a good way to group correlations, we can see that the width, height, length and weight create a nice block, and are also correlated with the price.

# #### Variance Inflation
# 
# We looked into the inflation inbetween those correlated columns, because it might be that they are telling the same story. To illustrate this information we can use a bar chart.

# In[68]:


pd.Series(vif_price).plot.bar()


# This way we can see that both the product length and width are highly correlated with other columns in the dataset i.e. their variation is explainable by other columns in the dataset. We opted to not remove any parameters here.

# #### Categorical correlation
# 
# We performed anova tests to know if and how much variance there is between categories for each numerical column. we can use a bar plot to visualize.

# In[69]:


anova_price.plot.bar()


# This might not be my best plot I ever made, but it quantifies the amount of variation of a numerical column compared to the category column 'category' item specs as size vary more whilst the name and description remain much more the same. In this plot, you can see I made a crucial mistake by using the same axis range for the test statistic and the p-value, which is much smaller (between 0-1). Don't do this yourself! (I didn't bother as all p-values are 0 except for order_item_id)

# #### Grouping by category
# 
# As last we grouped by category and recalculated the numerical correlation for each category apart. Note that we removed lowly populated categories as the prediction of the price might be not representative. I will use a boxplot to show any variation

# In[70]:


products_w_price_sorted_price = products_w_price.groupby('product_category_name').median().sort_values('price').index
products_w_price_sorted_price


# In[71]:


ax = sns.boxplot(data=products_w_price, x='product_category_name', y='price', order=products_w_price_sorted_price)
ax.set(yscale="log")
ax.set_xticklabels(ax.get_xticklabels(),rotation=-20,horizontalalignment='left');


# cool! here we can see the variation in groups for the price column, this way we can deduce wich categories are highly priced and which are lowly priced. Our machine learning solution later will use this information to help decide the price (if we of course use it to train the model). We can conclude that while the variation in each category can be high, there is a trend in price between categories.
# 
# We also calculated relative changes of correlation between price and other numerical columns inbetween categories. Let's see if we can visualize that information, my best guess would be a bar chart

# In[72]:


pricing_most_important.head()


# In[73]:


top_n = 10
ax = sns.barplot(x=pricing_most_important.head(top_n).index.to_list(), y=pricing_most_important.head(top_n)['relative_correlation'], alpha=0.7, palette='colorblind')
for idx, p in enumerate(ax.patches):
    ax.annotate(pricing_most_important.head(top_n)['parameter'][idx], 
                   (p.get_x() + p.get_width() / 2., 0), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 9), 
                rotation = 90,
                  color='white',
                   textcoords = 'offset points')
    ax.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()*0.9), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
ax.set_xticklabels(ax.get_xticklabels(),rotation=-20,horizontalalignment='left');


# You can see I put a little bit more effort in this last graph as I think this is the nice visualisation to show others. We can also make a similar plot but with the relatively least important features.

# In[74]:


top_n = 10
ax = sns.barplot(x=pricing_least_important.tail(top_n).index.to_list(), y=pricing_least_important.tail(top_n)['relative_correlation'], alpha=0.7, palette='colorblind')
for idx, p in enumerate(ax.patches):
    ax.annotate(pricing_least_important.tail(top_n)['parameter'][idx], 
                   (p.get_x() + p.get_width() / 2., 0), 
                   ha = 'center', va = 'top', 
                   xytext = (0, -9), 
                rotation = -90,
                  color='white',
                   textcoords = 'offset points')
    ax.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()*0.9), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
ax.set_xticklabels(ax.get_xticklabels(),rotation=-20,horizontalalignment='left');


# ## Summary

# ### Product pricing

# To conclude the product pricing analysis, we checked for normal distributions which werent present, so we had to opt for non-parametric/non-linear methods (although in many cases these will still do fine). We checked for numerical correlations but these were not really interesting, which led to the idea that perhaps per category our price could be predicted more accurate. This was proven by the fact that our price surely differs inbetween categories. 
# 
# We split up our dataset by grouping per category and removing small categories, now we could see that a relative change in correlation - meaning that the correlation of a column in our dataset with the price was different in that category compared to the overall correlation of this column with the price - was present for all categories. For each category we selected both the highest increase in correlation - meaning a 'spike' in importance - for that category and the highest decrease - meaning a 'drop' in importance - for that category.
# 
# These plots hence show the most important and least important attributes for an item concerning the price e.g. if we want to increase the price of an item in the computers category, we need to make sure it has enough pictures and not try to decrease the weight value.
