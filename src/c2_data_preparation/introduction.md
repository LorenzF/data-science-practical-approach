# Introduction

When performing data science, we often do not elaborate about the preparation that went into the dataset.
It is considered tedious and irrelevant to the story of the analysis, 
however it is often the most important part of data analysis.
Data Preparation is the metaphorical foundation of your construction, if you fail to prepare data, you prepare to fail your analysis.

> Good data beats a fancy algorithm

If you would perform an analysis and insert unprepared data, you will mostly be disappointed with the result.

### why Data Preparation?

Aside from metaphors let us make the reasoning behind this step more tangile, to explain the relevance of this step, we partitioned the answer into a few key points.
 
#### Accuracy

There is no excuse for incorrect data and accuracy is the most important attribute.
Let us assume that we have a dataset where for some reason the result are not accurate.
This would led us to an analysis where we conclude a result that contains a bias.
An example would be a dataset of sold cars, where the listed price is that of the stock car without options.
Options are not incorporated in the price and we are perhaps training an algorithm that predicts the stock price.
If you as a data scientist fail to report/correct this, your predictions are making sense, but always underestimate!

#### Consistency

They usually say something such as 'consistency is key' and with data preparation that is likewise true.
A dataset where we do not have consistent results will never converge towards a particular answer.
Note however that it might not be a problem of consistency but rather you are missing crucial information.
If we would have a dataset where local temperatures are logged, we would like to see a consistency each 24 hours.
However we do see there are day to day fluctuations, so perhaps we need to keep track of cloud and rain data to make the dataset more complete.
We could then see that the results are more consistent yet the possibility of outliers is still present.
Equally possible would be that our temperature sensor is not sensitive enough or has large fluctuations in readings, it is the task of the data scientist to figure this out.

To get a visual  about accuracy and consistency this picture might help:

![Accuracy](https://github.com/LorenzF/data-science-practical-approach/raw/main/src/c2_data_preparation/pics/accuracy.png)

#### Completeness

As hinted in the previous point, completeness is something you have to be aware of.
Having 'complete' data is crucial for you narrative to give a correct answer, as you might otherwise lose detail.
Note that you never will know if your data is complete as there might always be more data to mine.
Yet you have to make a consideration between collecting more data and the effort required.
This collecting can happen in multiple methods, as an example we use a survey where we asked several people 10 different questions, we could:

- gather new data, here our data grows 'longer' by asking the 10 question to more people.
It might be that our sample of people were only students at a campus, so our data was not complete.

- gather new feature, by asking more questions to the same people (in case we could still find them).
By doing this we get a better understanding of their opinion, again making our data more complete.

- fill missing values, by imputing the abstained questions with answers of similar records.
When someone answered they did not want to answer we could figure out what they would have answered by looking at what persons answered that reply in a similar way.
 
#### Timeliness

For some datasets we are dealing with data that is time related.
It can happen that data at specific timepoints is missing or delayed, resulting in a failure to use machine learning algorithms.
A well-organised data pipeline utilises techniques of data preparation to circumvent these outages, usually this would be to retain the last successful datapoint.
However in hindsight we could use more complex strategies to fill in these gaps or correct datetimes in our dataset,

In this example the data stream is interrupted and data preparation is there to handle these outages before we can perform analysis.

![Timeliness](https://github.com/LorenzF/data-science-practical-approach/raw/main/src/c2_data_preparation/pics/timeliness.png)

#### Believability

You could collect the most intricate dataset possible, but if the narrative that you are conducting contradicts itself, you will end up nowhere.
During the process of data analytics it is important to apply a critical mind to what your dataset is telling you.
Obviously this is not a reason to mask or mold the data so it agrees with your opinion.
Rather you should be wary when conflicts happen and act accordingly, unfortunately it is impossible to write a generic tactic for this.
As a data scientist your experience of the underlying subject should help create understanding of the topic, remember, gathering information from experts in the field is crucial here!

#### Interpretability

Another problem that might arise when you are diving deep into the data might be that you have created something no human could ever interpret.
The Machine Learning algorithms outputs plausible and believable results, but it is impossible to understand the reasoning behind.
For some this is perfectly acceptible, for some this is undesirable.
It is your task as a data scientist to cater the wishes of the product operator and if they desire understanding as they would like to learn from the data driven process you need to unfold the process.
Usually this comes down to which data transformations are used as some do produce an output that only makes mathematical sense.

#### In conclusion

There are multiple ways to deteriorate the quality of your data and raw formats of data often contain multiple.
Before we can do anything with it these problems need to be resolved, if you fail to do so, the final output fails too.

<!--### Tasks of Data Preparation

-->


### Further reading



[Towards Data Science](https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4)




