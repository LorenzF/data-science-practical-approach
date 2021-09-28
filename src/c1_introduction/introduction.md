# Introduction

this is an introduction


### Structured vs Unstructured

When performing data preparation an important aspect is to consider with the type of data we are working with.
In general there are 2 types of data, but you could consider a third.

#### Structured data

Structured data is data that adheres to a pre-defined data model and is therefore straightforward to analyze.
This data model is the description of our data, each record has to be conform to the model.
A table in a spreadsheet is a good example of the concept of structured data however often no data types are enforced, meaning a column can contain e.g. both numbers and text.
Later we will see that a mixture of data types is often problematic therefor the need of a data model.

#### Unstructured data

In contrast to structured data, there is no apparent data model but this does not mean the data is unusable or cluttered.
Usually it means either no data model has yet been applied or we are dealing with data that is difficult to confine in a model.
A great example of this would be images, or more general (binary) files.
These obviously are hard to sort yet often data structures also contain metadata from these files, with data describing things as when the file was uploaded, what is shown in the file, ...
In turn the metadata can be structured and a data model can be related to the unstructured data.

#### Semi-structured data

As an intermediate option, we have what is called semi-structured data.
The reasoning behind this is that the concept of tables is not always applicable, in some occasions e.g. data lakes there is no complex structure present compared to a database.
In a data lake files are stored similar to the folder structure in your computer, with no fancy infrastructure behind it, thus reducing operation costs.
This implies that a data model can not be enforced and the data is stored in generic files.

### Data Structures

There are several structures in which data can be stored and accessed, here we cover the 3 most important.

#### Data Lake

As mentioned earlier a data lake would be the most cost efficient method as it relies on the least infrastructure and can be serverless.
The concept behind a data lake is straight-forward, the data is stored in simple files with a specific notation e.g. parquet, csv, xml,...
What is important when designing a data lake would be partitioning, this can be achieved by using subfolders and saving parts of the data in different files.
To make this more tangible, take a look at this symbolic [example](https://github.com/LorenzF/data-science-practical-approach/tree/main/src/c2_data_preparation/data/temperatures) I provided.
Instead of putting all data in one csv file, subfolder divide the data in Country, City and then the year.
We could even further partition yet the data is here in daily frequency so that would create many small partitions.
The difficulty for a data lake lies in the method of interacting, when adding new data one has to adhere to a agreed upon data model that is not enforced, meaning you could create incorrect data which then need to be cleaned.
On the other hand efficiency of you data lake depends on good partitioning, as the order of divisioning of your folders. We could have also divided first on year and then on country and city.
As a data scientist seeing the data lake might not be as common, as this is rather an engineering task, however using the concepts of a data lake in experimental projects can make a big difference. 

#### Database

Another interesting data structure is the database, widely used for exceptional speeds and ease of use, yet costly in storage.
Numerous implementations of servers using the SQL language are developed over the years with each their own dialect and advantages.
The important take home message here is that you can easily perform queries on the database that pre-handles the data to retrieve the information you need.
these operations include filtering, grouping categories, joining tables, ordering and much more, as SQL is a complete language on its own. 
As a data scientist these databases are much more common, so SQL is a good asset to learn! 

#### Data Warehouse

A next step towards data analysis is the data warehouse, where a database is composed of the most pragmatic method of storing your data a data warehouse consist of multiple views on your data.
Based upon the data of a dataset the data warehouse transforms this data into a new format that displays the data in a new way.
Let me illustrate with with a simple example, we have a database with a table that contains the rentals of books from multiple libraries.
This table has a few columns: a timestamp, the library, the action (rent, return, ...), the client_id and the book_id.
If you would want to know if a book is available this database is perfect for your needs as you just have to find the last event for that book and if its a return the book is (or should be) there.
Now image we would want to know how many books are being rented per month this database is insufficient, yet our data warehouse might contain such a view!
It is up to the data engineer/scientist to create a computation that displays the amount of books rented per month.
If they also would like to subdivided it per category of books, you would need to incorporate another table of the database where information of the books is stored.
More on these operations of a data warehouse will be seen in the data preprocessing chapter.
One last remark about data warehousing, it is important to optimize between memory and computation.
Tables in our data warehouse compared to database can be computed in place reducing memory costs yet increasing computation costs.
If a visualization tool often queries a table in your warehouse it is favorable to create it as a table in your database.

### OLTP and OLAP

From the previous section you might have deduced that a database and Data Warehouse serve 2 different purposes.
These are denoted as OnLine Transaction Processing and OnLine Analytical Processing, as the names suggest these are used for transactional and analytical processes.

#### OLTP

For this method the database structure is optimal, let us review the example where we have libraries renting out books.
Renting out a book would send a message to our OLTP system creating a new record stating that specific book is at this moment rented out from our library.
OLTP handles day-to-day operational data that can be both written and read from our database. 

#### OLAP

In the case we would like to analyse data from the libraries we would use the OLAP method, creating multi-dimensional views from our transactional data.
Our dimensions would be the date (aggregated per month), the library and the category of book, the chapter of data preprocessing will use these operations practically.
I could write a whole chapter on OLAP operations however they are well described in [this](https://en.wikipedia.org/wiki/OLAP_cube) wikipedia page.