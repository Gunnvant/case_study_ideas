## **Problem statement and data**
You have been given access to data from the data warehouse of an online aggregator for properties that tourists can rent out for short durations. The current sample is from all the listings in Antwerp Belgium.

The task is to generate an ML based solution that can be used to suggest appropriate listing price to the property owner when they try to list a property out for rent.

The dataset was pulled from a datawarehouse and has the following tables:

1. Calendar (Data on listings in Chronological Order)
    - listing_id (Unique id for each listing)	
    - date	(Date on which listing was made)
    - available	(1->Available, 0->Rented out)
    - price	(Price in USD)
    - adjusted_price (Adjusted Price USD, use `price` column instead of this for any analysis)	
    - minimum_nights (Minimum number of nights a guest can book)	
    - maximum_nights (Maximum number of nights a guest can book)

2. Listings (Data on details of a particular listing)
    - listing_id (Unique id for each listing)
    - listing_url (Unique url to each listing)
    - name (Name of the listing)
    - description (Description of the listing)
    - lat (latitude)
    - long (longitude)
    - property_type (Type of property)
    - room_type (Type of room)
    - accommodates (Number of guests that can be accommodated)
    - bathrooms (Number of bathrooms)
    - bedrooms (Number of bedrooms)
    - beds (Number of beds)
    - amenities (List of amenities)
    - host_id (Unique Id of the host)
3. Hosts (Data on hosts who've posted their listing)
    - host_id (Unique id of host)
    - host_name (Name of the host)
    - host_since (Timestamp of host registration on the platform)
    - host_location (Location of host)
    - host_about (Self reported about section of each host)
4. Reviews (Data on Reviews)
    - listing_id (Unique id for each listing)
    - review_id (Unique review id)
    - date (Date of posting the review)
    - reviewer_id (Id of reviewer)
    - reviewer_name (Name of the reviewer)
    - comments (Review comment)


## **Dataset Details**
The dataset is given to you in an sqlite database named [`airbnb.db`](../../data/airbnb/airbnb.db). This has the tables listed above. You can use `sqlalchemy` or `sqlite3` and `pandas` to read data from this database. You can also export the data as csv files from this db and then read it in using pandas. Refer to this [link](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html) on more details about fetching data 

## **Tasks**
### Data Understanding and feature creation:

- Look at the table `Calendar` how many rows and unique listing ids are present? Are there any implications when it comes to having more rows and less unique listing ids?
- Look at the `price` column in `Calendar` table. What transformations you will need to perform so that you can create a column that can be used as a target/response variable?
- Look at the tables `Listings`, `Hosts` and `Reviews` to come up with a list of potential transformations needed in order to have predictors that can be used to predict the listing price.
- Create an aggregated view of data spread across different tables, containing the target as well as predictor variables.

### Data Quality and checks:
- Once the aggregated dataset has been created, do a data audit. Create a data quality report which has the following basic structure:
    - Continuous Variables: (#unique values, percentage_missing_values, min, max, average, 25th percentile, 75th percentile, 90th percentile, 95th percentile)
    - Categorical Variables: (#Unique values, percentage_missing_values)
- Highlight any data anomaly that you find and fix it.

### Variable profiling and checking relationships between variables:
- Assess the relationship between target and predictor variables. You can compute correlations, plot bivariate relationships. 
- Based on the above analysis summarize your findings and list down the transformations you will do on different predictors, remove the variables from further analysis.

### Modelling and insights:
- Explain your approach on creating train/test/validation splits.
- Create a comparison matrix to compare different regression models you've run
- Experiment with Linear Regression, Regression Trees, Random Forest Regressor and GBM. Not compulsory but you can also experiment with Xgboost, Lightgbm
- Explain which model you've finalized and why did you finalize the model.
- Explain what are the top 5 most important predictors and also explain the direction of impact of these top 5 predictors on the response variable.

### Deliverables:
- Finalize your code and push it to Github repository or create well structured collab notebooks or kaggle kernels. Make sure your code is well structured, break it down into modular files or notebooks.
- Finalize your findings in a markdown document and push it to your github account or create kaggle kernels, collab notebooks (notebooks that just explain your findings). You can also create a deck and push it to slideshare as well.

