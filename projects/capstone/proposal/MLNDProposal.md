# Machine Learning Engineer Nanodegree
## Capstone Proposal
Aditya Ranganathan  
September 10th, 2017

## New York City Taxi Trip Duration Prediction

### Domain Background

New York is one of the most crowded cities in the world and has a serious traffic issue. Most New Yorkers take a taxi/Uber/Lyft/etc to get from point A to point B. This dataset was published by the NYC Taxi and Limousine Commission as part of a Kaggle Playground competition to improve taxi trip time duration prediction. The new competition has a similar objective to the ECML/PKDD trip time challenge hosted on Kaggle in 2015. The challenge is a Playground challenge promoting collaboration between competitors and is an excellent oppurtunity to learn from expert Kagglers. The competitors were also encouraged to find external datasets that might help in creating a better model.

The problem at hand is a regression task requiring our model to predict trip duration given other details about the trip. This problem can be solved by using regression models in sci-kit learn. Neural networks could also be trained to predict trip duration. As to why this is an important problem, in order to improve the efficiency of electronic taxi dispatching systems it is important to be able to predict how long a driver will have his taxi occupied. If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better able to identify which driver to assign to each pickup request. This would in turn result in shorter waiting times for people using these taxi services and could alleviate the traffic problem in the city to an extent. 

The aim of this project is to build a model that accurately predicts trip duration of taxi rides taken in New York City. My personal motivation for this project is to familiarise myself with Kaggle and competing in Data Science competitions in general.

### Problem Statement

The problem at hand is to predict the duration of a taxi trip using features like pickup/dropoff coordinates, date/time of trip, weather, number of passengers etc. If taxi companies could accurately predict trip duration times it would result in much shorter waiting times for the customer. This is a regression problem and can be solved by fitting a regression model to the data. A K-nearest-neighbor model or neural network can also be tested to see how it fares. Gradient boosted trees have been a favourite when it comes to winning Kaggle competitions so an XGBoost model will also be explored. The evaluation metric for this competition is the Root Mean Squared Logarithmic Error or RMSLE. 

### Datasets and Inputs

The dataset used for this project is the one provided by Kaggle at their competition page [here](https://www.kaggle.com/c/nyc-taxi-trip-duration/data). The training set contains 1458644 rows of training data and the test set contains 625134 rows of data. We have sufficient data to train our regression models. The target variable 'trip_duration' has a skewed distribution with positive skew. This can be converted into a normal distribution with a simple log transformation. The features contained in the dataset are

* id - a unique identifier for each trip
* vendor_id - a code indicating the provider associated with the trip record
* pickup_datetime - date and time when the meter was engaged
* dropoff_datetime - date and time when the meter was disengaged
* passenger_count - the number of passengers in the vehicle (driver entered value)
* pickup_longitude - the longitude where the meter was engaged
* pickup_latitude - the latitude where the meter was engaged
* dropoff_longitude - the longitude where the meter was disengaged
* dropoff_latitude - the latitude where the meter was disengaged
* store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory  before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
* trip_duration - duration of the trip in seconds

Additional weather data for the corresponding days in the dataset were also used which can be found [here](https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016). The weather data was collected from the National Weather Service. The features contained in that dataset are

* date - the day in question
* maximum temperature - the maximum temperature observed on that day
* minimum temperature - the minimum temperature observed on that day
* average temperature - the average temperature observed on that day
* precipitation - the amount of precipitation on that day 
* snow fall - the amount of new snowfall observed on that day
* snow depth - the depth of snow cover on that day

For the problem of predicting travel times the location of the pickup and drop point will be important. The day of the week and time during the day wil also probably be useful since the traffic would change with respect to time and location. The weather data could also be very useful since factors like snow, rain etc could have a significant effect on trip times. The number of passengers could also potentially be useful in prediction.

### Solution Statement

For this machine learning problem the most suited model could be a RandomForest regressor or a boosted trees regressor. K-nearest neighbors and neural nets could also potentially be effective. Decision tree based models are better suited to unscaled data like our dataset so a tree based approach could be effective without having to do any additional feature scaling. Also turning the date features like the day of the month, day of week, day of year etc into categorical variables would vastly increase the number of features that the model has to deal with. Therefore a combination of decision tree models would probably be best to predict the trip times. 

### Benchmark Model

A simple benchmark model to compare all the models to would be a model that predicts the mean trip duration for all the data points in the test set. Such a model achieves a RMSLE of 0.892 on the Kaggle public leaderboard. A simple LinearRegression model trained on the data scores an RMSLE score of 0.762. All trained models can be compared to this score. Our trained models should have significantly better performance than the benchmark model.


### Evaluation Metrics

The evaluation metric for this competition is the Root Mean Squared Logarithmic Error or RMSLE. The RMSLE is calculated by taking the log of the predictions and actual values. RMSLE is usually used when you don't want to penalize huge differences in the predicted and the actual values when both predicted and true values are huge numbers. This metric can be used when you want to penalize under estimates more than over estimates. This makes sense since underestimating trip time duration would result in increase in the wait time for the customers of the taxi service comapared to an overestimate where the taxi drivers would be the ones waiting. The equation used for calculating RMSLE can be found [here](https://www.kaggle.com/c/nyc-taxi-trip-duration#evaluation).

### Project Design

* Programming language and libraries used
    * Python 3.5
    * pandas, numpy - For data manipulation
    * matplotlib - For data visualisation
    * geopy - For calculating distances using lat/long values
    * scikit-learn - Open source machine learning library
    * xgboost - Gradient Boosting package for python
    * keras - Deep learning Library 
* Workflow 

    * The first step in the project will be data exploration and to check the data for any missing values. The date/time features will also have to be converted from the format provided into more useful features such as day of week, pickup hour etc. The weather data will also have to be loaded and merged with the original dataset. Outlier detection and removal will also be carried out. The distance between the pickup and dropoff points will also be calculated.

    * The next step would be to visualise the data. Various visualisations will be used to potentially spot important features and their correlation to the target variable. 

    * K-Means clustering can be carried out to seperate the pickup/dropoff latitudes and longitudes into clusters. This will effectively split the city into a number of neighborhoods. This would be helpful since different neighborhoods might have different traffic charecteristics. The trip time would probably depend on which locality the trip originated in and ended in.

    * The next step would be to build the models and compare their performance with each other and the benchmark model. A K-nearest neighbor, RandomForests, Gradient Boosted tree model and potentially a neural network will be trained on the data. One hot encoding and scaling will be done on the dataset to see if performance improves compared to the unencoded unscaled data. If the dimensionality of the data increases too much after one hot encoding PCA could also be run to reduce the dimensionality of the dataset being given to the model.

    * The best model will be selected for hyperparameter tuning and final training and evaluation will be carried out.
