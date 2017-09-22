# Machine Learning Engineer Nanodegree
## Capstone Project

Aditya Ranganathan           
September 16, 2017

## New York City Taxi Trip Duration Prediction
## I. Definition

### Project Overview
New York is one of the most crowded cities in the world and has a major traffic issue. Most New Yorkers take a taxi/Uber/Lyft/etc to get from point A to point B. Knowing in advance how long a taxi trip will take given the pickup location, drop location etc would be invaluable for the proper allocation of taxis to customers. The dataset used for this project was published by the NYC Taxi and Limousine Commission as part of a Kaggle Playground competition to improve taxi trip time duration prediction. The new competition has a similar objective to the ECML/PKDD trip time challenge hosted on Kaggle in 2015. The challenge is a Playground challenge promoting collaboration between competitors and is an excellent opportunity to learn from expert Kagglers. The competitors were also encouraged to find external datasets that might help in creating a better model.

The problem at hand is a regression task requiring our model to predict trip duration given other details about the trip. This problem can be solved by using regression models in sci-kit learn. Neural networks could also be trained to predict trip duration. As to why this is an important problem, in order to improve the efficiency of electronic taxi dispatching systems it is important to be able to predict how long a driver will have his taxi occupied. If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better informed to identify which driver to assign to each pickup request. This would in turn result in shorter waiting times for people availing these taxi services and could alleviate the traffic congestion in the city to an extent. 

The aim of this project is to build a model that accurately predicts trip duration of taxi rides taken in New York City. My personal motivation for this project is to familiarise myself with Kaggle and competing in Data Science competitions in general.

### Problem Statement
The problem at hand is to predict the duration of a taxi trip using features like pickup/dropoff coordinates, date/time of trip, weather, number of passengers etc. If taxi companies could accurately predict trip duration times it would result in much shorter waiting time for the customer. This is a regression problem and can be solved by fitting a regression model to the data. The data will first be explored through visualizations. Then the dataset will be converted to a usable form and the most promising features will be chosen to be incuded in the final training data. Dimensionality reduction will be carried out depending on the model being trained and multiple models will be trained and evaluated. A K-nearest-neighbor model and a neural network will be tested to see how it fares. Gradient boosted trees have been a favourite when it comes to winning Kaggle competitions so an XGBoost model will also be explored. We are trying to achieve a lower Root Mean Squared Logarithmic Error (RMSLE) than the RMSLE of the benchmark model's.

### Metrics
The evaluation metric used for this project and for the Kaggle competition is the Root Mean Squared Logarithmic Error or RMSLE. The RMSLE is calculated by taking the square root of the average of the squared log error between the predictions and actual values. RMSLE is usually used when you don't want to penalize huge differences in the predicted and the actual values when both predicted and true values are huge numbers. This metric can be used when you want to penalize under estimates more than over estimates. This makes sense since underestimating trip time duration would result in increase in the wait time for the customers of the taxi service comapared to an overestimate where the taxi drivers would be the ones waiting. The equation used for calculating RMSLE can be found [here](https://www.kaggle.com/c/nyc-taxi-trip-duration#evaluation).

## II. Analysis

### Data Exploration

The dataset used for this project is the one provided by Kaggle at their competition page [here](https://www.kaggle.com/c/nyc-taxi-trip-duration/data). The training set contains 1458644 rows of training data and the test set contains 625134 rows of data. We have sufficient data to train our regression models. The features contained in the dataset are

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

Engineered features

* distance - the distance the taxi travelled calculated using the pickup and dropoff locations
* pickup cluster - the cluster that the trip originated in after clustering
* dropoff cluster - the cluster that the trip terminated in after clustering

| id        | ven id | pass count | pickup long | pickup lat | dropoff long | dropoff lat | s_f flag | trip dur | pickup month | pickup dayof week | pickup hour | pickup day | pickup second | max temp | min temp | avg temp | preci | snow fall | snow depth | log trip dur | dist | pick c | drop c |
|---|-----------|-----------|-----------------|------------------|-----------------|-------------------|------------------|--------------------|---------------|--------------|------------------|-------------|------------|---------------|--------------------|---------------------|---------------------|---------------|-----------|------------|-------------------|----------|----------------|-----------------|
| id2875421 | 2         | 1               | -73.982          | 40.768          | -73.965           | 40.766           | N                  | 455           | 3            | 0                | 17          | 14         | 6369878       | 51                 | 40                  | 45.5                | 0.29          | 0         | 0          | 6.120             | 1.502    | 8              | 4               |
| id2129090 | 1         | 1               | -73.975          | 40.759          | -73.953           | 40.765           | N                  | 1346          | 3            | 0                | 14          | 14         | 6357922       | 51                 | 40                  | 45.5                | 0.29          | 0         | 0          | 7.205             | 1.976    | 19             | 4               |
| id0256505 | 1         | 1               | -73.994          | 40.745          | -73.999           | 40.723           | N                  | 695           | 3            | 0                | 15          | 14         | 6361461       | 51                 | 40                  | 45.5                | 0.29          | 0         | 0          | 6.544             | 2.514    | 9              | 11              |
| id3863815 | 2         | 3               | -73.944          | 40.714          | -73.911           | 40.709           | N                  | 755           | 3            | 0                | 4           | 14         | 6323059       | 51                 | 40                  | 45.5                | 0.29          | 0         | 0          | 6.627             | 2.912    | 7              | 14              |
| id3817493 | 2         | 1               | -73.953          | 40.766          | -73.979           | 40.762           | N                  | 1050          | 3            | 0                | 14          | 14         | 6361059       | 51                 | 40                  | 45.5                | 0.29          | 0         | 0          | 6.957             | 2.232    | 4              | 8               |

The above table shows us the first 5 rows of the training data. The datetime features have been split into 'pickup_month', 'pickup_dayofweek', 'pickup_hour', 'pickup_day' and 'pickup_second'. The month, day, weekday and hour features will be one hot encoded for the models that require one hot encoding to be carried out on the features. The distance feature is calculated using the pickup/dropoff coordinates. The pickup and dropoff clusters are determined after K-means clustering is done on the pickup/dropoff coordinates.

|       | ven id | pass count | pickup long | pickup lat | dropoff long | dropoff lat | trip dur | pickup month | pickup dayof week | pickup hour | pickup day | pickup second | max temp | min temp | avg temp | preci | snow fall | snow depth | log trip dur | dist    |
|-------|-----------|-----------------|------------------|-----------------|-------------------|------------------|---------------|--------------|------------------|-------------|------------|---------------|--------------------|---------------------|---------------------|---------------|-----------|------------|-------------------|-------------|
| count | 1437168   | 1437168         | 1437168          | 1437168         | 1437168           | 1437168          | 1437168       | 1437168      | 1437168          | 1437168     | 1437168    | 1437168       | 1437168            | 1437168             | 1437168             | 1437168       | 1437168   | 1437168    | 1437168           | 1437168     |
| mean  | 1.53      | 1.66            | -73.97           | 40.75           | -73.97            | 40.75            | 824.82        | 3.52         | 3.05             | 13.62       | 15.50      | 7896344.67    | 61.76              | 46.73               | 54.24               | 0.09          | 0.06      | 0.41       | 6.45              | 3.29        |
| std   | 0.50      | 1.31            | 0.04             | 0.03            | 0.03              | 0.03             | 649.06        | 1.68         | 1.95             | 6.38        | 8.70       | 4455288.37    | 16.98              | 15.72               | 16.15               | 0.23          | 0.94      | 2.25       | 0.77              | 3.67        |
| min   | 1         | 0               | -74.029       | 40.630     | -74.029     | 40.630      | 1             | 1            | 0                | 0           | 1          | 0             | 15                 | -1                  | 7                   | 0             | 0         | 0          | 0                 | 0           |
| 25%   | 1         | 1               | -73.99           | 40.74           | -73.99            | 40.74            | 394           | 2            | 1                | 9           | 8          | 4117631.75    | 49                 | 36                  | 42.5                | 0             | 0         | 0          | 5.976       | 1.224 |
| 50%   | 2         | 1               | -73.98           | 40.75           | -73.98            | 40.75            | 655           | 4            | 3                | 14          | 15         | 7920125       | 61                 | 46                  | 53.5                | 0             | 0         | 0          | 6.484      | 2.068 |
| 75%   | 2         | 2               | -73.97           | 40.77           | -73.96            | 40.77            | 1056          | 5            | 5                | 19          | 23         | 11674783.5    | 76                 | 61                  | 69.5                | 0.04          | 0         | 0          | 6.962      | 3.767 |
| max   | 2         | 6               | -73.75           | 40.85           | -73.75            | 40.85            | 21411         | 6            | 6                | 23          | 31         | 15724762      | 92                 | 75                  | 83                  | 2.31          | 27.3      | 22         | 9.971       | 27.196 |

The above table shows us important statistics about a few important features in our dataset. From the table its clear that there are a few outliers in the dataset. The trip_duration has a maximum value of of almost 980 hours which is clearly an outlier that might affect our model. The pickup/dropoff latitude/longitude features also have outliers which need to be removed. The weather seems to be free from any outliers.

The target variable 'trip_duration' has a skewed distribution with positive skew. This can be converted into a normal distribution with a simple log transformation.

For the problem of predicting travel times the location of the pickup and drop point will be important. The day of the week and time of the day will also be useful since the traffic would change with respect to time and location. The weather data could also be very useful since factors like snow, rain etc could have a significant effect on trip times. The number of passengers could also potentially be useful in prediction.

### Exploratory Visualization

<p align="center">
  <img src="https://image.ibb.co/mmtwm5/trip_duration.png">
  <img src="https://image.ibb.co/n5h6m5/log_trip_duration.png">
</p> 

The two figures above show us the distributions of the target variable 'trip_duration' before and after log transformation. We can see that the log transformation removes the positive skew of the distribution to a large extent. The 2 figures above were created after removing outliers that had a difference of 4 standard deviations from the mean.

<p align="center">
  <img src="https://preview.ibb.co/n4X3zQ/kmeans.png">
</p> 

The above figure shows new york city after K-means clustering was applied to the taxi pickup and drop points. This effectively splits the city into different neighbourhoods. The pickup and dropoff clusters might be helpful as a feature while predicting trip duration as different neighbourhoods might have differences in traffic, quality of roads etc.

<p align="center">
  <img width="460" height="300" img src="https://preview.ibb.co/eO2E65/test.png">
  <img width="460" height="300" img src="https://preview.ibb.co/n5JMm5/train.png">
</p> 

The two figures above show the distribution of pickup points in the train data and the test data. We can see that the train data and the test data are from the same areas and mostly overlap. This suggests that a K-Nearest Neighbor model might be a good fit for the data.

### Algorithms and Techniques

The models that are going to be trained and tested are as follows

* K-Nearest Neighbour regressor - There is a complete overlap of the test and train data. There is also no need for the data to be extrapolated to new ranges of data like in other regression problems. This makes KNN a suitable machine learning model for this dataset. KNN's are however very sensitive to the scale of the data, so feature scaling will have to be carried out on the dataset. The KNN algorithm is also severely affected by the curse of dimensionality so the number of features will have to be reduced by using a suitable feature selection technique.

* Decision Tree based models - A RandomForest model and a gradient boosted trees model will be trained and evaluated. Decision Tree based models tend to be very effective at regression problems. These models also do not require categorical variables to be one hot encoded. They also do not require the features to be scaled so feature scaling can be skipped for these models. The number of estimators for the randomforest and other hyperparameters will be selected through gridsearch.

* Neural Network Regressor - A neural network will also be designed and tested. This will be done using Keras. 

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark

A simple benchmark model to compare all the models to would be a model that predicts the mean trip duration for all the data points in the test set. Such a model achieves a RMSLE of 0.892 on the Kaggle public leaderboard. A simple LinearRegression model trained on the data scores an RMSLE score of 0.559. All trained models can be compared to this score. Our trained models should have significantly better performance than the benchmark model.

## III. Methodology

### Data Preprocessing

The dataset was first verified to make sure that it does not contain any missing values. Next the pickup and dropoff datetime was converted into pandas datetime objects and were split into seperate features. The new features are 'pickup_month', 'pickup_dayofweek', 'pickup_hour', 'pickup_day' and 'pickup_second'. Certain features in the weather data had to be converted into numerical values from strings. Some rows in the weather data have a 'T' value for certain features which means 'Trace amounts of'. This had to be converted to a value of 0.01. The training dataset and the weather data were then merged.

Outlier detection and removal was carried out. Rows that had 'trip_duration' value outside of 4 standard deviations from the mean were removed from the training data. Latitude and longitude values that were outside New York city limits were also removed from the dataset. Next the distance feature was calculated using the 'geopy' package using their 'vincenty' distance function. Next K-means clustering was carried out on the pickup/dropoff coordinates to cluster the city into neighborhoods. A K value of 25 was used effectively splitting the city into 25 neighborhoods. A 'pickup_cluster' and 'dropoff_cluster' feature was added to the test and train datasets.

One hot encoding was carried out on the categorical variables in the data. This was not done on the 'pickup_cluster', 'dropoff_cluster' and 'pickup_day' features since just those 3 features would have resulted in 81 new features. Working with this large dataset was attempted but was not feasible due to computational limitations. The training data and test data were saved to new csv files 'new_train' and 'new_test'. The training data with the one hot encoded data was saved in 'new_train_cat' and 'new_test_cat'.

Running PCA on the large categorical variable dataset did not show any improvement in the cross validation error so PCA was not carried out in the final version of the data. The data was scaled using the 'RobustScaler' transformer in sci-kit learn before giving it to the K-nearest neighbour model. No scaling was done for both the RandomForest and XGBoost models.

### Implementation

* K-Nearest Neighbor model - Since KNN models perform poorly with datasets with a large number of features the 'new_train' data was used to train it. This contained the data without one hot encoding. Scaling was carried out since KNN models are very sensitive to the scale of the data. After trying out different combinations of features it was observed that leaving out the weather data and other features like 'passenger_count', 'vendor_id' and 'store_and_fwd_flag' improved the error of the model. This brought down the error and had a reasonable training and prediction time. Trying to do PCA on the larger one hot encoded dataset resulted in the computer freezing so PCA was abandoned.

* Tree based models - Since tree based models do not require feature scaling or one hot encoding the training data did not undergo scaling. Also the original data without one hot encoding was used. A Randomforest regressor was trained using the RandomForestRegressor estimator in sci-kit learn. An XGBoost model was also trained using the xgboost package. 

* Neural Network model - Various neural network architectures were tried using keras. Architectures having a large number of nodes had very poor performance. Increasing the number of hidden layers also negatively affected the performance of the network. The final network had four fully connected layers including the input and output layers. The sigmoid, linear and relu activation functions were tried and their impact on performance was evaluated. 

A function to calculate RMSLE was also implemented.

### Refinement

For the KNN model the initial model using the default values trained with the full unscaled data had an RMSKE score of 0.52. After trying out the StandardScaler, Normalizer, MinMaxScaler, RobustScaler and QuantileTransformer scaling methods in sci-kit learn the RobustScaler was found to reduce the crossvalidation error the most. After scaling, using fewer features and using 10 neighbors for the model the RMSLE improved to a value of 0.441. 

The random forest model wit the default hyperparameters had an initial RMSLE score of 0.442 which improved to 0.409 after increasing the number of trees in the model to 40. The XGBoost model initially had a good RMSLE of 0.41 with the default parameters. This improved to 0.397 after decresing max depth and tuning other hyperparamters. 

Different architectures and activation functions were tried out for the neural network model. With unscaled data the neural network was unable to properly train. The first few architectures were too large and had very high RMSLE scores of 0.80 and above. The neural network was trained on the one hot encoded data with 65 feature columns. The final architecture that seemed to give the best performance had the following architecture [65->32->16->1]. Adding the number of hidden layers or the number of nodes did not help improve the error. The 'relu' activation function was found to be most effective at regression. This improved the RMSLE score to 0.443.


## IV. Results

### Model Evaluation and Validation

The final model selected was the XGBoost model with an RMSLE score of 0.397 on the Kaggle private leaderboard. The model performed equally well with both the one hot encoded data and the unencoded datasets. There was also no requirement for any scaling transformations. The model is not very sensitive since the results do not significantly change with changes in the input data. The final model is reasonable and does align with our expectations as it does significanty better than the benchmark model score. The RMSLE score of 0.397 is for data that was previously unseen as it was predicting on the Kaggle test set. The results can be trusted within a small margin of error. Since the predictions are being made on the seconds scale the errors will not result in very long waiting times for taxi customers. All the RMSLE scores reported in this report are the scores given to the predictions by the Kaggle leaderboard so there is no concern of data leakage. The models had lower crossvalidation scores when being evaluated on a local crossvalidation split. The RMSE error metric was used to train the XGboost model. This is not a problem since the RMSE and the RMSLE are correlated metrics.

The final XGBoost model hyperparamters that minimised local crossvalidation RMSE are as follows

*  "objective" = "reg:linear"
*  "eta" = 0.3
*  "min_child_weight" = 30
*  "subsample" = 0.8
*  "colsample_bytree" = 0.3
*  "scale_pos_weight" = 1.0
*  "silent" = 1
*  "max_depth" = 10
*  "eval_metric"= "rmse"

### Justification

The XGBoost model with a RMSLE score of 0.397 outperforms the benchmark model which had a RMSLE score of 0.559. The benchmark model was a linear regression that was trained on the full one hot encoded data. The final solution requires both the original training data and the weather data. The data needs to be preprocessed and features such as distance need to be calculated. Clustering must be done to assign pickup/dropoff clusters. The full dataset must be trained for around 200 iterations with the above mentioned hyperparameters to converge to an RMSLE score of 0.397. The final model does solve the problem of predicting taxi trip times to a certain extent but this only applies to test data from the six month timeframe that the model was trained on. How it peforms on a different time of the year is unknown. The solution does perform reasonably well in the Kaggle competition leaderboard.

## V. Conclusion

### Free-Form Visualization

<p align="center">
  <img src="https://preview.ibb.co/jtGRM5/xgb.png">
</p> 

The above visualisation shows us the feature importances of the final XGBoost model. From the visualisation we can see that our initial assumptions are correct. The most important features are the pickup and dropoff coordinates. The calculated distance is also given a high importance value. The pickup and dropoff clusters are also important. The other features like pickup hour, day etc have comparitively lower importance. The chart also shows us that the average temperature and other weather data have moderate importance.  

### Reflection

The most challenging part of the project was catering to the strengths and weaknesses of each individual machine learning model. This meant splitting the preprocessing pipeline into seperate operations to best transform the data to each model. This is why the feature creation notebooks and the model notebooks have been seperated. Hyperparameter tuning for each model was also quite challenging. A comprehensive gridsearch was not done in many cases due to the high computational cost. Selecting a neural network architecture was also challenging since there are no fixed guidelines on how to go about designing architecture. The importance given to the pickup and dropoff cluster by the xgboost model was interesting to see.

The final model does reasonably well when thought of in the terms of the competition. But this could also be due to the train and test data coming form the same 6 month period of the same year. In the real world a trained model would be predicting trip times in a time frame where very little data is available. For example the current snow depth, precipitaton etc would not be available to a real world model. The effectiveness of the model in a truly general setting is unknown since the train and test data is from the same timeframe

### Improvement

There have been other datasets related to the challenge released on the Kaggle discussion forums. These datasets include more details about the trips taken like the shortest route, number of steps taken etc. There are many diverse datasets that other Kagglers have been using to help with this problem. There is even a dataset containing the nights that new york city had parties in certain locations collected by analysing noise complainy data. Including these extra features could be beneficial to the performance of our model. Feature selection and dimensionality reduction could also improve the models. The deep learning approach could also be explored. Only a few network architectures were tested due to high training times. Stacking of different models predictions could also be done to improve the final leaderboard RMSLE score. A better solution certainly exists as the current number 1 on the leaderboard has an RMSLE score of 0.289. However there was also an unfortunate leak of test data predictions 5 days before the end of the competition. This leak has not been incorporated in my predicitions. Prior to the leak the number 1 score was around 0.36. There were new algorithms that I learned of through Kaggle like a genetic algorithm that finds the optimal pipeline for prediction in sci-kit learn (Tpot package). This package was not used because it takes a few hours to a few days for the algorithm to finish.

-----------

