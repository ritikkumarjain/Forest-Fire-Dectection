# Forest-Fire-Detection

In this repo we will talk about Forest Fires detection. Forest Fires pose a threat not only to the forest wealth but also to the entire regime to fauna and flora seriously disturbing the bio-diversity and the ecology and environment of a region. During summer, when there is no rain for months, the forests become littered with dry senescent leaves and twinges, which could burst into flames ignited by the slightest spark.

To solve this problem machine learning plays an important role.ML would help us to predict the chaotic nature of wildfires. ML would help us to identify major fire hotspots and their severity.

I have used Forest fire dataset from kaggle(link:https://www.kaggle.com/elikplim/forest-fires-data-set)

During data analysis the data was found to have high skewness and kurtosis.
I tried to transform the data first but it still doesn't bring any change to skewness and kurtosis.
The I removed the outlier values using z-score method that helped.

The model is trained on XGB class of scikit learn library.
Then we found out the out sample to in sample ratio(OSE/ISE) = 1.0396922767800738.
These implies our model is ok.

Then we plotted the learning curve which also shows that model has a good fit.
And at last we solved our model so that we can simply deploy it later when needed.

