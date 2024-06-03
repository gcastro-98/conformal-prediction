# Conformal prediction
Uncertainty quantification for distribution-free and data-agnostic problems is applied in terms of conformal prediction (CP) methodologies. 
In particular, this is applied to regression problems involving both exchangeable and time-series data. 
This work was carried out as part of (UB 2023) MSc thesis development.

## Exchangeable data

### Toy problem

A toy problem is proposed, before dealing with more complex datasets, according to this [Kaggle discussion](https://www.kaggle.com/code/dipuk0506/toy-dataset-for-regression-and-uq/notebook). 

### Regression problem

The same dataset as the [`mapie`'s CQR tutorial](https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_cqr_tutorial.html) is proposed: the `sklearn` built-in [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). 

Chosen in view of being simple and reproducible, in particular no feature engineering is needed; it is composed of 20,640 samples of the following 8 different features:
- The median income in block group
- The median house age in block group
- The average number of rooms per household
- The average number of bedrooms per household
- The block group population
- The average number of household members
- The location (latitude & longitude) of the block group
- The label variable: the median house price for a given block group.

## Non-exchangeable data

### Time series problem

The same dataset as the [`mapie`'s time series tutorial](https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_ts-tutorial.html) was chosen: the Victoria electricity demand dataset, used in the book “_Forecasting: Principles and Practice_” [1].

It contains a total of 1340 samples and deals with an electricity demand forecasting problem: which not only features daily and weekly seasonality, but it is also impacted by temperature. Thus, apart from the electricty demand lagged up to 7 days (and other time features), temperature will be used as exogenous variable.

[1] _Forecasting: principles and practice_. Hyndman, R.J. and Athanasopoulos, G. ISBN: 9780987507105. 2014. OTexts. [Link](https://books.google.es/books?id=gDuRBAAAQBAJ).