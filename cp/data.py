"""
This module contains the data retrieval functions for the different problems.
"""

from os import makedirs, path
from math import pi as _pi
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from scipy.stats import norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SEED: int = 123
np.random.seed(SEED)
makedirs('input/', exist_ok=True)

C_STRONG: str = '#9B0A0A'
C_MEDIUM: str = '#800000'
C_LIGHT: str = '#F2BFBF'


# AUXILIARY FUNCTIONS

def split(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, ...]:
    """
    Split the data into training and testing sets.
    """
    return train_test_split(
        X.values, y.values, test_size=0.3, random_state=seed, shuffle=True)


# ########################################################################
# EXCHANGEABLE DATA
# ########################################################################

#  TOY PROBLEM

class ToyProblem:
    """
    Base class for toy problem's data.
    """
    def __init__(self, N: int = 5000, noise: float = 0.1):
        self.N = N
        self.noise = noise
        self.first_data = self._first_data()
        self.second_data = self._second_data()

    def _first_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve a dataset with the following characteristics:
        - Single Variable Input
        - Homoscedastic Uncertainty
        - Low Uncertainty
        - Trigonometric Relationships between Input and Output
        """
        X = np.arange(0, 2 * _pi, 2 * _pi / self.N)  
        y = np.sin(2 * X) + self.noise * (np.random.rand(self.N) - 1 / 2)
        return pd.DataFrame({'X': X}), pd.Series(y, name='y')

    def first_intervals(self, X: np.ndarray, miscoverage: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve lower and upper bounds for the true confidence intervals 
        (analitcally estimated using the CLT) of the first toy problem.
        """
        # Note \epsilon * (U - 1/2) has expectation 0 and variance \epsilon^2 / 12
        # thus according the CLT, we have that \sqrt{N} * \bar{Y} converges to a 
        # normal distribution with mean \sin(2X) and variance \epsilon^2 / 12.
        # Namely, the confidence intervals are given by:
        # \bar{Y} \pm z_{\alpha / 2} * \epsilon / \sqrt{12}
        _y_dif = np.full((X.shape[0], ), -norm.ppf(miscoverage/2) * self.noise / np.sqrt(12))
        _y_mean = np.sin(2 * X.ravel())
        return _y_mean - _y_dif, _y_mean + _y_dif

    def _second_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve a dataset with the following characteristics:
        - Two Variable Input
        - Heterocedastic Uncertainty
        - High Uncertainty
        - Trigonometric Relationships between Input and Output
        """
        X1 = np.arange(0, 2 * _pi, 2 * _pi / self.N) 
        X2 = np.arange(0, 5 * _pi, 5 * _pi / self.N)  

        y = np.sin(2 * X1) + 5 * self.noise * (np.random.rand(self.N) - 1 / 2) * (1 - np.sin(X2)) * np.cos(X2)
        return pd.DataFrame({'X1': X1, 'X2': X2}), pd.Series(y, name='y')

    def second_intervals(self, X1: np.ndarray, X2: np.ndarray, miscoverage: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve lower and upper bounds for the true confidence intervals 
        (analitcally estimated using the CLT) of the first toy problem.
        """
        # Note \epsilon * (U - 1/2) has expectation 0 and variance \epsilon^2 / 12
        # thus according the CLT, we have that \sqrt{N} * \bar{Y} converges to a 
        # normal distribution with mean \sin(2X) and variance \epsilon^2 / 12.
        # Namely, the confidence intervals are given by:
        # \bar{Y} \pm z_{\alpha / 2} * \epsilon / \sqrt{12}
        _y_dif = -norm.ppf(miscoverage/2) * 5 * self.noise / np.sqrt(12) * (1 - np.sin(X2)) * np.cos(X2)
        _y_mean = np.sin(2 * X1.ravel())
        return _y_mean - _y_dif, _y_mean + _y_dif


#  REGRESSION PROBLEM (Exoplanets: predicting the planet's mass)


class RegressionProblem:
    """
    Base class for regression problem's data.

    The [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), 
    from `sklearn`, was chosen for reproducibility and simplicity in the 
    feature engineering step (in particular, there's no `nan` & just 
    numerical variables). 

    This dataset is composed of 20,640 samples of 8 different features:
    - The median income in block group
    - The median house age in block group
    - The average number of rooms per household
    - The average number of bedrooms per household
    - The block group population
    - The average number of household members
    - The location (latitude & longitude) of the block group
    While the label variable is:
    - The median house price for a given block group

    """
    def __init__(self, N: int = 5000, noise: float = 0.1):
        self._label_name = 'MedHouseVal'
        self.df: pd.DataFrame = fetch_california_housing(as_frame=True).frame
        self.data: pd.DataFrame = (self._get_features(), self._get_label())

    def visualize_data(self, **kwargs) -> pd.DataFrame:
        """
        Create and display a pairplot of the dataset.
        """
        pd.plotting.scatter_matrix(
            self.df, figsize=(10,10), color=C_LIGHT, 
            hist_kwds={'color': C_STRONG, 'bins': 25},)
        plt.savefig(kwargs.get('save_path', 'output/regression/data-regression-problem.png'), dpi=200)
        plt.show()
        

    def _get_features(self) -> pd.DataFrame:
        return self.df.drop(columns=[self._label_name])

    def _get_label(self) -> pd.DataFrame:
        return self.df[self._label_name]
    

# ########################################################################
# TIME SERIES DATA
# ########################################################################

# ELECTRICITY DEMAND PROBLEM

class TimeSeriesProblem:
    """
    Base class for time series' data.
    """
    def __init__(
            self, 
            n_lag: int = 5, 
            test_days: int = 7, 
            right_pad_hours: int = 0,
            with_change_point: bool = False):
        """
        Time series data, for electricity demand forecasting problem, containing a 
        total of 1340 samples.

        Parameters:
        -----------
        n_lag : int
            Number of lagged features to consider in the model.
        test_days : int
            Number of days to consider for the test set.
        right_pad : int
            Number of hours to consider for padding the right side of the data.
            If right_pad is > 0, the test set will be interspersed with the training set.
        with_change_point : bool
            If True, then the electricity demand is decreased by 2 GW in the middle of test 
            set (simulating, this way, an exogenous event).

        """
        self._with_change_point: bool = with_change_point
        self._n_samples: int = 1340
        self._n_lag: int = n_lag 
        self._num_test_steps: int = 24 * test_days
        self._right_pad: int = right_pad_hours
        assert 0 <= self._right_pad <= self._n_samples - self._num_test_steps, \
            f"Right pad must be between 1 and {self._n_samples} - test_days * 24 hours = {self._n_samples} - {test_days * 24} = {self._n_samples - test_days * 24}"
        
        self.features = ["Weekofyear", "Weekday", "Hour", "Temperature"] 
        self.features += [f"Lag_{hour}" for hour in range(1, self._n_lag)]
        self.label = "Demand"
        
        self.train_df, self.test_df = self._get_train_test_df()
    
    def visualize_data(self, save_path: str = None) -> None:
        plt.figure(figsize=(15, 5))
        if self._right_pad > 0:
            self.train_df['Demand'].iloc[:-self._right_pad].plot(color=C_STRONG, label='Train')
            self.test_df['Demand'].plot(color=C_LIGHT, label='Test')
            self.train_df['Demand'].iloc[-self._right_pad:].plot(color=C_STRONG)
        else:
            self.train_df['Demand'].plot(color=C_STRONG, label='Train')
            self.test_df['Demand'].plot(color=C_LIGHT, label='Test')
        
        plt.legend()
        plt.ylabel("Hourly demand (GW)")
        plt.xlabel("Date")
        if save_path is not None:
            plt.savefig(save_path, dpi=250)
        plt.show()
        plt.close()
        
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = self.train_df[self.features]
        y_train = self.train_df[self.label]
        X_test = self.test_df[self.features]
        y_test = self.test_df[self.label]
        return X_train.values, X_test.values, y_train.values, y_test.values
        
    def _get_train_test_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not path.exists('input/demand_temperature.csv'):
            url_file = "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/examples/data/demand_temperature.csv"
            df = pd.read_csv(url_file, parse_dates=True, index_col=0)
            df["Date"] = pd.to_datetime(df.index)
            df["Weekofyear"] = df.Date.dt.isocalendar().week.astype("int64")
            df["Weekday"] = df.Date.dt.isocalendar().day.astype("int64")
            df["Hour"] = df.index.hour
            for hour in range(1, self._n_lag):
                df[f"Lag_{hour}"] = df["Demand"].shift(hour)
            
            df.to_csv('input/demand_temperature.csv', index=False)
        else:
            df = pd.read_csv('input/demand_temperature.csv')

        if self._right_pad > 0:
            train_df: pd.DataFrame = df.iloc[:-self._num_test_steps-self._right_pad, :].copy()
            test_df: pd.DataFrame = df.iloc[-self._num_test_steps-self._right_pad:-self._right_pad, :].copy()
            train_df = pd.concat([train_df, df.iloc[-self._right_pad:, :].copy()], axis=0)
        else:
            train_df = df.iloc[:-self._num_test_steps, :].copy()
            test_df = df.iloc[-self._num_test_steps:, :].copy()

        train_df = train_df.loc[~np.any(train_df[self.features].isnull(), axis=1)]
        if self._with_change_point:
            test_df[self.label].iloc[len(test_df)// 2:] -= 2

        return train_df, test_df
        

# ################ PADDED TIMESERIES FOR K-FOLDS ################

def compute_test_days_and_pad_multiplicator(K: int) -> None:
    n_samples = TimeSeriesProblem()._n_samples
    pad_hour_multiplicator = int(n_samples // K)
    test_days = int(pad_hour_multiplicator // 24)

    return test_days, pad_hour_multiplicator


def visualize_ts_K_folds(K, with_change_point: bool = False, save_path: str = None) -> None:
    """
    Visualize the K-folds for the time series problem.
    """
    plt.figure(figsize=(15, 5))
    test_days, pad_hours_multiplicator = compute_test_days_and_pad_multiplicator(K)
    colors: np.ndarray = _get_k_folds_cmap()(plt.Normalize(0, K)(np.arange(1, K + 1)))

    _train_df, _test_df = TimeSeriesProblem()._get_train_test_df()
    pd.concat([_train_df, _test_df], axis=0)['Demand'].plot(
        color='black', ls='--', lw=1, 
        label='Original' if with_change_point else 'Original')
    for _j in range(K):
        _, test_df = TimeSeriesProblem(
            test_days=test_days, 
            right_pad_hours=int(pad_hours_multiplicator * _j), 
            with_change_point=with_change_point)._get_train_test_df()
        test_df['Demand'].plot(color=colors[_j], label=f'Fold nº {_j + 1}')
    

    plt.legend()
    plt.ylabel("Hourly demand (GW)")
    plt.xlabel("Date")
    if save_path is not None:
        plt.savefig(save_path, dpi=250)
    plt.show()
    plt.close()


def _get_k_folds_cmap(n_colors: int = 256):
    return mcolors.LinearSegmentedColormap.from_list(
        "folds_cmap", [C_LIGHT, C_STRONG], N=n_colors)