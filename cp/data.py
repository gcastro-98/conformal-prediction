"""
This module contains the data retrieval functions for the different problems.
"""

from os import makedirs
from math import pi as _pi
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from scipy.stats import norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        plt.savefig(kwargs.get('save_path', 'output/data-regression-problem.png'), dpi=200)
        plt.show()
        

    def _get_features(self) -> pd.DataFrame:
        return self.df.drop(columns=[self._label_name])

    def _get_label(self) -> pd.DataFrame:
        return self.df[self._label_name]