import warnings
from typing import Any, List
from sklearn.model_selection import KFold
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import regression_coverage_score_v2
from cp import logger as _logger

import pandas as pd
import numpy as np
SEED: int = 123
np.random.seed(SEED)

logger = _logger.Logger()


def coverage_in_function_of_alpha(X: pd.DataFrame, y: pd.Series, miscoverages_list: List[float], base_estimator: Any, strategy_params: dict, strategy_name: dict, 
                                  seed: int, K: int = 5, silent: bool = False) -> np.ndarray:
    """
    Perform a K-fold cross validation and return the coverage of the predicted confidence intervals in function of the miscoverage level. 
    Thus, the returned array is of shape (len(miscoverages_list), K).

    **Note**: this can be applied just to the exchangeable case because, 
    otherwise, the autocorrelation in the data would be kept due to the shuffling.

    """
    coverages_arr: np.ndarray = np.zeros((len(miscoverages_list), K))

    for _i, miscoverage in enumerate(miscoverages_list, start=0):
        if not silent:
            logger.debug(4 * " " + f"Computing coverage scores for alpha = {miscoverage} ({_i + 1}/{len(miscoverages_list)})")

        for _j, (train_index, test_index) in enumerate(KFold(n_splits=K, random_state=seed, shuffle=True).split(X), start=0):
            if not silent:
                logger.debug(8 * " " + f"Training {strategy_name} for fold {_j + 1}/{K}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            warnings.filterwarnings("ignore")
    
            if strategy_name != 'CQR':
                mapie = MapieRegressor(base_estimator, **strategy_params)
                mapie.fit(X_train, y_train)
                _, _int_pred = mapie.predict(X_test, alpha=miscoverage)

            else:
                strategy_params.update({'alpha': miscoverage})
                mapie = MapieQuantileRegressor(base_estimator, **strategy_params)
                mapie.fit(X_train, y_train, random_state=seed)
                _, _int_pred = mapie.predict(X_test)

            coverages_arr[_i, _j] = float(regression_coverage_score_v2(y_test, _int_pred))
    return coverages_arr
            
