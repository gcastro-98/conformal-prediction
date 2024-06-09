import warnings
from typing import Any, List
from time import process_time
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor  # as underlying model
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform  # for the random search hyperparameters
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import regression_coverage_score_v2
from cp import data, ts, logger as _logger, validate

import pandas as pd
import numpy as np
SEED: int = 123
np.random.seed(SEED)

logger = _logger.Logger()


# ######## MODEL FINE-TUNING ########

def fine_tune_lgbm(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    estimator = LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        random_state=SEED,
        verbose=0
    )
    params_distributions = dict(
        num_leaves=randint(low=10, high=50),
        max_depth=randint(low=3, high=20),
        n_estimators=randint(low=50, high=100),
        learning_rate=uniform()
    )

    optim_model = RandomizedSearchCV(
        estimator,
        param_distributions=params_distributions,
        n_jobs=-1,
        n_iter=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        verbose=0,
        random_state=SEED
    )

    logger.info("Computing best hyperparameters from randomized search")
    logger.debug(4 * " " + "This may take a while (around 30')")
    optim_model.fit(X_train, y_train)

    return optim_model.best_params_

def fine_tune_rf_for_ts(X_train: np.ndarray, y_train: np.ndarray, 
                        n_iter: int = 100,
                        n_splits: int = 5
                        ) -> dict:
    rf_model = RandomForestRegressor(random_state=SEED)
    rf_params = {
        "max_depth": randint(2, 30), 
        "n_estimators": randint(10, 100)
    }
    cv_obj = RandomizedSearchCV(
        rf_model,
        param_distributions=rf_params,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=SEED,
        verbose=0,
        n_jobs=-1,
    )

    logger.info("Computing best hyperparameters from randomized search")
    logger.debug(4 * " " + "This may take a while")
    cv_obj.fit(X_train, y_train)

    return cv_obj.best_params_


# ######## CROSS-VALIDATIONS FOR METRICS TABLE ########

# ### REGRESSION PROBLEM ###

def regression_metrics(
        X: pd.DataFrame, y: pd.Series, 
        miscoverage: float, 
        base_estimators: Any,
        strategy_params: dict, 
        seed: int, K: int = 5,
        silent: bool = False,
        **kwargs) -> None:
    """
    Perform a K-fold cross validation and return the metrics table for all the strategies.

    **Note**: this can be applied just to the exchangeable case because, 
    otherwise, the autocorrelation in the data would be kept due to the shuffling.

    """
    coverage: np.ndarray = np.zeros((len(strategy_params), K))
    width: np.ndarray = np.zeros((len(strategy_params), K))
    train_time: np.ndarray = np.zeros((len(strategy_params), K))
    pred_time: np.ndarray = np.zeros((len(strategy_params), K))
    rmse: np.ndarray = np.zeros((len(strategy_params), K))
    cwc: np.ndarray = np.zeros((len(strategy_params), K))
    ssc: np.ndarray = np.zeros((len(strategy_params), K))

    for _i, _strat in enumerate(list(strategy_params.keys()), start=0):
        if not silent:
            logger.debug(4 * " " + f"Computing metrics for strategy {_strat} and miscoverage {miscoverage}")

        for _j, (train_index, test_index) in enumerate(KFold(n_splits=K, random_state=seed, shuffle=True).split(X), start=0):
            if not silent:
                logger.debug(8 * " " + f"Training fold {_j + 1}/{K}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            warnings.filterwarnings("ignore")
    
            if _strat != 'CQR':
                mapie = MapieRegressor(base_estimators[_strat], **strategy_params[_strat])

                start = process_time()
                mapie.fit(X_train, y_train)
                train_time[_i, _j] = process_time() - start

                start = process_time()
                _y_pred, _int_pred = mapie.predict(X_test, alpha=miscoverage)
                pred_time[_i, _j] = process_time() - start

            else:
                strategy_params[_strat].update({'alpha': miscoverage})
                mapie = MapieQuantileRegressor(base_estimators[_strat], **strategy_params[_strat])

                start = process_time()
                mapie.fit(X_train, y_train, random_state=seed)
                train_time[_i, _j] = process_time() - start

                start = process_time()
                _y_pred, _int_pred = mapie.predict(X_test)
                pred_time[_i, _j] = process_time() - start

            int_pred, y_pred = {_strat: _int_pred}, {_strat: _y_pred}
            coverage[_i, _j] = validate.coverage(int_pred, y_test)[_strat]
            width[_i, _j] = validate.width(int_pred)[_strat]
            rmse[_i, _j] = validate.rmse(y_pred, y_test)[_strat]
            cwc[_i, _j] = validate.cwc(int_pred, y_test, miscoverage)[_strat]
            ssc[_i, _j] = validate.cond_coverage(_int_pred, y_test, num_bins=kwargs.get('num_bins', 10))[_strat]

    # and we print the results
    for _i, _strat in enumerate(strategy_params.keys(), start=0):
        print(f"Miscoverage: {miscoverage}")
        print(f"Coverage: {np.mean(coverage[_i, :]):.2f} ± {np.std(coverage[_i, :]):.2f}")
        print(f"Width: {np.mean(width[_i, :]):.2f} ± {np.std(width[_i, :]):.2f}")
        print(f"RMSE: {np.mean(rmse[_i, :]):.2f} ± {np.std(rmse[_i, :]):.2f}")
        print(f"CWC: {np.mean(cwc[_i, :]):.2f} ± {np.std(cwc[_i, :]):.2f}")
        print(f"SSC: {np.mean(ssc[_i, :]):.2f} ± {np.std(ssc[_i, :]):.2f}")
        print(f"Training time: {np.mean(train_time[_i, :]):.2f} ± {np.std(train_time[_i, :]):.2f}")
        print(f"Prediction time: {np.mean(pred_time[_i, :]):.2f} ± {np.std(pred_time[_i, :]):.2f}")
           
    return None


# ######## CROSS-VALIDATIONS FOR COVERAGE EXPERIMENTS ########

# ### REGRESSION PROBLEM ###

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


# ### TIMESERIES PROBLEM ###

def ts_coverage_in_function_of_alpha(
        miscoverages_list: list, 
        base_model_params: Any, 
        strategy_name: str, 
        with_change_point: bool = False,
        silent: bool = False, 
        K: int = 10) -> np.ndarray:
    """
    Perform a K-fold cross validation and return the coverage of the predicted confidence intervals in function of the miscoverage level. 
    Thus, the returned array is of shape (len(miscoverages_list), K).

    **Note**: in the case of timeseries, and in order to break as much autocorrelation as possible,
      the test data is obtained by making continuous holes in the middle of the data.

    """
    coverages_arr: np.ndarray = np.zeros((len(miscoverages_list), K))

    for _i, miscoverage in enumerate(miscoverages_list, start=0):
        if not silent:
            logger.debug(4 * " " + f"Computing coverage scores for alpha = {miscoverage} ({_i + 1}/{len(miscoverages_list)})")

        test_days, pad_hours_multiplicator = data.compute_test_days_and_pad_multiplicator(K)

        for _j in range(K):
            if not silent:
                logger.debug(8 * " " + f"Training {strategy_name} for fold {_j + 1}/{K}")

            ts_problem = data.TimeSeriesProblem(
                test_days=test_days, right_pad_hours=int(pad_hours_multiplicator * _j),
                with_change_point=with_change_point)
            X_train, X_test, y_train, y_test = ts_problem.get_arrays()
            
            warnings.filterwarnings("ignore")
            if strategy_name != 'EnbPI':  # then we train EnbPI without partial fit
                _, _int_pred, _ = ts.train_without_partial_fit(X_train, y_train, X_test, miscoverage, RandomForestRegressor(**base_model_params))

            else:
                _, _int_pred, _ = ts.train(X_train, y_train, X_test, y_test, miscoverage, RandomForestRegressor(**base_model_params))
                
            coverages_arr[_i, _j] = float(regression_coverage_score_v2(y_test, _int_pred))
    return coverages_arr