import os
import warnings

from typing import Tuple
from mapie.regression import MapieQuantileRegressor, MapieRegressor

from cp.logger import Logger as _logger
logger = _logger()


def train_strategies(params: dict, base_estimator: dict,
                     strategies_params: dict, strategies_name: dict) -> Tuple[dict, dict, dict]:
    y_pred, int_pred, mapie_estimator = {}, {}, {}  
    X_train, X_test, y_train = params['X_train'], params['X_test'], params['y_train']
    miscoverage, seed = params['miscoverage'], params['seed']
    
    warnings.filterwarnings("ignore")
    for _strat, _strat_params in strategies_params.items():
        logger.info(f"Training {strategies_name[_strat]}")
        if _strat != 'CQR':
            mapie = MapieRegressor(base_estimator[_strat], **_strat_params)
            mapie.fit(X_train, y_train)
            y_pred[_strat], int_pred[_strat] = mapie.predict(X_test, alpha=miscoverage)

        else:
            mapie = MapieQuantileRegressor(base_estimator[_strat], **_strat_params)
            mapie.fit(X_train, y_train, random_state=seed)
            y_pred[_strat], int_pred[_strat] = mapie.predict(X_test)
        mapie_estimator[_strat] = mapie

    return y_pred, int_pred, mapie_estimator
