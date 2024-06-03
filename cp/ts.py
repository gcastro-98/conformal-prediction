from mapie.subsample import BlockBootstrap
from mapie.time_series_regression import MapieTimeSeriesRegressor
from typing import Any

from cp import logger as _logger 

import numpy as np
SEED: int = 123
np.random.seed(SEED)

logger = _logger.Logger()

ENBPI_PARAMS = {
    'cv': BlockBootstrap(n_resamplings=100, length=48, overlapping=True, random_state=SEED),
    'agg_function': "mean", 
    'n_jobs':-1
}


def train_without_partial_fit(
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray,
        miscoverage: float, 
        base_model: Any, 
        silent: bool = False) \
    -> MapieTimeSeriesRegressor:
    enbpi = MapieTimeSeriesRegressor(base_model, method="enbpi", **ENBPI_PARAMS)
    if not silent:
        logger.info("Traning EnbPI without partial fit to adjust residuals")
        logger.debug(4 * " " + "This may take a while")
    enbpi.fit(X_train, y_train)
    if not silent:
        logger.info("Inferring")
    y_pred, int_pred = enbpi.predict(
        X_test, alpha=miscoverage, 
        ensemble=True, optimize_beta=True)  
    
    return y_pred, int_pred, enbpi

def train(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray,
    y_test: np.ndarray,
    miscoverage: float, 
    base_model: Any,
    gap: int = 1, 
    silent: bool = False) \
    -> MapieTimeSeriesRegressor:

    enbpi = MapieTimeSeriesRegressor(base_model, method="enbpi", **ENBPI_PARAMS)
    if not silent:
        logger.info("Traning EnbPI")
        logger.debug(4 * " " + "This may take a while")
    enbpi.fit(X_train, y_train)

    y_pred, int_pred = np.zeros((X_test.shape[0], )), np.zeros((X_test.shape[0], 2, 1))
    if not silent:
        logger.info("Inferring while adjusting residuals (partial fit)")
    y_pred[:gap], int_pred[:gap, :, :] = enbpi.predict(
        X_test[:gap, :], alpha=miscoverage, ensemble=True, optimize_beta=True
    )
    
    for step in range(gap, len(X_test), gap):
        if not silent:
            logger.debug(4 * " " + f"Adjusting residuals for step {step}")
        enbpi.partial_fit(
            X_test[(step - gap):step, :],
            y_test[(step - gap):step])
        
        (y_pred[step:step + gap], int_pred[step:step + gap, :, :],) = \
            enbpi.predict(
                X_test[step:(step + gap), :],
                alpha=miscoverage,
                ensemble=True,
                optimize_beta=True)

    return y_pred, int_pred, enbpi