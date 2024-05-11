from mapie.metrics import (coverage_width_based,
                           regression_coverage_score,
                           regression_mean_width_score)
from sklearn.metrics import mean_squared_error
from numpy import ndarray, sqrt as _sqrt
from cp.logger import Logger as _logger

logger = _logger()


def rmse(y_pred: dict, y_test: ndarray) -> dict:
    errors = {}

    for _strat in y_pred.keys():
        logger.info(f"Validating {_strat} RMSE")
        errors[_strat] = _sqrt(mean_squared_error(
            y_test, y_pred[_strat]))
    return errors


def cwc(intervals: dict, y_test: ndarray, miscoverage: float) -> dict:
    """
    Coverage Width-based Criterion (CWC) obtained by the prediction intervals.
    
    The effective coverage score is a criterion used to evaluate the quality
    of prediction intervals (PIs) based on their coverage and width.
    
    Khosravi, Abbas, Saeid Nahavandi, and Doug Creighton.
    "Construction of optimal prediction intervals for load forecasting
    problems."
    IEEE Transactions on Power Systems 25.3 (2010): 1496-1503.

    Usage
    -----
    >>> cwb = coverage_width_based(y_true, y_preds_low, y_preds_up, eta, alpha)
    
    alpha: float
        The level of miscoverage of the prediction intervals.
    eta : float
        A user-defined parameter that balances the contributions of
        Mean Width Score and Coverage score in the CWC calculation.
    """

    scores = {}

    for _strat in intervals.keys():
        logger.info(f"Validating {_strat} CWC")
        scores[_strat] = coverage_width_based(
            y_test,
            intervals[_strat][:, 0, 0],
            intervals[_strat][:, 1, 0],
            eta=0.01, alpha=miscoverage)

    return scores


def coverage(intervals: dict, y_test: ndarray) -> dict:
    coverages = {}

    for _strat in intervals.keys():
        logger.info(f"Validating {_strat} coverage")
        coverages[_strat] = regression_coverage_score(
            y_test, 
            intervals[_strat][:, 0, 0],
            intervals[_strat][:, 1, 0])
    
    return coverages


def width(intervals: dict) -> dict:
    widths = {}

    for _strat in intervals.keys():
        logger.info(f"Validating {_strat} width")
        widths[_strat] = regression_mean_width_score(
            intervals[_strat][:, 0, 0],
            intervals[_strat][:, 1, 0])
    
    return widths
