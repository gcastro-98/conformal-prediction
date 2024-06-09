from mapie.metrics import (coverage_width_based,
                           regression_coverage_score,
                           regression_coverage_score_v2,
                           regression_mean_width_score)
from mapie.metrics import (hsic, regression_ssc_score)
from sklearn.metrics import mean_squared_error
from numpy import ndarray, sqrt as _sqrt, abs as _abs, std as _std
from cp.logger import Logger as _logger

logger = _logger()


def rmse(
        y_pred: dict, y_test: ndarray, 
        silent: bool = False) -> dict:
    errors = {}

    for _strat in y_pred.keys():
        if not silent:
            logger.info(f"Validating {_strat} RMSE")
        errors[_strat] = _sqrt(mean_squared_error(
            y_test, y_pred[_strat]))
    return errors


def cwc(
        intervals: dict, y_test: ndarray, 
        miscoverage: float, eta: float = 0.01,
        silent: bool = False) -> dict:
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
        if not silent:
            logger.info(f"Validating {_strat} CWC")
        scores[_strat] = coverage_width_based(
            y_test,
            intervals[_strat][:, 0, 0],
            intervals[_strat][:, 1, 0],
            eta=eta, alpha=miscoverage)

    return scores


def coverage(
        intervals: dict, y_test: ndarray, 
        type: str = 'v2', silent: bool = False) -> dict:
    coverages = {}

    if type == 'v2':  # we use 'regression_coverage_score_v2'
        _func = lambda _y, _int: regression_coverage_score_v2(_y, _int)
    else:
        _func = lambda _y, _int: regression_coverage_score(
            _y, _int[:, 0, 0], _int[_strat][:, 1, 0])

    for _strat in intervals.keys():
        if not silent:
            logger.info(f"Validating {_strat} coverage")
        coverages[_strat] = float(_func(y_test, intervals[_strat]))
    
    return coverages


def width(
        intervals: dict, 
        silent: bool = False) -> dict:
    widths = {}

    for _strat in intervals.keys():
        if not silent:
            logger.info(f"Validating {_strat} width")
        widths[_strat] = regression_mean_width_score(
            intervals[_strat][:, 0, 0],
            intervals[_strat][:, 1, 0])
    
    return widths


# CONDITIONAL vs. MARGINAL COVERAGE

def cond_coverage(
        intervals: dict, y_test: ndarray, 
        num_bins: int = 3, silent: bool = False) -> dict:
    """
    For each strategy: aggregate by the minimum for each alpha the Size-Stratified Coverage.
    Then, it returns the maximum violation of the conditional coverage 
    (with the groups defined).
    """
    cond_coverages = {}

    for _strat in intervals.keys():
        if _std(_abs(intervals[_strat][:, 0, 0] - intervals[_strat][:, 1, 0])) < 1e-10:
            cond_coverages[_strat] = 0.
            if not silent:
                logger.warning("This metric should be used only with non constant intervals (intervals of different sizes), with constant intervals the result may be misinterpreted.")        
                logger.warning(f"Size-Stratified Coverage score set to 0 for {_strat}")
            continue 
        if not silent:
            logger.info(f"Validating {_strat} Size-Stratified Coverage")
        cond_coverages[_strat] = float(regression_ssc_score(
            y_test, intervals[_strat][:, :, 0], num_bins=num_bins))
        
    return cond_coverages


def hsic_coefficient(
        intervals: dict, y_test: ndarray,
        silent: bool = False) -> dict:
    """
    Compute the square root of the hsic coefficient for each strategy. 
    HSIC is Hilbert-Schmidt independence criterion that is a 
    correlation measure. It is used as proposed in [1], to 
    compute the correlation between the indicator of coverage 
    and the interval size.

    [1] Feldman, S., Bates, S., & Romano, Y. (2021). 
    Improving conditional coverage via orthogonal 
    quantile regression. Advances in Neural 
    Information Processing Systems, 34, 2060-2071.
    """
    coef_correlations = {}

    for _strat in intervals.keys():
        if not silent:
            logger.info(f"Validating {_strat} HSIC coefficient")
        coef_correlations[_strat] = float(hsic(y_test, intervals[_strat]))
            
    return coef_correlations
    
 


