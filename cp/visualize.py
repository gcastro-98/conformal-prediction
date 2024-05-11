import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

os.makedirs('output', exist_ok=True)
C_STRONG: str = '#9B0A0A'
C_MEDIUM: str = '#800000'
C_LIGHT: str = '#F2BFBF'


def _sort(data: Dict, ref: np.ndarray, subsample: float = None) -> Dict:
    indices = np.argsort(ref)
    if subsample is not None:
        indices = np.random.choice(
            indices, replace=False,
            size=int(len(indices) * subsample))

    return {_k: _v[indices] for _k, _v in data.items()}


def data(
    points: Dict, 
    bounds=None, 
    intervals=None, 
    ax=None, 
    **kwargs):
    # points = _sort(points)
    
    if ax is None:
        fig, ax = plt.subplots()

    if bounds is not None:
        bounds = _sort(bounds, bounds['X'])
        # we plot the predicted bounds
        ax.fill_between(bounds['X'], bounds['y_low'], bounds['y_up'], alpha=0.9, color=C_LIGHT,
        label=kwargs.get('bounds_label', 'Predicted intervals'))

    if intervals is not None:
        intervals = _sort(intervals, intervals['X'])
        # we plot the true confidence intervals
        ax.plot(intervals['X'], intervals['y_low'], color='black', lw=0.7)
        ax.plot(intervals['X'], intervals['y_up'], color='black', lw=0.7, label=kwargs.get('intervals_label', 'True intervals'))
    
    ax.scatter(points['X'], points['y'], color=C_STRONG, label=kwargs.get('points_label', 'Data points'),s=kwargs.get('s', 0.2))

    ax.set_xlabel(kwargs.get('xlabel', "X"))
    ax.set_ylabel(kwargs.get('ylabel', "Y"))
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    ax.legend()

    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/data.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax

def goodness(
    y_test,
    y_pred,
    lower_bound,
    upper_bound,
    coverage,
    width,
    rmse,
    cwc,
    ax=None,
    subsample=None,
    **kwargs
):
    # we sort the data for proper visualization
    # and, if subsample not None, we take only the passed percentage
    _sorted = _sort(
        {'test': y_test, 'pred': y_pred, 
        'low': lower_bound, 'up': upper_bound}, 
        y_test, subsample)
    y_test, y_pred = _sorted['test'], _sorted['pred']
    lower_bound, upper_bound = _sorted['low'], _sorted['up']

    if ax is None:
        fig, ax = plt.subplots()

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))

    error = y_pred - lower_bound
    _outside = (y_test > y_pred + error) + (y_test < y_pred - error)
    ax.errorbar(
        y_test[~_outside],
        y_pred[~_outside],
        yerr=np.abs(error[~_outside]),
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        color=C_STRONG,
        label="Inside PI"
        )
    ax.errorbar(
        y_test[_outside],
        y_pred[_outside],
        yerr=np.abs(error[_outside]),
        capsize=5, marker="o", elinewidth=2, linewidth=0, 
        color=C_LIGHT,
        label="Outside PI"
        )
    ax.scatter(
        y_test[_outside],
        y_test[_outside],
        marker="*", color="black",
        label="True value"
    )
    ax.legend(loc='lower right')
    ax.set_xlabel(kwargs.get('xlabel', "Groundtruth"))
    ax.set_ylabel(kwargs.get('ylabel', "Prediction"))

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, '--', alpha=0.75, color="black", label="x=y")

    # # and finally, the coverage and width as text
    ax.annotate(
        f"Coverage: {np.round(coverage, 3)}\n"
        + f"Interval width: {np.round(width, 3)}\n"
        + f"CWC: {np.round(cwc, 3)}\n"
        + f"RMSE: {np.round(rmse, 3)}",
        xy=(0., 0.), # point to annotate
        xytext=(np.min(y_test) * 1.175, np.max(y_pred) * 0.72), 
        bbox=dict(boxstyle="round", fc="white", ec="grey", lw=1)
    )

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontweight='bold')
    
    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/goodness.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax


