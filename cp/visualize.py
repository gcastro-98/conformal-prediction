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

