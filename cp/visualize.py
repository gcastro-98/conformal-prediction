import os
import six
from typing import Dict, List
from mapie.metrics import regression_ssc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        _, ax = plt.subplots()

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
        _, ax = plt.subplots()

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

    # and finally, some metrics as text
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


# ######## MARGINAL VS. CONDITIONAL COVERAGE ########

# WIDTH SIZE OCURRENCE

def width_size_occurrence(
    intervals: np.ndarray, 
    train_intervals: np.ndarray = None,
    num_bins: int = None, 
    ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    _width = np.abs(intervals[:, 0, 0] - intervals[:, 1, 0])

    if np.std(_width) < 1e-10:
        # then all the intervals are the same width 
        # thus, the histogram will not be displayed
        _width[0], _width[1] = (1 + 1e-3) * _width[0], (1 - 1e-3) * _width[1]

    if train_intervals is not None:
        _width_train = np.abs(train_intervals[:, 0, 0] - train_intervals[:, 1, 0])
        if np.std(_width_train) < 1e-10:
            _width_train[0] = (1 + 1e-3) * _width_train[0]
            _width_train[1] = (1 - 1e-3) * _width_train[1]

        ax.hist(_width, bins=num_bins, color=C_STRONG, label='Test data')
        ax.hist(_width_train, bins=num_bins, color=C_LIGHT, alpha=0.7, label='Train data')
        ax.legend()
    else:
        ax.hist(_width, bins=num_bins, color=C_STRONG)

    if 'x_lim' in kwargs:
        ax.set_xlim(kwargs['x_lim'])
    ax.set_xlabel(kwargs.get('xlabel', 'Interval width'))
    ax.set_ylabel(kwargs.get('ylabel', 'Occurrence'))
    # ax.legend()

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontweight='bold')

    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/width-occurrence.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax

# COVERAGE BY WIDTH SIZE

def coverage_by_width(
    y_test: np.ndarray, 
    intervals: np.ndarray, 
    miscoverage: float,
    cond_coverage: float,
    hsic_coeff: float,
    num_bins: int = 3, 
    ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(np.arange(num_bins),
           regression_ssc(y_test, intervals, 
                          num_bins=num_bins)[0],
                          color=C_STRONG)
    ax.axhline(y=1 - miscoverage, color='black', linestyle='-')
    
    ax.set_ylabel(kwargs.get('ylabel', "Coverage"))
    ax.set_xlabel("Interval width")
    
    ax.tick_params(
        axis='x', which='both', 
        bottom=False, top=False, 
        labelbottom=False)
    
    ax.annotate(
        f"SSC score: {np.round(cond_coverage, 4)}\n" 
        + f"HSIC coefficient: {np.round(hsic_coeff, 4)}",
        xy=(0., 0.), # point to annotate
        xytext=(-0.1, 0.05), 
        bbox=dict(boxstyle="round", fc="white", ec="grey", lw=1)
    )
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/coverage-by-width.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax


# METRICS TABLE

def dicts_to_dataframe(metrics: Dict[str, dict]) -> pd.DataFrame:
    columns: List[str] = list(metrics.keys())
    dicts: List[dict] = list(metrics.values())
    rows: List[str] = dicts[0].keys()
    df = pd.DataFrame({f: [_d.get(f, None) for _d in dicts] for f in rows}).transpose()
    df.columns = columns
    return df


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([.25, 1])) * np.array([col_width, row_height])
        _, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=np.round(data.values, 3), bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        cell.set_text_props(ha='center')
        if k[0] == 0:  # if this is a column header cell
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(C_STRONG)
        elif k[1] == -1:  # if this is a row header cell
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(C_LIGHT)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax


def dataframe_to_png(df, filename):
    _ = render_mpl_table(df, header_columns=0, col_width=2.0)
    plt.savefig(filename)


# ######## COVERAGE SOUGTH vs. ALPHA for a given strategy ########

def coverage_by_alpha(
    coverages_arr: np.ndarray,
    miscoverages_list: List[float],
    strategy_name: str, ax=None, 
    **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    sought_coverage = [coverages_arr[_i, :] for _i in range(len(miscoverages_list))][::-1]
    expected_coverage = np.round(1 - np.array(miscoverages_list), 2).tolist()[::-1]

    ax.set_ylabel("Sought coverage")
    ax.set_xlabel("Expected coverage")

    bplot = ax.boxplot(sought_coverage, patch_artist=True)
    ax.plot(
        range(1, len(miscoverages_list) + 1), 
        expected_coverage,
        color='black', linestyle='--', 
        label='Ideal case',
        )
    ax.set_xticks(range(1, len(miscoverages_list) + 1), 
                  labels=expected_coverage)
    ax.set_yticks(expected_coverage, 
                  labels=expected_coverage)
    
    for patch in bplot['boxes']:
        patch.set_facecolor(C_STRONG)
    
    # ax.tick_params(
    #     axis='x', which='both', 
    #     bottom=False, top=False, 
    #     labelbottom=False)
    
    # ax.annotate(
    #     f"SSC score: {np.round(cond_coverage, 4)}\n" 
    #     + f"HSIC coefficient: {np.round(hsic_coeff, 4)}",
    #     xy=(0., 0.), # point to annotate
    #     xytext=(-0.1, 0.05), 
    #     bbox=dict(boxstyle="round", fc="white", ec="grey", lw=1)
    # )
    
    ax.set_title(strategy_name)

    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/coverage-by-alpha.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax