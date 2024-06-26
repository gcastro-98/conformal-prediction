import os
import six
from typing import Dict, List, Any
from mapie.metrics import regression_ssc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs('output', exist_ok=True)
C_STRONG: str = '#9B0A0A'
C_MEDIUM: str = '#800000'
C_LIGHT: str = '#F2BFBF'
C_PRED: str = '#F87500'


# ########### AUXILIARY FUNCTIONS ###########

def _sort(data: Dict, ref: np.ndarray, subsample: float = None) -> Dict:
    indices = np.argsort(ref)
    if subsample is not None:
        indices = np.random.choice(
            indices, replace=False,
            size=int(len(indices) * subsample))

    return {_k: _v[indices] for _k, _v in data.items()}


def _subsample(data: Dict, ref: np.ndarray, subsample: float = None) -> Dict:
    if subsample is not None:
        indices = np.random.choice(
            range(len(ref)), replace=False,
            size=int(len(ref) * subsample))

    return {_k: _v[indices] for _k, _v in data.items()}


# ########### MAIN FUNCTIONS ###########

def data(
    points: Dict, 
    bounds=None, 
    intervals=None, 
    ax=None, 
    **kwargs) -> Any:
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
    fading_with_lead_time: bool = False,
    subsample: float = None,
    **kwargs
) -> Any:
    
    if subsample is not None:
        _subset = _subsample(
            {'test': y_test, 'pred': y_pred, 
            'low': lower_bound, 'up': upper_bound}, 
            y_test, subsample)
        y_test, y_pred = _subset['test'], _subset['pred']
        lower_bound, upper_bound = _subset['low'], _subset['up']

    if ax is None:
        _, ax = plt.subplots()

    _outside = (y_test > upper_bound) + (y_test < lower_bound)
    error = np.zeros((2, len(y_pred)))
    error[0, :] = np.abs(y_pred - lower_bound)
    error[1, :] = np.abs(upper_bound - y_pred)

    if fading_with_lead_time:
        alphas: np.ndarray = (np.logspace(0, 1, num=len(y_pred), base=2) - 1)[::-1]
    else:
        alphas: np.ndarray = np.ones(len(y_pred))
    out_legend_printed, in_legend_printed = False, False
    
    for i, alpha in enumerate(alphas):
        in_kwargs = {} if in_legend_printed else {'label': "Inside PI"}
        out_kwargs = {} if out_legend_printed else {'label': "Outside PI"}
        true_kwargs = {} if out_legend_printed else {'label': "True value"}

        if not _outside[i]:
            ax.errorbar(
                y_test[i],
                y_pred[i],
                yerr=error[:, i][:, np.newaxis],
                capsize=5, marker="o", elinewidth=2, linewidth=0,
                color=C_STRONG,
                alpha=alpha,
                **in_kwargs,
                )
            in_legend_printed = True

        if _outside[i]:
            ax.errorbar(
                y_test[i],
                y_pred[i],
                yerr=error[:, i][:, np.newaxis],
                capsize=5, marker="o", elinewidth=2, linewidth=0, 
                color=C_LIGHT,
                alpha=alpha,
                **out_kwargs,
                )
            ax.scatter(
                y_test[i],
                y_test[i],
                marker="*", color="black",
                alpha=alpha,
                **true_kwargs,
                )
            out_legend_printed = True

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
        xy=kwargs.get('xy', (0.,0.)),
        xytext=kwargs.get('xytext', (np.min(y_test) * 1.175, np.max(y_pred) * 0.72)), 
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
    ax=None, **kwargs) -> Any:
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
    hsic_coeff: float = None,
    num_bins: int = 3, 
    ax=None, **kwargs) -> Any:

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
        bottom=True, top=False, 
        labelbottom=False)
    
    _annotation_text: str = f"SSC score: {np.round(cond_coverage, 4)}"
    if hsic_coeff is not None:
        _annotation_text += f"\nHSIC coefficient: {np.round(hsic_coeff, 4)}"

    ax.annotate(
        _annotation_text,
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
                     ax=None, **kwargs) -> Any:
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


def dataframe_to_png(df, filename) -> None:
    _ = render_mpl_table(df, header_columns=0, col_width=2.0)
    plt.savefig(filename)


# ######## COVERAGE SOUGTH vs. ALPHA for a given strategy ########

def coverage_by_alpha(
    coverages_arr: np.ndarray,
    miscoverages_list: List[float],
    strategy_name: str, ax=None, 
    **kwargs) -> Any:

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
    
    ax.set_title(strategy_name)

    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/coverage-by-alpha.png'), dpi=200)
        plt.show()
        plt.close()
        return

    return ax


# ######## TIME SERIES DATA ########

# PREDICTED VALUES & INTERVALS

def ts(
        points, 
        intervals=None, 
        ax=None,
        **kwargs
        ) -> Any:
    
    if ax is None:
        _, ax = plt.subplots()
    
    ax.set_ylabel("Hourly demand (GW)")
    ax.plot(
        points['X_train'][len(points['X_train']) * 3 // 4:], 
        points['y_train'][len(points['X_train']) * 3 // 4:], 
        lw=2, label="Training data", color=C_STRONG)
    ax.plot(points['X'], points['y_pred'], lw=1.5, label="Predictions", color=C_PRED)
    ax.plot(points['X'], points['y'], lw=0.75, label="Test data", color=C_LIGHT)
    ax.fill_between(
        intervals['X'],
        intervals['y_low'],
        intervals['y_up'],
        color=C_PRED,
        alpha=0.5,
        label="Prediction intervals",
    )
    ax.legend()
    ax.set_xlabel(kwargs.get('xlabel', "Date"))
    ax.set_ylabel(kwargs.get('ylabel', "Hourly demand (GW)"))

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if ax is None:
        plt.savefig(kwargs.get('save_path', 'output/data.png'), dpi=200)
        plt.show()
        plt.close()
        return
    return ax


def rolling_coverage(
        rolling_coverage: dict, 
        x_values: pd.Index,
        window_size: int, 
        **kwargs) -> None:
    plt.figure(figsize=(15, 5))
    plt.xlabel(kwargs.get('xlabel', "Date"))
    plt.ylabel(kwargs.get('ylabel', f"Rolling coverage [{window_size} hours]"))
    
    plt.plot(x_values, rolling_coverage['EnbPI_nP'], label="Without update of residuals", color=C_LIGHT)
    plt.plot(x_values, rolling_coverage['EnbPI'], label="With update of residuals", color=C_STRONG)
    plt.legend()

    if 'title' in kwargs:
        plt.title(kwargs['title'])
        
    plt.savefig(kwargs.get('save_path', 'output/rolling-coverage.png'), dpi=200)
    plt.show();
    plt.close();
