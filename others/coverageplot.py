# #################### IMPORTS ####################
import os.path
import random
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# #################### PLOT OPTIONS ####################

colors_blindness = sns.color_palette("colorblind")
# color_test = colors_blindness[3]  # 4  # strong one
# color_train = colors_blindness[1]  # 1  # medium one
# color_cal = colors_blindness[-2]  # 9  # light one
color_test = '#800000'
color_train = '#9B0A0A'
color_cal = '#F2BFBF'

size: int = 19
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': 'Times',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': size,
    'axes.labelsize': size,
    'axes.titlesize': size,
    'figure.titlesize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'legend.fontsize': size,
    'axes.unicode_minus': False,
})

OUTPUT_PATH = os.path.join('output', 'coverage')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ################### DATA #####################

# FEATURES

s = 11
random.seed(s)
np.random.seed(s)
n = 300
X = np.random.uniform(low=0,high=5,size=n)
eps = np.random.normal(size=n)
alpha: float = 0.1

# LABEL

Y = (1-np.cos(X))*eps
X_f = np.cos(X)

# SPLIT

idx = np.random.permutation(n)
n_half = int(np.floor(n/3))
idx_train, idx_cal, idx_test = idx[:n_half], idx[n_half:2*n_half], idx[2*n_half:]

# ################### MODEL #####################

data = pd.DataFrame(
    data=[X_f[idx_train], Y[idx_train]],
    index=["x", "y"]).T
mod = smf.quantreg('y ~ x', data)
res_sup = mod.fit(q=1-alpha/2)
res_inf = mod.fit(q=alpha/2)

# ################### PREDICTIONS #####################

y_sup_train = res_sup.predict({'x': X_f[idx_train]})
y_sup_cal = res_sup.predict({'x': X_f[idx_cal]})
y_sup_test = res_sup.predict({'x': X_f[idx_test]})
y_inf_train = res_inf.predict({'x': X_f[idx_train]})
y_inf_cal = res_inf.predict({'x': X_f[idx_cal]})
y_inf_test = res_inf.predict({'x': X_f[idx_test]})

# ################### PLOTS #####################

# NO coverage

res_cal = np.maximum(Y[idx_cal]-y_sup_cal.values, y_inf_cal.values-Y[idx_cal])
q = np.quantile(res_cal,(1-alpha)*(1+1/len(idx_test)))
plt.scatter(X[idx_test],Y[idx_test],marker='.',color=color_test,zorder=2)
plt.plot(X[idx_test][np.argsort(X[idx_test])],0.5*y_inf_test[np.argsort(X[idx_test])],'--',color=color_train)
plt.plot(X[idx_test][np.argsort(X[idx_test])],0.5*y_sup_test[np.argsort(X[idx_test])],'--',color=color_train)
plt.fill_between(X[idx_test][np.argsort(X[idx_test])],
                 0.5*y_sup_test[np.argsort(X[idx_test])],
                 0.5*y_inf_test[np.argsort(X[idx_test])],
                 fc=color_cal,alpha=.3)
plt.ylim(min(Y)-0.3,max(Y)+0.3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.savefig(os.path.join(OUTPUT_PATH, 'no-coverage.png'),
            bbox_inches='tight', dpi=300)
plt.show()

# MARGINAL coverage
# # (mean-based SCP with absolute values residuals as scores·)

reg = LinearRegression()
reg.fit(X_f[idx_train].reshape(-1, 1), Y[idx_train])
y_pred_cal = reg.predict(X_f[idx_cal].reshape(-1, 1))
y_pred_train = reg.predict(X_f[idx_train].reshape(-1, 1))
res_cal = Y[idx_cal]-y_pred_cal
y_pred_test = reg.predict(X_f[idx_test].reshape(-1, 1))

q = np.quantile(np.abs(res_cal),(1-alpha)*(1+1/len(idx_test)))
plt.scatter(X[idx_test],Y[idx_test],marker='.',color=color_test,zorder=2, s=50)
# plt.plot(X[idx_test][np.argsort(X[idx_test])],y_pred_test[np.argsort(X[idx_test])], color=color_train)
plt.plot(X[idx_test][np.argsort(X[idx_test])],
         y_pred_test[np.argsort(X[idx_test])]+q,
         '--', color=color_train)
plt.plot(X[idx_test][np.argsort(X[idx_test])],
         y_pred_test[np.argsort(X[idx_test])]-q,
         '--', color=color_train)
plt.fill_between(X[idx_test][np.argsort(X[idx_test])],
                 y_pred_test[np.argsort(X[idx_test])]+q,
                 y_pred_test[np.argsort(X[idx_test])]-q,
                 fc=color_cal, alpha=.3)
plt.ylim(min(Y)-0.3, max(Y)+0.3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.savefig(os.path.join(OUTPUT_PATH, 'marginal-coverage.png'),
            bbox_inches='tight', dpi=300)
plt.show()

# (perfect) CONDITIONAL coverage

perfect_quantile_inf = (1 - np.cos(X[idx_test][np.argsort(X[idx_test])])) * norm.ppf(alpha/2)
perfect_quantile_sup = (1 - np.cos(X[idx_test][np.argsort(X[idx_test])])) * norm.ppf(1-alpha/2)

plt.scatter(X[idx_test],Y[idx_test],marker='.',color=color_test,zorder=2)
plt.plot(X[idx_test][np.argsort(X[idx_test])],perfect_quantile_inf, '--',color=color_train)
plt.plot(X[idx_test][np.argsort(X[idx_test])],perfect_quantile_sup, '--',color=color_train)
plt.fill_between(X[idx_test][np.argsort(X[idx_test])],
                 perfect_quantile_inf,
                 perfect_quantile_sup,
                 fc=color_cal, alpha=.3)

plt.ylim(min(Y)-0.3, max(Y)+0.3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.savefig(os.path.join(OUTPUT_PATH, 'conditional-coverage.png'),
            bbox_inches='tight', dpi=300)
plt.show()

