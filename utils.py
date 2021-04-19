import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from scipy.integrate import odeint
from scipy.linalg import hankel, svd, svdvals
from scipy.signal import savgol_filter, lsim, StateSpace
from scipy.special import binom
from scipy.optimize import curve_fit
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera


def generate_hankel_matrix(timeseries, time_embedding):
    if timeseries.ndim == 1:
        H = hankel(timeseries[:time_embedding], timeseries[time_embedding:])

    else:
        H = np.vstack(
            [hankel(timeseries[i, :time_embedding], timeseries[i, time_embedding:]) for i in
             range(timeseries.shape[0])])
    return H


def svht(X, sv=None):
    # svht for sigma unknown
    m, n = sorted(X.shape)  # ensures m <= n
    beta = m / n  # ratio between 0 and 1
    if sv is None:
        sv = svdvals(X)
    sv = np.squeeze(sv)
    omega_approx = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return np.median(sv) * omega_approx


def robust_differentiator(x, dt=1 / 700, filter_length=71):
    if not filter_length % 2 == 1:
        raise ValueError('Filter length must be odd.')
    M = int((filter_length - 1) / 2)
    m = int((filter_length - 3) / 2)
    coefs = [(1 / 2 ** (2 * m + 1)) * (binom(2 * m, m - k + 1) - binom(2 * m, m - k - 1))
             for k in range(1, M + 1)]
    coefs = np.array(coefs)
    kernel = np.concatenate((coefs[::-1], [0], -coefs))
    filtered = np.convolve(kernel, x, mode='valid')
    filtered = (1 / dt) * filtered
    filtered = np.concatenate((np.nan * np.ones(M), filtered, np.nan * np.ones(M)))
    return filtered