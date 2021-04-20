import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, svdvals
from scipy.special import binom
from scipy.optimize import curve_fit


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


def first_order_kutta_runge(dx_dt, x, starting_point):
    # runge-kutta integration
    n = x[starting_point:].shape[0]
    y0 = x[starting_point - 1][:-1]
    y = np.zeros((n, len(y0)))
    dt = 0.01
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + (dt) * dx_dt[i]
    return y


def plot_time_series(timeseries, r, threshold=15):
    if timeseries.shape[0] < timeseries.shape[1]:
        timeseries = timeseries.T
    if r > threshold:
        fig, axes = plt.subplots(threshold, 1, figsize=(10, 10), sharex=True)
        for i, ax in zip(range(threshold - 1), axes):
            ax.plot(timeseries[:, i], alpha=0.6, lw=1, label=str(i))
        axes[int(threshold - 1)].plot(timeseries[:, -1], alpha=0.6, lw=1, label='r')
    else:
        fig, axes = plt.subplots(r, 1, figsize=(10, 10))
        for i, ax in enumerate(axes):
            ax.plot(timeseries[:, i], alpha=0.6, lw=1, label=str(i))
    return


def calc_probability_distribution(timeseries, n_bins=10):
    counts, edges = np.histogram(timeseries, bins=n_bins)
    xpoints = edges[1:] + ((edges[1:] - edges[0:-1]) / 2)
    prob = counts / np.sum(counts)
    return xpoints, prob


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_gauss(timeseries, n_bins):
    xpoints, prob = calc_probability_distribution(timeseries, n_bins)

    # seed initial gaussian parameters
    mean = sum(xpoints * prob) / sum(prob)
    sigma = np.sqrt(sum(prob * (xpoints - mean) ** 2) / sum(prob))

    # optimise the coefficients
    popt, pcov = curve_fit(Gauss, xpoints, prob, p0=[max(prob), mean, sigma])
    xpt = np.linspace(xpoints[0], xpoints[-1], 10000)

    return xpt, popt


def rank_forcing_component(timeseries):
    if timeseries.shape[0] < timeseries.shape[1]:
        timeseries = timeseries.T

    vals = np.zeros((timeseries.shape[-1], 2))
    for i in range(timeseries.shape[-1]):
        real_x, real_y = calc_probability_distribution(timeseries[:, i], n_bins=10)
        try:
            gauss_x, gauss_y = fit_gauss(timeseries[:, i], n_bins=10)
            idx_intercepts = np.argsort(np.abs(real_y - Gauss(real_x, *gauss_y)))[1:3]
        except:
            # fail to fit gaussian
            print("failed with component", i)
            # label with -1
            vals[i] = [i, -1]
            continue

        # label timeseries which lack extreme tails
        if np.std(real_y - Gauss(real_x, *gauss_y)) > 0.001:
            vals[i] = [i, 0]
            continue

        print("V" + str(i) + "<", np.min(real_x[idx_intercepts]), "V" + str(i) + ">", np.max(real_x[idx_intercepts]))
        proportion_extreme_vals = np.sum([(timeseries[:, i] < np.min(real_x[idx_intercepts])) |
                                          (timeseries[:, i] > np.max(real_x[idx_intercepts]))]
                                         ) / timeseries.shape[0]
        vals[i] = [i, proportion_extreme_vals]
    return vals


# rotations
def Rz(gamma):
    return np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                      [np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])


def Ry(beta):
    return np.matrix([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])


def Rx(alpha):
    return np.matrix([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])