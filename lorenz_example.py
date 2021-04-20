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

from utils import *


def lorenz_system(current_state, t):
    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho = 28.
    beta = 8. / 3.

    # positions of x, y, z in space at the current time point
    x, y, z = current_state

    # define the 3 ordinary differential equations known as the lorenz equations
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    # return a list of the equations that describe the system
    return [dx_dt, dy_dt, dz_dt]


def generate_time_steps(start_time=0, end_time=100, dt=100):
    time_points = np.linspace(start_time, end_time, end_time * dt)
    return time_points


def plot_time_series(timeseries, threshold=15):
    if timeseries.shape[0] < timeseries.shape[1]:
        timeseries = timeseries.T
    if r > threshold:
        fig, axes = plt.subplots(threshold, 1, figsize=(10, 10))
        for i, ax in zip(range(threshold-1), axes):
            ax.plot(timeseries[:, i], alpha=0.6, lw=1, label=str(i))
        axes[int(threshold-1)].plot(timeseries[:, -1], alpha=0.6, lw=1, label='r')
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
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

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
            gauss_x, gauss_y = fit_gauss(Vtilde[:, i], n_bins=10)
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

        print("V"+str(i)+"<", np.min(real_x[idx_intercepts]), "V"+str(i)+">", np.max(real_x[idx_intercepts]))
        proportion_extreme_vals = np.sum([(timeseries[:, i] < np.min(real_x[idx_intercepts])) |
                                          (timeseries[:, i] > np.max(real_x[idx_intercepts]))]
                                         )/timeseries.shape[0]
        vals[i] = [i, proportion_extreme_vals]
    return vals


if __name__ == '__main__':
    # generate data
    initial_state = [-8, 8, 27]
    xyz = odeint(lorenz_system, initial_state, generate_time_steps(0, 100, 100))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # # visualise - lorenz dynamics
    # fig = plt.figure()
    # axes = plt.axes(projection="3d")
    # camera = Camera(fig)
    # for i in range(1, len(x), 25):
    #     axes.plot(x[:i], y[:i], z[:i], lw=0.1, color='blue')
    #     camera.snap()
    # animation = camera.animate()
    # # animation.save('lorenz.gif', writer = 'imagemagick')

    # Generate hankel matrix with 20 time embedding
    H = generate_hankel_matrix(x, time_embedding=35)

    # Apply SVD
    U, sv, Vh = LA.svd(H, full_matrices=0)
    S = np.diag(sv)

    # determine truncation
    tau = svht(H, sv=sv)
    r = sum(sv > tau)
    print("rank:", r)
    # fig, ax = plt.subplots()
    # ax.scatter(range(1, len(sv) + 1), sv, s=5)
    # ax.axhline(tau, c='r')
    # ax.set_xlim([0.9, r + 3])
    # ax.set_title("rank: " + r.astype(str))

    # set truncation
    Utilde = U[:, :r]
    Stilde = S[:r, :r]
    Vtilde = Vh.conj().T[:, :r]

    # # plot embedded attractor
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # camera = Camera(fig)
    #
    # for i in range(1, len(x), 25):
    #     ax.plot(Vtilde.T[0][:i], Vtilde.T[1][:i], Vtilde.T[2][:i], lw=0.2, color='blue')
    #     camera.snap()
    #
    # animation = camera.animate()
    # # animation.save('embedded_lorenz.gif', writer = 'imagemagick')

    # Calculate time derivatives of eigen-timeseries
    filter_len = 7
    dVtilde_dt = np.r_[[robust_differentiator(Vtilde[:, i], dt=1 / (100 * 100), filter_length=filter_len) for i in
                        range(np.min(Vtilde.shape))]].T

    # reshape eigen-timeseries
    Vtilde = Vtilde[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]
    dVtilde_dt = dVtilde_dt[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]

    # # plot eigen-timeseries
    # plot_time_series(Vtilde, threshold=15)
    # # plot eigen-timeseries time derivatives
    # plot_time_series(dVtilde_dt, threshold=15)
    # # plot eigen-spatial mode
    # plot_time_series(Utilde, threshold=15)

    component_score = rank_forcing_component(Vtilde)
    print(component_score)

    fig, axes = plt.subplots(np.sum(component_score.T[-1]>=0))
    for i, ax in zip(component_score.T[0][component_score.T[-1]>=0].astype(int), axes):
        real_x, real_y = calc_probability_distribution(Vtilde[:, i], n_bins=10)
        gauss_x, gauss_y = fit_gauss(Vtilde[:, i], n_bins=10)

        ax.plot(real_x, real_y, c='r', label="raw")
        ax.plot(gauss_x, Gauss(gauss_x, *gauss_y), 'k--', label='gauss fit')
        ax.set_yscale('log')
        ax.set_ylim([10 ** -5, 1])
        ax.legend()

    # mark regions along timeseries with intermittent forcing
    counts, edges = np.histogram(Vtilde[:, -1], bins=10)
    gauss_x, gauss_y = fit_gauss(Vtilde[:, -1], n_bins=10)
    idx_intercepts = np.argsort(np.abs(real_y - Gauss(real_x, *gauss_y)))[1:3]
    upper_bound, lower_bound = np.max(edges[idx_intercepts]), np.min(edges[idx_intercepts])

    # create masks for extreme values
    extremes = np.where((Vtilde[:, -1] < lower_bound) | (Vtilde[:, -1] > upper_bound))
    mx = np.ma.masked_array(Vtilde[:, 0], mask=[(Vtilde[:, -1] < lower_bound) | (Vtilde[:, -1] > upper_bound)])

    # plot regions
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(Vtilde[:, 0], alpha=0.6, lw=1, label=str(1))
    ax1.plot(mx, alpha=1, lw=1, label='xtr', c='r')
    # ax1.scatter(extremes, np.repeat(0, len(extremes[0])), s=2, c='r', label='crit')
    ax1.legend()

    ax2.plot(Vtilde[:, -1], alpha=0.6, lw=1, label='r')
    ax2.scatter(extremes, np.repeat(0, len(extremes[0])), s=2, c='r', label='crit')
    ax2.legend()

