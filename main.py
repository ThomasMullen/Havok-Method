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

import pysindy as ps
from sklearn import linear_model

from utils import *
from bout_import import load_filepaths, print_bout_keys

if __name__ == "__main__":

    x = load_filepaths(specific_bout=8, bout_spacing=100, specific_segment=6)

    # visualise - lorenz dynamics
    # fig = plt.figure()
    # axes = plt.axes()
    # camera = Camera(fig)
    # for i in range(1, len(x), 25):
    #     axes.plot(x[:i], lw=0.1, color='blue')
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
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(sv) + 1), sv, s=5)
    ax.axhline(tau, c='r')
    ax.set_xlim([0.9, r + 3])
    ax.set_title("rank: " + r.astype(str))

    # set truncation
    Utilde = U[:, :r]
    Stilde = S[:r, :r]
    Vtilde = Vh.conj().T[:, :r]

    # plot embedded attractor
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # camera = Camera(fig)
    #
    # for i in range(1, len(x), 25):
    #     ax.plot(Vtilde.T[0][:i], Vtilde.T[1][:i], Vtilde.T[5][:i], lw=0.1, color='blue')
    #     camera.snap()
    #
    # animation = camera.animate()
    # animation.save('SLC_space.gif', writer='imagemagick')


    # plot predicted projection
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.grid(False)
    ax.axis('off')
    ax.plot(Vtilde.T[0], Vtilde.T[1], Vtilde.T[4], lw=0.1)


    # Calculate time derivatives of eigen-timeseries
    filter_len = 7
    dVtilde_dt = np.r_[[robust_differentiator(Vtilde[:, i], dt=1 / (100 * 100), filter_length=filter_len) for i in
                        range(np.min(Vtilde.shape))]].T

    # reshape eigen-timeseries
    Vtilde = Vtilde[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]
    dVtilde_dt = dVtilde_dt[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]

    # plot eigen-timeseries
    plot_time_series(Vtilde, r=r, threshold=15)
    # plot eigen-timeseries time derivatives
    plot_time_series(dVtilde_dt, r=r, threshold=15)
    # plot eigen-spatial mode
    plot_time_series(Utilde, r=r, threshold=15)

    # component_score = rank_forcing_component(Vtilde)
    # print(component_score)
    #
    # fig, axes = plt.subplots(np.sum(component_score.T[-1] >= 0))
    # for i, ax in zip(component_score.T[0][component_score.T[-1] >= 0].astype(int), axes):
    #     real_x, real_y = calc_probability_distribution(Vtilde[:, i], n_bins=10)
    #     gauss_x, gauss_y = fit_gauss(Vtilde[:, i], n_bins=10)
    #
    #     ax.plot(real_x, real_y, c='r', label="raw")
    #     ax.plot(gauss_x, Gauss(gauss_x, *gauss_y), 'k--', label='gauss fit')
    #     ax.set_yscale('log')
    #     ax.set_ylim([10 ** -5, 1])
    #     ax.legend()

    # mark regions along timeseries with intermittent forcing
    real_x, real_y = calc_probability_distribution(Vtilde[:, -1], n_bins=10)
    counts, edges = np.histogram(Vtilde[:, -1], bins=10)
    gauss_x, gauss_y = fit_gauss(Vtilde[:, -1], n_bins=10)
    idx_intercepts = np.argsort(np.abs(real_y - Gauss(real_x, *gauss_y)))[1:3]
    upper_bound, lower_bound = np.max(edges[idx_intercepts]), np.min(edges[idx_intercepts])

    # create masks for extreme values
    extremes = np.where((Vtilde[:, -1] < lower_bound) | (Vtilde[:, -1] > upper_bound))
    mx = np.ma.masked_array(Vtilde[:, 0], mask=[(Vtilde[:, -1] > lower_bound) & (Vtilde[:, -1] < upper_bound)])

    # plot regions
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(Vtilde[:, 0], alpha=0.6, lw=1, label=str(1))
    ax1.plot(mx, alpha=1, lw=1, label='xtr', c='r')
    # ax1.scatter(extremes, np.repeat(0, len(extremes[0])), s=2, c='r', label='crit')
    ax1.legend()

    ax2.plot(Vtilde[:, -1], alpha=0.6, lw=1, label='r')
    ax2.scatter(extremes, np.repeat(0, len(extremes[0])), s=2, c='r', label='crit')
    ax2.legend()

    # fit Koopman operator
    # model = ps.SINDy()
    # model.fit(Vtilde[:1000, :-1], t=.01)
    # model.print()

    # linear regression:
    reg = linear_model.LinearRegression()
    reg.fit(Vtilde[100:600, :-1], dVtilde_dt[100:600, :-1])
    A = reg.coef_
    fig, ax = plt.subplots()
    ax = sns.heatmap(A, center=0)

    dVtilde_dt_pred = reg.predict(Vtilde[600:, :-1])

    fig, ax = plt.subplots()
    ax.plot(dVtilde_dt_pred[:, 0], alpha=0.6, c='k', linestyle='--', label="pred")
    ax.plot(dVtilde_dt[500:, 0], alpha=0.6, lw=1, label="true")
    ax.legend()

    # runge-kutta integration
    y = first_order_kutta_runge(dx_dt=dVtilde_dt_pred, x=Vtilde, starting_point=600)

    # plot the predicted 1st component
    fig, ax = plt.subplots()
    ax.plot(y.T[0], alpha=0.6, lw=1, label="pred")
    ax.legend()

    # plot predicted projection
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection="3d")
    # ax.grid(False)
    # ax.axis('off')
    # camera = Camera(fig)
    #
    # for a, b, g in zip(np.linspace(0, 2 * np.pi, 360, endpoint=False),
    #                    np.linspace(0, 2 * np.pi, 360, endpoint=False),
    #                    np.linspace(0, 2 * np.pi, 360, endpoint=False)):
    #     true = np.asarray(Rx(a) @ Ry(2 * b * a) @ Rx(g * g) @ Vtilde.T[:3])
    #     pred = np.asarray(Rx(a) @ Ry(2 * b * a) @ Rx(g * g) @ (0.01 * y.T[:3]))
    #     ax.plot(true[0], true[1], true[2], lw=0.1, c='b')
    #     ax.plot(pred[0], pred[1], pred[2], lw=0.1, c='orange')
    #     camera.snap()

    # animation = camera.animate()
