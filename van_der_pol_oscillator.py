import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq

from utils import *


def van_der_pol(current_state, t):
    # define the system parameters sigma, rho, and beta
    # mu = 0.05
    mu = 0.8
    # positions of x, y, z in space at the current time point
    x, y = current_state

    # define the 3 ordinary differential equations known as the lorenz equations
    dx_dt = y
    dy_dt = -x -mu * (x**2 - 1) * y

    # return a list of the equations that describe the system
    return [dx_dt, dy_dt]


def generate_time_steps(start_time=0, end_time=100, dt=100):
    time_points = np.linspace(start_time, end_time, end_time * dt)
    return time_points


if __name__ == "__main__":
    # generate data
    dt = 1/10
    initial_state = [0, 0.001]
    xy = odeint(van_der_pol, initial_state, generate_time_steps(0, 1000, int(1/dt)))
    x = xy[:, 0]
    y = xy[:, 1]

    # plot system
    fig, ax = plt.subplots()
    ax.plot(x, y, lw=0.5)

    # Generate hankel matrix with 20 time embedding
    H = generate_hankel_matrix(x, time_embedding=100)

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

    # plot embedding
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(Vtilde.T[0], Vtilde.T[1], Vtilde.T[2], lw=0.2, color='blue')

    # Calculate time derivatives of eigen-timeseries
    filter_len = 7
    dVtilde_dt = np.r_[[robust_differentiator(Vtilde[:, i], dt=1 / (100 * 100), filter_length=filter_len) for i in
                        range(np.min(Vtilde.shape))]].T

    # reshape eigen-timeseries
    Vtilde = Vtilde[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]
    dVtilde_dt = dVtilde_dt[int((filter_len - 1) / 2):-int((filter_len - 1) / 2), :]

    # plot eigen-timeseries
    plot_time_series(Vtilde, r=12, threshold=15)
    # plot eigen-timeseries time derivatives
    plot_time_series(dVtilde_dt, r=12, threshold=15)
    # plot eigen-spatial mode
    plot_time_series(Utilde, r=12, threshold=15)

    # Number of sample points
    N = int(1000 * 1/dt)
    # sample spacing
    T = dt

    yf = fft(Vtilde.T[0])
    xf = fftfreq(N, T)[:N // 2]

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    ax.grid()

    # fit regression
    A = np.dot(Vtilde[:-1].T, LA.pinv(Vtilde[1:].T))
    fig, ax = plt.subplots()
    sns.heatmap(A, ax=ax)

    # plot prediction
    fig, ax = plt.subplots()
    ax.plot(np.dot(A, Vtilde.T)[0])
    ax.plot(Vtilde[1:].T[0])

    # calculate the forcing term
    F = Vtilde[1:].T - np.dot(A, Vtilde[:-1].T)