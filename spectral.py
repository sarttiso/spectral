import numpy as np
from scipy.signal import windows

"""
multi-taper spectral density estimation using DPSS sequences as described by Percival and Walden.

IN:
y: time series of evenly spaced observations in time
dt: sample spacing in sample coordinate
nw: (default 4) time-halfbandwidth product

OUT:
S_est: estimate for the power spectral density estimation
f: frequency axis

TO DO:
- allow user to specify which/how many frequencies to get
"""
def multitaper(y, dt, nw=4):
    N = len(y)
    W = nw/N
    K = int(2*nw-1)
    # frequencies
    f = freq(N, dt)
    Nf = len(f)
    # time indices
    t = np.arange(N)
    # get discrete prolate spheroidal sequence (dpss) windows
    wins = windows.dpss(N, nw, Kmax=K)
    # get tapered spectra
    Sk = np.zeros((Nf, K))
    for ii in range(K):
        # loop over frequencies
        for ff in range(Nf):
            # compute spectral density estimate
            Sk[ff, ii] = dt*np.abs(np.sum(wins[ii,:]*y*np.exp(-1j*2*np.pi*f[ff]*t*dt)))**2

    # get eigenvalues for N, W
    evals = dpss_evals(N, W)
    # implement adaptive multitaper spectral estimator (Percival and Walden, pg. 370)
    # start with eqn 369a (i.e. no estimate for weights bk)
    K_cur = 1
    S_est = np.sum(np.tile(evals[0:K_cur], (Nf,1))*Sk[:, 0:K_cur], axis=1)/np.sum(evals[0:K_cur])
    bk = np.zeros((Nf, K))
    # make convenient tiled version of eigenvalues
    evals_tile = np.tile(evals[0:K], (Nf,1))
    # iterate over equations 368a and 370a
    for ii in range(5):
        # get weights
        bk = np.tile(S_est, (K, 1)).T/(evals_tile*np.tile(S_est, (K, 1)).T + (1-evals_tile)*np.var(y))
        # update spectrum
        S_est = np.sum(bk**2*evals_tile*Sk[:, 0:K], axis=1)/np.sum(bk**2*evals_tile, axis=1)

    return S_est, f

# get dpss eigenvalues for given N, W
def dpss_evals(N, W):
    t = np.arange(N)
    t1, t2 = np.meshgrid(t, t)
    dt = t1-t2
    dt[np.diag_indices(N)] = 1
    # construct matrix
    A = np.sin(2*np.pi*W*dt)/(np.pi*dt)
    # set diagonal manually (l'hopital)
    A[np.diag_indices(N)] = 2*W
    # compute eigenvalues (should all be real)
    evals = np.real(np.linalg.eig(A)[0])
    # sort by magnitude
    evals[::-1].sort()
    return evals

# generate one-sided frequency axis for given N data and dt sample spacing
def freq(N, dt):
    fs = 1/dt
    fi = fs/N
    fx = np.arange(0, fs/2+fi, fi)
    return fx
