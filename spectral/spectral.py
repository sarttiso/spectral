import numpy as np
from scipy.signal import windows
from scipy import stats
import tqdm as tqdm


def multitaper(y, dt, nw=4):
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
    N = len(y)
    W = nw / N
    K = int(2 * nw - 1)
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
            Sk[ff, ii] = (dt * np.abs(
                np.sum(wins[ii, :] * y *
                       np.exp(-1j * 2 * np.pi * f[ff] * t * dt)))**2)

    # get eigenvalues for N, W
    evals = dpss_evals(N, W)
    # implement adaptive multitaper spectral estimator (Percival and Walden, pg. 370)
    # start with eqn 369a (i.e. no estimate for weights bk)
    K_cur = 1
    S_est = np.sum(np.tile(evals[0:K_cur], (Nf, 1)) * Sk[:, 0:K_cur],
                   axis=1) / np.sum(evals[0:K_cur])
    bk = np.zeros((Nf, K))
    # make convenient tiled version of eigenvalues
    evals_tile = np.tile(evals[0:K], (Nf, 1))
    # iterate over equations 368a and 370a
    for ii in range(5):
        # get weights
        bk = np.tile(S_est, (K, 1)).T / (evals_tile * np.tile(S_est,
                                                              (K, 1)).T +
                                         (1 - evals_tile) * np.var(y))
        # update spectrum
        S_est = np.sum(bk**2 * evals_tile * Sk[:, 0:K], axis=1) / np.sum(
            bk**2 * evals_tile, axis=1)

    return S_est, f


def dpss_evals(N, W):
    """compute eigenvalues of DPSS sequence with time halfbandwidth product NW

    :param N: [description]
    :type N: [type]
    :param W: [description]
    :type W: [type]
    :return: [description]
    :rtype: [type]
    """
    t = np.arange(N)
    t1, t2 = np.meshgrid(t, t)
    dt = t1 - t2
    dt[np.diag_indices(N)] = 1
    # construct matrix
    A = np.sin(2 * np.pi * W * dt) / (np.pi * dt)
    # set diagonal manually (l'hopital)
    A[np.diag_indices(N)] = 2 * W
    # compute eigenvalues (should all be real)
    evals = np.real(np.linalg.eig(A)[0])
    # sort by magnitude
    evals[::-1].sort()
    return evals


# generate one-sided frequency axis for given N data and dt sample spacing
def freq(N, dt):
    fs = 1 / dt
    fi = fs / N
    fx = np.arange(0, fs / 2 + fi, fi)
    return fx


def prewhiten(X, phi):
    """This function prewhitens a time series by applying a prewhitening filter that
    generates a new time series whose spectrum more closely matches that of white noise.
    See Percival and Walden 6.5 and 9.10.

    Args:
        X ([type]): [description]
        phi ([type]): [description]
    """
    n = len(X)

    ncoeff = len(phi)

    # with AR(p) parameters, subtract weighted observations from ts to generate ws. The
    # prewhitened time series is et(p) as on pg. 438 of Percival and Walden
    e = np.zeros(n - ncoeff)
    for ii in range(ncoeff, n):
        e[ii - ncoeff] = X[ii] - np.sum(phi[::-1] * X[ii - ncoeff:ii])


def acvs(X, k, taper=[], biased=True):
    """autocovariance sequence of time series X at desired lags k

    can be improved by using Fourier transform (i.e., exercise 6.4 in Percival and Walden)

    Args:
        X (1d array like): time series, assumed to be second order stationary
        k (1d array like): lag(s)
        taper (1d array like): taper to apply
        biased (boolean): whether to use biased or unbiased estimator

    Returns:
        1d array like: acvs at lags
    """
    X = np.asarray(X).reshape(-1)
    k = np.asarray(k).reshape(-1)

    n = len(X)
    nk = len(k)
    # subtract mean
    mu = np.mean(X)
    X = X - mu

    assert np.max(k) < n, 'lag is greater than length of time series'

    s = np.zeros(nk)

    if len(taper) == 0:
        if biased:
            for ii, kk in enumerate(k):
                s[ii] = 1 / n * np.sum(X[slice(0, n - kk)] * X[kk:])
        else:
            for ii, kk in enumerate(k):
                s[ii] = 1 / (n - kk) * np.sum(X[slice(0, n - kk)] * X[kk:])
    else:
        assert len(taper) == n, 'taper must be same length as data'
        if biased:
            for ii, kk in enumerate(k):
                lidx = slice(0, n-kk)
                ridx = slice(kk, n+1)
                s[ii] = 1 / n * np.sum(taper[lidx]*X[lidx]*taper[ridx]*X[ridx])
        else:
            for ii, kk in enumerate(k):
                lidx = slice(0, n-kk)
                ridx = slice(kk, n+1)
                s[ii] = 1 / (n-kk) * np.sum(taper[lidx]*X[lidx]*taper[ridx]*X[ridx])

        s = s*np.var(X)/np.var(X*taper)
            
    return s


def yulewalker(Y, p, taper=[]):
    """Yule-Walker estimation of AR(p) coefficients.

    Args:
        Y ([type]): [description]
        p ([type]): [description]

    Returns:
        [type]: [description]
    """

    s = acvs(Y, np.arange(p + 1), biased=True, taper=taper)

    # populate Gamma
    Gamma = np.zeros((p, p))
    Gamma[np.diag_indices(p)] = s[0]
    for ii in range(p):
        Gamma[ii, ii + 1:p] = s[1:p - ii]
        Gamma[ii + 1:p, ii] = s[1:p - ii]

    # populate gamma
    gamma = s[1:].reshape(-1, 1)

    # solve for phi
    phi = np.matmul(np.linalg.inv(Gamma), gamma)

    # as per Eqn 395b
    sigma2 = s[0] - np.sum(phi * gamma)

    return phi, sigma2


# def levinson(Y, p, taper=[]):

#     s = acvs(Y, np.arange(p+1), taper=taper)
    # need to make 2d arrays!
#     phi = np.zeros(p)
#     sigma2 = np.zeros(p)

#     phi[0] = s[1]/s[0]
#     sigma2[0] = s[0]*(1-phi[0]**2)

    # FINISH
    # now iterate 
    # for kk in range(2, p+1):
    #     # first equation
    #     phi[kk-1] = s[kk]
    #     for jj in range(1, kk):
    #         phi[kk-1] = phi[kk-1] - phi[kk-1-jj)]*s[kk-jj]
    #     phi[kk-1] = phi[kk-1]/sigma2[kk-2]

    #     # second equation
    #     for jj in range(1, kk):
    #         phi[jj-1] = phi[jj-1]




# def ARfit(p, w, S_est, dt, S0=None, rho0=None, weights=[]):
#     """

#     """

def AR(phi, sig, n, scale=1.2):
    p = len(phi)
    w = stats.norm.rvs(loc=0, scale=sig, size=int(scale*n+p))
    X = np.zeros(len(w))
    X[0:p] = w[0:p]
    for ii in range(p, len(w)):
        # print(ii)
        X[ii] = w[ii]
        for jj, pp in enumerate(phi):
            X[ii] = X[ii] + pp*X[ii-jj-1]
    return X[-n:]


def ARpsd(sigma2, phi, dt, f, N=0):
    """
    The % output function will accept as input an innovations variance (S), a
    vector of lag coefficients (p), a vector of frequencies at which to
    evaluate the density (f), and the Nyquist frequency (fn)
    using modified definition from p. 392 of Percival and Walden:
    S(f) = 1/fn * sig^2/|1 - sum(p_j exp(-ij pi f/fn))|^2
    """
    nf = len(f)
    p = len(phi)
    num = sigma2 * dt
    den = np.abs((1 - np.sum(np.tile(phi, (1, nf)).T * np.exp(
        -1j * 2 * np.pi * np.tile(f,
                                  (p, 1)).T * np.tile(np.arange(1, p + 1),
                                                      (nf, 1)) * dt),
                             axis=1)))**2
    
    # return confidence if user supplied number of data
    if N != 0:
        ff = 1 # change later see https://www.ldeo.columbia.edu/users/menke/research_notes/menke_research_note154.pdf
        Nf = np.round(N/2) + 1
        fn = 1/(2*dt) # nyquist
        df = fn/Nf
        c = ff * sigma2 * dt / (2*Nf*df)
        var_num = 2*p*c**2
        var_psd = var_num / (den**2)

        return num/den, var_psd

    return num / den


# def spectrogram(y, window=[], nw=4):
#     """
#     Spectrogram with desired psd estimator.
#     """
