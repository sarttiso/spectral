import numpy as np
from scipy.signal import windows
from scipy import stats
import tqdm as tqdm
from scipy.fft import fft


def periodogram(y, dt):
    """one sided periodogram (real-valued signals)

    Args:
        y (array): data
        dt (float): sampling interval

    Returns:
        array: power spectral density estimate
        array: frequency axis
    """
    N = len(y)
    fy = fft(y)
    # just need one half for real-valued signals
    fy = fy[0:N//2+1]

    f = freq(N, dt)

    # square and scale
    P = dt/N * np.abs(fy)**2

    # make one-sided
    # if odd length signal, don't double zero frequency but double everything else.
    # highest frequency is just below Nyquist
    if np.mod(N, 2):
        P[1:] = 2*P[1:]
    # for even length signal, last frequency is Nyquist, but occurs only once, so don't
    # dobule Nyquist or zero frequency
    else:
        P[1:-1] = 2*P[1:-1]

    return P, f


def multitaper(y, dt, nw=4, return_bk=False):
    """
    multi-taper spectral density estimation using DPSS sequences as described by Percival and Walden.

    IN:
    y: time series of evenly spaced observations in time
    dt: sample spacing in sample coordinate
    nw: (default 4) time-halfbandwidth product

    OUT:
    S_est: estimate for the power spectral density estimation (one sided, i.e. multiplied by 2)
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
    # get discrete prolate spheroidal sequence (dpss) windows as well as eigenvalues
    wins, evals = windows.dpss(N, nw, Kmax=K, return_ratios=True)
    # get tapered spectra
    Sk = np.zeros((Nf, K))

    # use fft you dum dum
    for ii in range(K):
        Sk[:, ii] = N * periodogram(wins[ii, :] * y, dt)[0]

    # implement adaptive multitaper spectral estimator (Percival and Walden, pg. 370)
    # start with eqn 369a (i.e. no estimate for weights bk)
    K_cur = 1
    S_est = np.sum(np.tile(evals[0:K_cur], (Nf, 1)) * Sk[:, 0:K_cur],
                   axis=1) / np.sum(evals[0:K_cur])
    bk = np.zeros((Nf, K))
    # make convenient tiled version of eigenvalues
    evals_tile = np.tile(evals[0:K], (Nf, 1))
    # precompute variance
    var_y = np.var(y)

    # iterate over equations 368a and 370a
    for ii in range(5):
        # smart tile
        S_est_tile = np.empty((K, *S_est.shape), S_est.dtype)
        S_est_tile[...] = S_est
        # get weights
        bk = S_est_tile.T / (evals_tile * S_est_tile.T + (1 - evals_tile) * var_y)
        # update spectrum
        S_est = np.sum(bk**2 * evals_tile * Sk[:, 0:K], axis=1) / \
                np.sum(bk**2 * evals_tile, axis=1)

    if return_bk:
        return S_est, f, bk
    return S_est, f


def LjungBox_p(residuals, order, lags=15):
    """Ljung-Box statistic for the evaluation of the significance of autocorrelation

    Args:
        residuals (array): series to test; can be thought of as "prewhitened". typically
        residuals after ARIMA fitting
        order (int): p+q+d for the ARIMA fit
        lags (int, optional): number of lags to include in evaluation. must be larger
        than order. Defaults to 15.

    Returns:
        float: p-value. small p indicate significant autocorrelation, which is
        inconsistent with truly prewhitened residuals
    """
    T = len(residuals)
    Q = np.sum((acvs(residuals, np.arange(1, lags+1))/acvs(residuals, 0))**2/(T-np.arange(1, lags+1))) * T * (T+2)
    return 1 - stats.chi2.cdf(Q, lags-order)


def white_psd_conf(conf, sig2, dt, K=1, fac=1):
    """return confidence interval for white noise

    Args:
        conf (float): [0, 1] confidence level
        sig2 (float): variance of white noise process
        dt (float): sampling interval
        K (int, optional): number of applied tapers. Defaults to 1.
        fac (float, optional): Factor to apply to account for tapering. Typicall 1/N *
        sum(win^2) Defaults to 1. Also 1 for multitaper method, which preserves variance.

    Returns:
        float: constant power spectral density for requested confidence level
    """
    S_sig = 2*sig2*dt*fac
    S_conf = stats.gamma.ppf(conf, K, scale=S_sig/K)

    return S_conf


def dpss_evals(N, W):
    """compute eigenvalues of DPSS sequence with time halfbandwidth product NW. Super
    slow, gotta be a better way, but haven't found it yet.

    Fortunately, scipy.windows.dpss returns the evals!
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


def freq(N, dt):
    """generate one-sided frequency axis for given N data and dt sample spacing

    Args:
        N (int): number of data
        dt (float): sampling interval

    Returns:
        array: one-sided frequency axis
    """
    fs = 1 / dt
    fi = fs / N
    # the fi/2 just ensures that, for even numbered signals, the nyquist frequency is included
    fx = np.arange(0, fs / 2 + fi/4, fi)
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


def AR(phi, sig, n, scale=1.2):
    """instead should just use statsmodels.tsa.arima_process.ArmaProcess

    Args:
        phi (array-like): autoregressive coefficients
        sig (float): innovation variance
        n (int): number of data to generate
        scale (float, optional): "burn-in" amount; 1.2 means 20% burn-in. Defaults to 1.2.

    Returns:
        _type_: _description_
    """
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


def ARpsd(sigma2, phi, dt, f, return_conf=False, conf=0.975, fac=1):
    """
    The % output function will accept as input an innovations variance (S), a
    vector of lag coefficients (p), a vector of frequencies at which to
    evaluate the density (f), and the Nyquist frequency (fn)
    using modified definition from p. 392 of Percival and Walden:
    S(f) = 1/fn * sig^2/|1 - sum(p_j exp(-ij pi f/fn))|^2

    if N (number of data) is provided, function returns the confidence interval for the
    psd.
    confidence defaults to 97.5%.
    fac is the factor to apply based on the varaicne contributed by applied tapers
    """
    nf = len(f)
    p = len(phi)
    num = sigma2 * dt
    den = np.abs((1 - np.sum(np.tile(phi, (1, nf)).T * np.exp(
        -1j * 2 * np.pi * np.tile(f, (p, 1)).T * \
                          np.tile(np.arange(1, p + 1), (nf, 1)) * dt),
                             axis=1)))**2
    
    psd = num/den
    
    # return confidence if user supplied number of data
    if return_conf:
        # ff = 1 # change later see https://www.ldeo.columbia.edu/users/menke/research_notes/menke_research_note154.pdf
        # still super sketchy  
        var_num = sigma2**2*dt**2*fac
        psd_sig = np.sqrt(var_num / (den**2))

        psd_conf = stats.norm.ppf(conf, loc=psd, scale=psd_sig)

        return psd, psd_conf

    return psd

def ARMA_psd(f, phi, theta, sig2, dt=1.0):
    """theoretical power spectral density for an ARMA process evaluated at
    frequency(ies) f

    Args:
        f (float or array): frequency or frequencies at which to evluate the PSD
        phi (array-like): list of autoregressive coefficients
        theta (array-like): list of moving average coefficients
        sig2 (float): variance for white noise process
        dt (float, optional): sampling interval. Defaults to 1.

    Returns:
        array: evaluation of theoretical PSD
    """
    nf = len(f)
    p = len(phi)
    q = len(theta)
    num = np.abs((1 + np.sum(np.tile(theta, (1, nf)).T * np.exp(
        -1j * 2 * np.pi * np.tile(f, (q, 1)).T * \
                          np.tile(np.arange(1, q + 1), (nf, 1)) * dt),
                             axis=1)))**2
    den = np.abs((1 - np.sum(np.tile(phi, (1, nf)).T * np.exp(
        -1j * 2 * np.pi * np.tile(f, (p, 1)).T * \
                          np.tile(np.arange(1, p + 1), (nf, 1)) * dt),
                             axis=1)))**2
    return sig2*dt*num/den

# def spectrogram(y, window=[], nw=4):
#     """
#     Spectrogram with desired psd estimator.
#     """
