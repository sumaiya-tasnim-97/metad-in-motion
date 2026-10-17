"""Shared helpers for infrequent-metadynamics kinetics (demo 3 and validate.py)."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest, chi2


def bias_at(Vb, x, g0, dg):
    """Linear interpolation of per-walker bias grids Vb[k, :] at positions x[k].
    Returns the bias and its slope (the derivative of the interpolant)."""
    f = (x - g0) / dg
    i = np.clip(np.floor(f).astype(int), 0, Vb.shape[1] - 2)
    r = f - i
    rows = np.arange(len(x))
    v0, v1 = Vb[rows, i], Vb[rows, i + 1]
    return v0 * (1 - r) + v1 * r, (v1 - v0) / dg


def fit_tau(t):
    """Least-squares fit of the Poisson CDF 1 - exp(-t/tau) to the empirical CDF (JCTC eq. 13)."""
    s = np.sort(np.asarray(t)); c = np.arange(1, s.size + 1) / s.size
    return curve_fit(lambda x, tc: 1 - np.exp(-x / tc), s, c, p0=[s.mean()])[0][0]


def calibrated_ks(t, B=2000, seed=12345):
    """KS distance between the data and the fitted exponential, with a p-value calibrated by a
    parametric bootstrap: tau is refitted on every synthetic data set, so the test accounts for
    tau having been estimated from the same data (the issue the Lilliefors test addresses)."""
    t = np.asarray(t); n = t.size; rng = np.random.default_rng(seed)
    tc = fit_tau(t); D = kstest(t, 'expon', args=(0, tc)).statistic
    Ds = np.empty(B)
    for b in range(B):
        x = rng.exponential(tc, n)
        Ds[b] = kstest(x, 'expon', args=(0, fit_tau(x))).statistic
    return D, float((Ds >= D).mean())


def kaminsky_ci(t, level=0.95):
    """Chi-squared confidence interval for the mean of exponential data (JCTC eq. 14)."""
    t = np.asarray(t); n = t.size; S = t.sum(); a = (1 - level) / 2
    return 2 * S / chi2.ppf(1 - a, 2 * n), 2 * S / chi2.ppf(a, 2 * n)
