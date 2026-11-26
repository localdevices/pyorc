"""Hydraulic model functions."""

import numpy as np


def strickler_to_z0(k_st, depth):
    """Convert Strickler roughness k_st [m^(1/3)/s] to roughness length z0 [m].

    We use the White Colebrook formulation combined with the Chezy formulation for conveyance to estimate z0

    Reference: SOBEK Suite - 1D/2D modelling suite for integral water solutions - Technical Reference Manual,
    Deltares, Revision 80735, Version 3.8.0, November 2026.

    C = z^(1/6) * k_st

    C = 18 * log10(12*z/(z0 * 30))

    Note that z0 * 30 is the Nikuradse roughness.
    """
    g = 9.81
    kappa = 0.4
    chezy = k_st * depth**0.166667
    exponent = -(1 + (kappa * chezy) / np.sqrt(g))
    return depth * np.exp(exponent)


def log_shore_model(x, k_st=30, S0=0.005, dist_scale=0.0, alpha=1.0, pw=0.0):
    """Compute the velocity as function of roughness and shape of u_star.

    The following principles are used:

    - basis is the normal log-wall relationship with u = u*/kappa * ln(z/z0) with u* established from
    - z0 is directly determined by combining Strickler and White Colebrook
    - With presence of large boulders or water plants, the strickler coefficient will increase with depth also,
        typically with a power of 1/3 (De Bos and Bijkerk (1963), Cultuurtechnisch Tijdschrift, no 3. p. 149-155).
        This is introduced with a power addition ``pw``. If there are no water plants, ``pw`` should be set to 0
        (default).
    - The influence of vicinity to the shore linearly increases with the depth with multiplier ``dist_scale``. This
        influence was proven to be reasonably independent of the effect of vegetation through ``pw`` through a
        sensitivity analysis. ``dist_scale`` simulates a combination of roughness increases towards the banks, and
        increases in secondary flows and energy loss with higher depth.
    - The overall velocity is then computed using the log-wall relationship, modified by the effects of shore proximity
        and water plants/boulders as per above.
    """
    g = 9.81
    kappa = 0.4
    depth, dist = x
    z0 = strickler_to_z0(k_st, depth)
    # the effect of roughness due to shore distance increases with water level
    dist_max = dist_scale * depth
    f_w = 1 - (1 - np.clip(dist / np.maximum(dist_max, 1e-6), 0, 1)) ** alpha
    v_star = np.sign(S0) * np.sqrt(g * np.abs(S0) * depth)
    # use log-wall to estimate v, increase with depth**pw for water plants/boulder effects (De Bos and Bijkerk, 1963)
    v = np.maximum(np.log(np.maximum(depth, 1e-6) / np.maximum(z0, 1e-6)), 0) * v_star / kappa * f_w * depth**pw
    return v


def weighted_mse(pars, func, x, y, weights):
    """Calculate weighted mean squared error based on precalculated weights per sample.

    Parameters
    ----------
    pars : list or tuple
        Parameters passed to func.
    func : callable
        Model function returning predicted y.
    x : tuple or array-like
        Independent variable(s).
    y : array-like
        Observed target values.
    weights : array-like
        predefined weights per sample

    Returns
    -------
    float
        Weighted mean squared error.

    """
    y = np.asarray(y)
    y_pred = func(x, *pars)

    # Weighted mean squared error
    squared_errors = (y_pred - y) ** 2
    return np.sum(weights * squared_errors) / np.sum(weights)


# create optimization function and fit
def mse(pars, func, x, y):
    """Give mean of sum of squares between evaluation of function with provided parameters and observations.

    Parameters
    ----------
    pars : list or tuple
        parameter passed as *args to func.
    func: function def
        receiving X and *pars as input and returning predicted Y as result
    x: tuple with lists or array-likes
        indepent variable(s).
    y: list or array-like
        dependent variable, predicted by func

    Returns
    -------
    y_pred : list or array-like
        predicted Y from X and pars by func

    """
    y_pred = func(x, *pars)
    ms_error = np.mean((y_pred - y) ** 2)
    return ms_error


def wrap_mse(pars_iter, func, x, y, weights, lower, span):
    """Wrap function mse for optimization purposes with norm to real bounds."""
    real_params = lower + pars_iter * span
    return weighted_mse(real_params, func, x, y, weights)
