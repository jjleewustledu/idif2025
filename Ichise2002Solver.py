# The MIT License (MIT)
#
# Copyright (c) 2024 - Present: John J. Lee.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from numba import jit
from numpy.typing import NDArray

from TwoTCSolver import TwoTCSolver


@jit(nopython=True)
def prior_transform(
    u: np.ndarray,
    sigma: float
) -> np.ndarray:
    v = u
    v[0] = u[0] * 1e3 + 0.01  # K_1 / k_2
    v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
    v[2] = u[2] * 1e3 + 1  # k_3 / k_4
    v[3] = u[3] * 0.5 + 0.00001  # k_4 (1/s)
    v[4] = u[4] * 999.9 + 0.1  # V (mL/cm^{-3}) is total volume := V_N + V_S
    v[5] = u[5] * 20  # t_0 (s)
    v[6] = u[6] * sigma  # sigma ~ fraction of M0
    return v


@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    delta_time: int,
    isidif: bool
) -> float:
    assert rho.ndim == 1, "rho must be 1-dimensional"
    rho_pred, _, _, _ = signalmodel(v, rho, timesMid, taus, rho_input_func_interp, delta_time, isidif)
    sigma = v[-1]
    residsq = (rho_pred - rho) ** 2 / sigma ** 2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))
    if not np.isfinite(loglike):
        loglike = -1e300
    return loglike


@jit(nopython=True)
def signalmodel(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    rho_input_func_interp: np.ndarray,
    delta_time: int,
    isidif: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Ichise2002Solver assumes all input params to be decay-corrected """

    ks = np.zeros(4)
    ks[0] = v[0] * v[1]  # K_1
    ks[1] = v[1]  # k_2
    ks[2] = v[2] * v[3]  # k_3
    ks[3] = v[3]  # k_4
    V = v[4] 
    g1 = V * ks[1] * ks[3]
    g2 = -1 * ks[1] * ks[3]
    g3 = -(ks[1] + ks[2] + ks[3])
    g4 = ks[0]
    t_0 = v[5]

    n_times = rho_input_func_interp.shape[0]
    timesIdeal = np.arange(0, n_times)

    rho_input_func_interp = slide_input_func(
        rho_input_func_interp,
        timesIdeal,
        t_0,
        isidif
    )

    # propagate input function

    # rho_oversampled over-samples rho to match rho_t
    # rho_t is the inferred source signal, sampled to match input_func_interp
    # Downsample timesIdeal for faster integration
    timesIdeal_ds = timesIdeal[::delta_time]
    
    rho_oversampled = np.interp(timesIdeal_ds, timesMid, rho)
    rho_input_func_interp_ds = np.interp(timesIdeal_ds, timesIdeal, rho_input_func_interp)
    
    rho_ideal_ds = np.zeros(len(timesIdeal_ds))
    for tidx, time in enumerate(timesIdeal_ds):
        _tidx_interval = np.arange(0, tidx + 1) 
        _time_interval = timesIdeal_ds[_tidx_interval]

        _rho_interval = rho_oversampled[_tidx_interval]
        _rho_p_interval = rho_input_func_interp_ds[_tidx_interval]  # integration interval
        
        dt = _time_interval[1] - _time_interval[0]  # Uniform time step

        int4 = dt * np.sum((_rho_p_interval[1:] + _rho_p_interval[:-1])) / 2
        int3 = dt * np.sum((_rho_interval[1:] + _rho_interval[:-1])) / 2    

        # Compute double integral \int_0^T ds \int_0^s dt _rho_interval(t) using uniform time sampling
        # For each s, we sum up all values of _rho_interval from 0 to s
        # Then multiply by appropriate weights for double integral

        cumsum2 = np.cumsum(_rho_interval[:-1])  # np.cumsum is supported by numba
        int2 = np.sum(cumsum2) * dt * dt / 2  # dt for each integral

        cumsum1 = np.cumsum(_rho_p_interval[:-1])  # np.cumsum is supported by numba
        int1 = np.sum(cumsum1) * dt * dt / 2  # dt for each integral

        rho_ideal_ds[tidx] = g1 * int1 + g2 * int2 + g3 * int3 + g4 * int4

    rho_ideal_ds[rho_ideal_ds < 0] = 0
    
    # Upsample back to original time points
    rho_ideal = np.interp(timesIdeal, timesIdeal_ds, rho_ideal_ds)

    if not isidif:
        rho_pred = np.interp(timesMid, timesIdeal, rho_ideal)
    else:
        rho_pred = apply_boxcar(rho_ideal, timesMid, taus)
    return rho_pred, timesMid, rho_ideal, timesIdeal


@jit(nopython=True)
def apply_boxcar(rho: np.ndarray, timesMid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    times0_int = (timesMid - taus / 2).astype(np.int_)
    timesF_int = (timesMid + taus / 2).astype(np.int_)

    # Original implementation with loop ---------------------------------------
    # rho_sampled = np.full(times0_int.shape, np.nan)
    # for idx, (t0, tF) in enumerate(zip(times0_int, timesF_int)):
    #     rho_sampled[idx] = np.mean(rho[t0:tF])
    # return np.nan_to_num(rho_sampled, 0)

    # Optimized implementation using cumsum ------------------------------------
    # padding rho with 0 at beginning
    cumsum = np.cumsum(np.concatenate((np.zeros(1), rho)))
    rho_sampled = (cumsum[timesF_int] - cumsum[times0_int]) / taus
    return np.nan_to_num(rho_sampled, 0)


@jit(nopython=True)
def slide_input_func(
    rho_input_func_interp: np.ndarray,
    timesIdeal: np.ndarray,
    t_0: float,
    isidif: bool,
) -> np.ndarray:
    """ slide input function, aif or idif, to fit """

    if not isidif:

        # Find indices where input function exceeds 5% of max
        indices = np.where(rho_input_func_interp > 0.05 * np.max(rho_input_func_interp))
        # Handle case where no values exceed threshold
        if len(indices[0]) == 0:
            idx_a = 1  # Default to 1 if no values exceed threshold
        else:
            idx_a = max(indices[0][0], 1)  # Take first index but ensure >= 1

        # slide input function to left,
        # since its measurements is delayed by radial artery cannulation
        rho_input_func_interp = slide(
            rho_input_func_interp,
            timesIdeal,
            -timesIdeal[idx_a])
    return slide(
        rho_input_func_interp,
        timesIdeal,
        t_0)


@jit(nopython=True)
def slide(rho: np.ndarray, t: np.ndarray, dt: float, halflife: float = 0) -> np.ndarray:
    """ slides rho by dt seconds, optionally decays it by halflife. """

    if abs(dt) < 0.1:
        return rho
    rho = np.interp(t - dt, t, rho)  # copy of rho array
    if halflife > 0:
        return rho * np.power(2, -dt / halflife)
    else:
        return rho


class Ichise2002Solver(TwoTCSolver):
    """Solver implementing Ichise's 2002 model for PET data analysis.

    This class implements the tissue model described in Ichise et al. 2002 [1] for analyzing
    PET data using dynamic nested sampling. 

    Args:
        context: Context object containing PET data and configuration.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements.

    Example:
        >>> context = TissueContext(data_dict)
        >>> solver = Huang1980Solver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    References:        
        [1] Ichise M, Toyama H, Innis RB, Carson RE. 
            "Strategies to Improve Neuroreceptor Parameter Estimation by Linear Regression Analysis."
            Journal of Cerebral Blood Flow & Metabolism. 2002;22(10):1271-1281. 
            doi:10.1097/01.WCB.0000038000.34930.4E
    """

    def __init__(self, context):
        super().__init__(context)

    @property
    def labels(self):
        return [r"$K_1/k_2$", r"$k_2$", r"$k_3/k_4$", r"$k_4$", r"$V_N + V_S$", r"$t_0$", r"$\sigma$"]

    def volume_specific(self, v: list | tuple | NDArray):
        K1 = v[0] * v[1]  # K_1
        k2 = v[1]  # k_2
        k3 = v[2] * v[3]  # k_3
        k4 = v[3]  # k_4
        return K1*k3/(k2*k4)

    def volume_nonspecific(self, v: list | tuple | NDArray):
        return v[4] - self.volume_specific(v)

    def volume_total(self, v: list | tuple | NDArray):
        return v[4]
