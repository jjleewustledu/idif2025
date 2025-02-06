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


import dynesty
from dynesty import utils as dyutils
import numpy as np
from numba import jit
from numpy.typing import NDArray

from InputFuncSolver import InputFuncSolver


@jit(nopython=True)
def prior_transform(u: np.ndarray, halflife: float, sigma: float) -> np.ndarray:
    v = u
    v[0] = u[0] * 30  # t_0
    v[1] = u[1] * 30  # \tau_2 ~ t_2 - t_0
    v[2] = u[2] * 20  # \alpha - 1
    v[3] = u[3] * 30 + 3  # 1/\beta
    v[4] = u[4] * 2.5 + 0.5  # p
    v[5] = u[5] * 3 - 3  # \delta p_2 ~ p_2 - p
    v[6] = u[6] * 3 - 3  # \delta p_3 ~ p_3 - p_2
    v[7] = u[7] * 5 * halflife  # 1/\gamma for s.s.
    v[8] = u[8] * 0.9  # f_2
    v[9] = u[9] * 0.9  # f_3
    v[10] = u[10] * 4 + 0.5  # A is amplitude adjustment
    v[11] = u[11] * sigma  # sigma ~ fraction of M0
    return v


@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesIdeal: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray
) -> float:
    rho_pred, _, _ = signalmodel(v, timesIdeal, timesMid, taus)
    sigma = v[-1]
    residsq = (rho_pred - rho) ** 2 / sigma ** 2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))
    if not np.isfinite(loglike) or np.isnan(loglike):
        loglike = -1e300
    return loglike


@jit(nopython=True)
def signalmodel(
    v: np.ndarray,
    timesIdeal: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_0 = v[0]
    tau_2 = v[1]
    a = v[2]
    b = 1 / v[3]  # \beta <- 1/\beta    
    p = v[4]
    dp_2 = v[5]
    dp_3 = v[6]
    g = 1 / v[7]  # \gamma <- 1/\gamma
    f_2 = v[8]
    f_3 = v[9]
    A = v[10]

    rho_ = A * solution_3bolus_series(timesIdeal, t_0, tau_2, a, b, p, dp_2, dp_3, g, f_2, f_3)
    rho = apply_boxcar(rho_, timesMid, taus)
    A_qs = 1 / np.max(rho)
    signal = A_qs * rho
    ideal = A_qs * rho_
    return signal, ideal, timesIdeal


@jit(nopython=True)
def solution_3bolus_series(
    t: np.ndarray, 
    t_0: float, tau_2: float, 
    a: float, b: float, p: float, dp_2: float, dp_3: float, g: float, 
    f_2: float, f_3: float
) -> np.ndarray:
    """ three sequential generalized gamma distributions """
    f_1_ = (1 - f_3) * (1 - f_2)
    f_2_ = (1 - f_3/2) * f_2
    f_3_ = (1 - f_2/2) * f_3 
    rho = (
        f_1_ * solution_1bolus(t, t_0, a, b, p) +
        f_2_ * solution_1bolus(t, t_0 + tau_2, a, b, max(0.5, p + dp_2)) +
        f_3_ * solution_1bolus(t, t_0, a, b + g, max(0.5, p + dp_2 + dp_3))
    )
    return rho


@jit(nopython=True)
def solution_1bolus(t: np.ndarray, t_0: float, a: float, b: float, p: float) -> np.ndarray:
    """Generalized gamma distribution, using numpy with optimized memory allocation.

    Args:
        t (array_like): Time points
        t_0 (float): Time offset
        a (float): Shape parameter
        b (float): Scale parameter
        p (float): KWW shape parameter

    Returns:
        ndarray: Normalized gamma distribution values
    """
    t_ = t - t_0
    t_ = np.maximum(t_, 0)
    # Only compute where t_ > 0 to avoid negative powers
    rho = np.zeros_like(t_)
    mask = t_ > 0
    rho[mask] = np.power(t_[mask], a) * np.exp(-np.power((b * t_[mask]), p))
    
    max_val = np.max(rho)
    if max_val > 0:
        rho /= max_val
    return np.nan_to_num(rho, 0)


@jit(nopython=True)
def apply_boxcar(rho: np.ndarray, timesMid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    times0_int = (timesMid - taus / 2).astype(np.int_)
    timesF_int = (timesMid + taus / 2).astype(np.int_)

    # Original implementation with loop
    # rho_sampled = np.full(times0_int.shape, np.nan)
    # for idx, (t0, tF) in enumerate(zip(times0_int, timesF_int)):
    #     rho_sampled[idx] = np.mean(rho[t0:tF])        
    # return np.nan_to_num(rho_sampled, 0)

    # Optimized implementation using cumsum, padding rho with 0 at beginning
    cumsum = np.cumsum(np.concatenate((np.zeros(1), rho)))
    rho_sampled = (cumsum[timesF_int] - cumsum[times0_int]) / taus
    return np.nan_to_num(rho_sampled, 0)


class BoxcarSolver(InputFuncSolver):
    """Solver for fitting boxcar input functions to PET data.

    This class implements methods for fitting a boxcar-shaped input function model to PET data
    using dynamic nested sampling. The model consists of three gamma distributions with variable
    parameters and relative weights.

    Args:
        context: The context object containing data and configuration for the solver.

    Attributes:
        data: Reference to the context's data object containing PET measurements and metadata.
        labels (list): Parameter labels for plotting and output.

    Example:
        >>> context = BoxcarContext(data_dict)
        >>> solver = BoxcarSolver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    Notes:
        The boxcar model uses 12 parameters:
        - t_0: Time offset
        - tau_2: Time delay for second bolus
        - alpha: Shape parameter
        - beta: Scale parameter
        - p: KWW shape parameter
        - dp_2, dp_3: Shape parameter offsets
        - gamma: Steady state decay rate
        - f_2, f_3: Relative weights
        - A: Amplitude adjustment
        - sigma: Noise parameter
    """
    def __init__(self, context):
        super().__init__(context)
        self.data = self.context.data

    @property
    def labels(self):
        return [
            r"$t_0$", r"$\tau_2$", 
            r"$\alpha - 1$", r"$1/\beta$", r"$p$", r"$\delta p_2$", r"$\delta p_3$", r"$1/\gamma$",
            r"$f_2$", r"$f_3$", 
            r"$A$", 
            r"$\sigma$"]
    
    @staticmethod
    def _loglike(selected_data: dict):
        # Cache values using nonlocal for faster access
        rho = selected_data["rho"]
        timesIdeal = selected_data["timesIdeal"]
        timesMid = selected_data["timesMid"]
        taus = selected_data["taus"]
        
        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesIdeal, timesMid, taus
            return loglike(v, rho, timesIdeal, timesMid, taus)
        return wrapped_loglike

    @staticmethod
    def _prior_transform(selected_data: dict):
        # Create wrapper that matches dynesty's expected signature
        halflife = selected_data["halflife"]
        sigma = selected_data["sigma"]

        def wrapped_prior_transform(v):
            nonlocal halflife, sigma
            return prior_transform(v, halflife, sigma)
        return wrapped_prior_transform
    
    @staticmethod
    def _run_nested(selected_data: dict) -> dyutils.Results:
        if selected_data["resume"]:
            sampler = dynesty.DynamicNestedSampler.restore(selected_data["checkpoint_file"])
        else:
            loglike = BoxcarSolver._loglike(selected_data)
            prior_transform = BoxcarSolver._prior_transform(selected_data)
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=loglike,
                prior_transform=prior_transform,
                ndim=selected_data["ndim"],
                sample=selected_data["sample"],
                nlive=selected_data["nlive"],
                rstate=selected_data["rstate"]
            )
        sampler.run_nested(
            checkpoint_file=selected_data["checkpoint_file"], 
            print_progress=selected_data["print_progress"], 
            resume=selected_data["resume"],
            wt_kwargs={"pfrac": selected_data["pfrac"]}
        )
        return sampler.results
        
    def _run_nested_single(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False
    ) -> dyutils.Results:
        
        args = {
            "rho": self.data.rho,
            "timesIdeal": self.data.timesIdeal,
            "timesMid": self.data.timesMid,
            "taus": self.data.taus,
            "halflife": self.data.halflife, 
            "sigma": self.data.sigma,
            "ndim": self.ndim,
            "sample": self.data.sample,
            "nlive": self.data.nlive,
            "rstate": self.data.rstate,
            "checkpoint_file": checkpoint_file,
            "resume": resume,
            "pfrac": self.data.pfrac,
            "print_progress": print_progress
        }

        _results = BoxcarSolver._run_nested(args)
        self._set_cached_dynesty_results(_results)
        return _results
    
    def signalmodel(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        return signalmodel(v, self.data.timesIdeal, self.data.timesMid, self.data.taus)
    
    def loglike(
            self,
            v: list | tuple | NDArray,
            parc_index: int | None = None
    ) -> float:
        v = np.array(v, dtype=float)
        if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != self.ndim:
            raise ValueError(f"v must be 1-dimensional array of length {self.ndim}")
        return loglike(v, self.data.rho, self.data.timesIdeal, self.data.timesMid, self.data.taus)
