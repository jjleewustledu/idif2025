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


from copy import deepcopy
import dynesty
from dynesty import utils as dyutils
import numpy as np
from numba import jit, float64
import pandas as pd

from DynestySolver import DynestySolver
from PETUtilities import PETUtilities


@jit(nopython=True)
def prior_transform(u: np.ndarray, halflife: float, sigma: float) -> np.ndarray:
    v = u
    v[0] = u[0] * 30  # t_0
    v[1] = u[1] * 30  # \tau_2 ~ t_2 - t_0
    v[2] = u[2] * 30  # \tau_3 ~ t_3 - t_2
    v[3] = u[3] * 20  # \alpha - 1
    v[4] = u[4] * 30 + 3  # 1/\beta
    v[5] = u[5] * 2.382 + 0.618  # p
    v[6] = u[6] * 3 - 3  # \delta p_2 ~ p_2 - p
    v[7] = u[7] * 3 - 3  # \delta p_3 ~ p_3 - p_2
    v[8] = u[8] * 5 * halflife  # 1/\gamma for s.s.
    v[9] = u[9] * 0.5 # f_2
    v[10] = u[10] * 0.5  # f_3
    v[11] = u[11] * 0.5  # f_{ss}
    v[12] = u[12] * 4 + 0.5  # A is amplitude adjustment
    v[13] = u[13] * sigma  # sigma ~ fraction of M0
    return v

@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesIdeal: np.ndarray,
    kernel: np.ndarray
) -> float:
    rho_pred, _, _ = signalmodel(v, timesIdeal, kernel)
    sigma = v[-1]
    residsq = (rho_pred - rho) ** 2 / sigma ** 2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))
    if not np.isfinite(loglike):
        loglike = -1e300
    return loglike

@jit(nopython=True)
def signalmodel(
    v: np.ndarray,
    timesIdeal: np.ndarray, 
    kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_0 = v[0]
    tau_2 = v[1]
    tau_3 = v[2]
    a = v[3]
    b = 1 / v[4]  # \beta <- 1/\beta    
    p = v[5]
    dp_2 = v[6]
    dp_3 = v[7]
    g = 1 / v[8]  # \gamma <- 1/\gamma
    f_2 = v[9]
    f_3 = v[10]
    f_ss = v[11]
    A = v[12]

    rho_ = A * solution_3bolus_series(timesIdeal, t_0, tau_2, a, b, p, dp_2, dp_3, g, f_2, f_3)
    rho = apply_dispersion(rho_, kernel)
    A_qs = 1 / np.max(rho)
    rho_pred = A_qs * rho
    rho_ideal = A_qs * rho_
    return rho_pred, rho_ideal, timesIdeal

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
        f_2_ * solution_1bolus(t, t_0 + tau_2, a, b, max(0.618, p + dp_2)) +
        f_3_ * solution_1bolus(t, t_0, a, b + g, max(0.618, p + dp_2 + dp_3))
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
def apply_dispersion(rho: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rho_sampled = np.convolve(rho, kernel, mode="full")
    return rho_sampled[:rho.size]


class RadialArterySolver(DynestySolver):
    def __init__(self, context):
        super().__init__(context)
        self.data = self.context.data

    @property
    def labels(self):
        return [
            r"$t_0$", r"$\tau_2$", r"$\tau_3$",
            r"$\alpha - 1$", r"$1/\beta$", r"$p$", r"$\delta p_2$", r"$\delta p_3$", r"$1/\gamma$",
            r"$f_2$", r"$f_3$", r"$f_{ss}$",
            r"$A$", r"$\sigma$"]

    def __create_loglike_partial(self):
        # Cache values using nonlocal for faster access
        rho = self.data.rho
        timesIdeal = self.data.timesIdeal
        kernel = self.data.kernel

        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesIdeal, kernel
            return loglike(v, rho, timesIdeal, kernel)
        return wrapped_loglike

    def __create_prior_transform_partial(self):
        # Cache values using nonlocal for faster access
        halflife = self.data.halflife
        sigma = self.data.sigma

        # Create wrapper that matches dynesty's expected signature
        def wrapped_prior_transform(v):
            nonlocal halflife, sigma
            return prior_transform(v, halflife, sigma)
        return wrapped_prior_transform
    
    def _run_nested(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False, 
            parc_index: int = 0
    ) -> dyutils.Results:
        if resume:
            sampler = dynesty.DynamicNestedSampler.restore(checkpoint_file)
        else:
            loglike = self.__create_loglike_partial()
            prior_transform = self.__create_prior_transform_partial()
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=loglike,
                prior_transform=prior_transform,
                ndim=self.ndim,
                sample=self.data.sample,
                nlive=self.data.nlive,
                rstate=self.data.rstate
            )
        sampler.run_nested(checkpoint_file=checkpoint_file, print_progress=print_progress, resume=resume)
        return sampler.results

    def signalmodel(self, v: np.ndarray, parc_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return signalmodel(v, self.data.timesIdeal, self.data.kernel)
    
    def loglike(self, v: np.ndarray, parc_index: int = 0) -> float:
        return loglike(v, self.data.rho, self.data.timesIdeal, self.data.kernel)
