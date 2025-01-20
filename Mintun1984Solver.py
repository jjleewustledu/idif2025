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
def prior_transform(
    u: np.ndarray,
    sigma: float
) -> np.ndarray:
        v = u
        v[0] = u[0] * 0.8 + 0.1  # OEF
        v[1] = u[1] * 1.8 + 0.1  # frac. water of metab. at 90 s
        v[2] = u[2] * 0.9 + 0.1  # {v_{post} + 0.5 v_{cap}} / v_1
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * (-60)  # \tau_a (s)
        v[5] = u[5] * 20  # \tau_d (s)
        v[6] = u[6] * sigma  # sigma ~ fraction of A0
        return v

@jit(nopython=True)
def loglike(
    v: np.ndarray,
    rho: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    input_func_interp: np.ndarray,
    ks: np.ndarray,
    v1: float,
    isidif: bool
) -> float:
    rho_pred, _, _ = signalmodel(v, timesMid, taus, input_func_interp, ks, v1, isidif)
    sigma = v[-1]
    residsq = (rho_pred - rho) ** 2 / sigma ** 2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))
    if not np.isfinite(loglike):
        loglike = -1e300
    return loglike

@jit(nopython=True)
def signalmodel(
    v: np.ndarray,
    timesMid: np.ndarray,
    taus: np.ndarray,
    input_func_interp: np.ndarray,
    ks: np.ndarray,
    v1: float,
    isidif: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Mintun1984Model is only valid for [15O] """
        HL = 122.2416  # [15O]
        ALPHA = np.log(2) / HL
        DENSITY_PLASMA = 1.03
        DENSITY_BLOOD = 1.06

        oef = v[0]  # extraction fraction
        f_h2o = v[1] * DENSITY_PLASMA / DENSITY_BLOOD  # fraction of water of metabolism at 90 s 
        v_post_cap = v[2]  # volume of post-capillary and capillaryspace
        t_0 = v[3]  # delay of input function
        tau_a = v[4]  # delay of input function
        tau_dispersion = v[5]  # dispersion of input function
        f = ks[0]  # flow, s^{-1}
        lamb = ks[1]  # partition coefficient
        PS = ks[2]  # permeability surface area product, s^{-1} 

        n_times = max(input_func_interp.shape)
        timesIdeal = np.arange(0, n_times)

        # disperse input function, compatible with numba

        dispersion = np.exp(-timesIdeal / tau_dispersion)
        z_dispersion = np.sum(dispersion)  # partition function of dispersion
        n_input = len(input_func_interp)
        n_disp = len(dispersion)
        n_conv = n_input + n_disp - 1
        conv_result = np.zeros(n_conv)
        for i in range(n_conv):
            for j in range(max(0, i - n_disp + 1), min(i + 1, n_input)):
                conv_result[i] += input_func_interp[j] * dispersion[i - j]
        input_func_interp = conv_result[:n_times] / z_dispersion
        if not isidif:
            # slide input function to left, with decay adjustments, 
            # since its measurements is delayed by radial artery cannulation
            input_func_interp = slide(input_func_interp, timesIdeal, tau_a, HL)

        # estimate shape of water of metabolism

        indices = np.where(input_func_interp > 0.05 * max(input_func_interp))
        try:  
            idx0 = max([indices[0], 1])
        except IndexError:
            idx0 = 1
        idxU = min([idx0 + 90, n_times - 1])  # time of eval of magnitude of water of metab; cf. Mintun1984
        shape = np.zeros(n_times)
        n_times_1 = n_times - idx0 + 1
        try:
            y = (n_times - idx0) / (idxU - idx0)
        except ZeroDivisionError:
            y = 1
        shape[-n_times_1:] = np.linspace(0, y, n_times_1)  # shape(idxU) == 1
        timesDuc = np.zeros(n_times)
        timesDuc[idx0:] = np.linspace(0, n_times_1 - 2, n_times_1 - 1)
        shape_duc = shape * np.power(2, -(timesDuc - idxU + 1) / HL)  # decay-uncorrected

        # set scale of artery_h2o
        # activity of water of metab \approx activity of oxygen after 90 sec

        artery_h2o = f_h2o * input_func_interp[idxU] * shape_duc

        # compartment 2, using m, f, lamb, compatible with numba

        artery_o2 = input_func_interp - artery_h2o
        artery_o2[artery_o2 < 0] = 0
        m = 1 - np.exp(-PS / f)
        propagator = np.exp(-m * f * timesIdeal / lamb - ALPHA * timesIdeal)
        n_propagator = len(propagator)
        n_artery = len(artery_h2o) 
        n_conv = n_propagator + n_artery - 1        
        conv_h2o = np.zeros(n_conv)
        conv_o2 = np.zeros(n_conv)        
        for i in range(n_conv):
            for j in range(max(0, i - n_propagator + 1), min(i + 1, n_artery)):
                conv_h2o[i] += propagator[i - j] * artery_h2o[j]
                conv_o2[i] += propagator[i - j] * artery_o2[j]                
        rho2 = m * f * (conv_h2o + oef * conv_o2)

        # compartment 1
        # v_post = 0.83*v1
        # v_cap = 0.01*v1
        # R = 0.85  # ratio of small-vessel to large-vessel Hct needed when v1 := CBV * R

        rho1 = v1 * (1 - oef * v_post_cap) * artery_o2

        # package compartments

        rho_ideal = rho1[:n_times] + rho2[:n_times]  # rho_ideal is interpolated to the input function times
        if not isidif:
            rho_ideal = slide(rho_ideal, timesIdeal, t_0, HL)
            rho = np.interp(timesMid, timesIdeal, rho_ideal)
        else:
            rho = apply_boxcar(rho_ideal, timesMid, taus)
        return rho, timesMid, rho_ideal, timesIdeal

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

@jit(nopython=True)
def slide(rho: np.ndarray, t: np.ndarray, dt: float, halflife: float=None) -> np.ndarray:
    """ slides rho by dt seconds, optionally decays it by halflife. """

    if abs(dt) < 0.1:
        return rho
    rho = np.interp(t - dt, t, rho)  # copy of rho array
    if halflife:
        return rho * np.power(2, -dt / halflife)
    else:
        return rho


class Mintun1984Solver(DynestySolver):
    def __init__(self, context):
        super().__init__(context)
        self.data = self.context.data

    @property
    def labels(self):
        return [
            r"OEF", r"$f_{H_2O}$", r"$v_p + 0.5 v_c$", r"$t_0$", r"$\tau_a$", r"$\tau_d$", r"$\sigma$"]

    def __create_loglike_partial(self, parc_index: int = 0):
        # Cache values using nonlocal for faster access
        rho = self.data.rho[parc_index]
        timesMid = self.data.timesMid
        taus = self.data.taus
        input_func_interp = self.data.input_func_interp
        ks = self.data.ks[parc_index]
        v1 = self.data.v1[parc_index]
        isidif = self.data.isidif

        # Check dimensions
        if rho.ndim != 1:
            raise ValueError("rho must be 1-dimensional")
        if input_func_interp.ndim != 1:
            raise ValueError("input_func_interp must be 1-dimensional") 
        if ks.ndim != 1:
            raise ValueError("ks must be 1-dimensional") 
        if not np.isscalar(v1):
            raise ValueError("v1 must be scalar")
        
        # Create wrapper that matches dynesty's expected signature
        def wrapped_loglike(v):
            nonlocal rho, timesMid, taus, input_func_interp, ks, v1, isidif
            return loglike(
                v,
                rho,
                timesMid,
                taus, 
                input_func_interp,
                ks,
                v1,
                isidif)
        return wrapped_loglike

    def __create_prior_transform_partial(self):
        # Create wrapper that matches dynesty's expected signature
        sigma = self.data.sigma

        def wrapped_prior_transform(v):
            nonlocal sigma
            return prior_transform(v, sigma)
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
            loglike = self.__create_loglike_partial(parc_index)
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

    def run_nested_all_parcs(self):
        pass

    def save_results_all_parcs(self):
        pass
    
    def signalmodel(self, v: np.ndarray, parc_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return signalmodel(
            v, 
            self.data.timesMid,
            self.data.taus,
            self.data.input_func_interp,
            self.data.ks[parc_index],
            self.data.v1[parc_index],
            self.data.isidif
        )
    
    def loglike(self, v: np.ndarray, parc_index: int = 0) -> float:
        return loglike(
            v,
            self.data.rho,
            self.data.timesMid,
            self.data.taus,
            self.data.input_func_interp,
            self.data.ks[parc_index],
            self.data.v1[parc_index],
            self.data.isidif
        )
