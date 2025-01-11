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

from __future__ import absolute_import
from TissueModel import TissueModel
from Boxcar import Boxcar

# general & system functions
import os

# basic numeric setup
import numpy as np
import scipy.integrate as integrate


class SpectralAnalysis(TissueModel):
    """
    """

    sigma = None  # class attribute needed by dynesty
    M = None  # class attribute needed by SpectralAnalysis
    T_0 = None
    TAU_A = None

    def __init__(self, input_function, pet_measurement, M=1, t_0=10, tau_a=0, **kwargs):
        super().__init__(input_function, pet_measurement, **kwargs)
        SpectralAnalysis.sigma = 0.2
        SpectralAnalysis.M = M
        SpectralAnalysis.T_0 = t_0
        SpectralAnalysis.TAU_A = tau_a
        if "-m" not in self.TAG:
            self.TAG = self.TAG + f"-m{M}"
        if "-nlive" not in self.TAG:
            self.TAG = self.TAG + f"-nlive{nlive}"
        self._labels = None

    @property
    def labels(self):
        if self._labels:
            return self._labels

        M = SpectralAnalysis.M
        lbls = []
        for m in np.arange(M):
            lbls.extend([fr"$\alpha_{{{m + 1}}}$", fr"$\beta_{{{m + 1}}}$"])
        lbls.extend([r"$a_0$", r"$\sigma$"])
        # print(lbls)
        self._labels = lbls
        return lbls

    def prior_transform(self):
        return {
            "SpectralAnalysis": self.prior_transform_sa,
        }.get(self.__class__.__name__, self.prior_transform_sa)
        # default is self.prior_transform_sa for spectral analysis

    @staticmethod
    def prior_transform_sa(u):
        M = SpectralAnalysis.M
        T = 2 * M + 1
        v = u
        v[0] = u[0]
        v[1] = u[1] * 0.05 + 0.00003  # 1/(3*T_{end}) < \beta < 3/T_{in}
        for m in np.arange(1, M, 1):
            v[2 * m] = u[2 * m]  # \alpha_1 ~ K_1
            v_max = v[2 * m - 1] - np.finfo(float).eps
            v[2 * m + 1] = u[2 * m + 1] * v_max  # \beta_1 ~ k_2; \beta_2 < \beta_1
        v[2 * M] = u[2 * M] * 0.05  # \alpha_0 ~ V_p
        v[T] = u[T] * SpectralAnalysis.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def signalmodel(data: dict, verbose=False):
        M = SpectralAnalysis.M
        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        input_func_interp1 = np.append(input_func_interp, input_func_interp[-1])
        v = data["v"]
        a = np.zeros(M)
        b = np.zeros(M)
        for m in np.arange(M):
            a[m] = v[2 * m]
            b[m] = v[2 * m + 1]
        a_0 = v[2 * M]
        t_0 = SpectralAnalysis.T_0
        tau_a = SpectralAnalysis.TAU_A

        n = len(input_func_interp)
        times = np.arange(0, n * data["delta_time"], data["delta_time"])
        times1 = np.arange(0, (n + 1) * data["delta_time"], data["delta_time"])
        if tau_a != 0:
            input_func_interp = SpectralAnalysis.slide(input_func_interp, times, tau_a, None)

        # rho_t is the inferred source signal
        rho_t = 0
        for m in np.arange(M):
            rho_t = rho_t + np.convolve(a[m] * np.exp(-b[m] * times), input_func_interp, mode="full")
        rho_t = rho_t[:n] + a_0 * integrate.cumulative_trapezoid(input_func_interp1, times1)
        rho_t = SpectralAnalysis.slide(rho_t, times, t_0, None)

        if data["rhoUsesBoxcar"]:
            rho = Boxcar.apply_boxcar(rho_t, data)
        else:
            rho = np.interp(timesMid, times, rho_t)

        if verbose:
            return rho, timesMid, rho_t, times
        else:
            return rho, timesMid, rho_t, times
