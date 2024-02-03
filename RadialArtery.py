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

# system functions
import time, sys, os

# basic numeric setup
import numpy as np

# NIfTI support
import nibabel as nib

from Artery import Artery

KERNEL = None
RHO = None
TAUS = None
TIMES_MID = None

__all__ = ["RadialArtery"]


class RadialArtery(Artery):

    def __init__(self, input_func_measurement,
                 kernel_measurement,
                 remove_baseline=True,
                 tracer=None,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        super().__init__(input_func_measurement,
                         remove_baseline=remove_baseline,
                         tracer=tracer,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate)

        self.__kernel_measurement = kernel_measurement

        global KERNEL, RHO, TAUS, TIMES_MID
        km = self.kernel_measurement
        KERNEL = km["img"]
        ifm = self.input_func_measurement
        RHO = ifm["img"] / np.max(ifm["img"])
        TAUS = ifm["taus"]
        TIMES_MID = ifm["timesMid"]

    @property
    def kernel_measurement(self):
        if isinstance(self.__kernel_measurement, dict):
            return self.__kernel_measurement

        assert os.path.isfile(self.__kernel_measurement), f"{self.__kernel_measurement} was not found."
        fqfn = self.__kernel_measurement
        base, ext = os.path.splitext(fqfn)
        fqfp = os.path.splitext(base)[0]

        # load img
        nii = nib.load(fqfn)
        img = nii.get_fdata()

        # assemble dict
        self.__input_func_measurement = {
            "fqfp": fqfp,
            "img": np.array(img, dtype=float).reshape(-1)}
        return self.__input_func_measurement

    @staticmethod
    def data(v):
        return {"timesMid": TIMES_MID, "taus": TAUS, "v": v, "kernel": KERNEL}

    @staticmethod
    def loglike(v):
        data = RadialArtery.data(v)
        rho_pred, _, _ = RadialArtery.signalmodel(data)
        sigma = v[-1]
        residsq = (rho_pred - RHO) ** 2 / sigma ** 2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))

        if not np.isfinite(loglike):
            loglike = -1e300

        return loglike

    @staticmethod
    def signalmodel(data: dict):
        t_ideal = np.arange(RHO.size)
        v = data["v"]
        t_0 = v[0]
        tau_2 = v[1]
        tau_3 = v[2]
        a = v[3]
        b = 1 / v[4]
        p = v[5]
        dp_2 = v[6]
        dp_3 = v[7]
        g = 1 / v[8]
        f_2 = v[9]
        f_3 = v[10]
        f_ss = v[11]
        A = v[12]

        #rho_ = A * RadialArtery.solution_1bolus(t_ideal, t_0, a, b, p)
        #rho_ = A * RadialArtery.solution_2bolus(t_ideal, t_0, a, b, p, g, f_ss)
        #rho_ = A * RadialArtery.solution_3bolus(t_ideal, t_0, tau_2, a, b, p, dp_2, g, f_2, f_ss)
        rho_ = A * RadialArtery.solution_4bolus(t_ideal, t_0, tau_2, tau_3, a, b, p, dp_2, dp_3, g, f_2, f_3, f_ss)
        rho = RadialArtery.apply_dispersion(rho_, data)
        A_qs = 1 / max(rho)
        signal = A_qs * rho
        ideal = A_qs * rho_
        return signal, ideal, t_ideal

    @staticmethod
    def apply_dispersion(vec, data: dict):
        k = data["kernel"]
        k = k.T
        vec_sampled = np.convolve(vec, k, mode="full")
        return vec_sampled[:vec.size]