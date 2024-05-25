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

from TCModel import TCModel
from Boxcar import Boxcar

# general & system functions
import os

# basic numeric setup
import numpy as np
import scipy.integrate as integrate


class Ichise2002VascModel(TCModel):
    """

    This class is a subclass of the TCModel class and represents the Ichise2002 model for PET data analysis.

    Attributes:
    - input_function: The input function of the model.
    - pet_measurement: The PET measurement data.
    - truths: A dictionary containing the ground truth values for the model parameters. (optional)
    - home: The home directory where the output files will be saved. (default: current working directory)
    - sample: The type of sampling to be used. (default: "rslice")
    - nlive: The number of live points for the Nested Sampling algorithm. (default: 1000)
    - rstate: The random state for reproducibility. (default: np.random.default_rng(916301))
    - tag: A tag for identifying the model instance. (default: "")

    Methods:
    - labels: Returns the labels for the model parameters.
    - signalmodel: Computes the signal model for the given data.

    """
    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 time_last=None,
                 tag="",
                 delta_time=1):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         time_last=time_last,
                         tag=tag,
                         delta_time=delta_time)

    @property
    def labels(self):
        return [
            r"$K_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$V_P$", r"$V^*$", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):

        rho = data["rho"]
        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        v = data["v"]
        n = len(input_func_interp)
        tf_interp = n * data["delta_time"]
        times = np.arange(0, tf_interp, data["delta_time"])
        ks = v[:4]

        K1 = ks[0]
        VP = v[4]
        Vstar = v[5]  # total volume of distribution V^\star = V_P + V_N + V_S, per Ichise 2002 appendix
        g1 = Vstar * ks[1] * ks[3]
        g2 = -1 * ks[1] * ks[3]
        g3 = -(ks[1] + ks[2] + ks[3])
        g4star = K1
        g5 = VP
        t_0 = v[6]
        tau_a = v[7]

        # rho_t is the inferred source signal

        rho_t = np.zeros(len(timesMid))
        rho_p = Ichise2002VascModel.slide(input_func_interp, times, tau_a, None)
        timesMid1 = timesMid[1:]
        for tidx, tMid in enumerate(timesMid1):
            _m = rho[:tidx]
            _t = timesMid[:tidx]
            _times = np.arange(min(tMid, n - 1))  # integration interval
            _times_int = _times.astype(int)  # integration interval cast as integer
            _rho_p_times_int = rho_p[_times_int]  # integration interval

            int3 = integrate.trapezoid(_m, _t)
            int4 = integrate.trapezoid(_rho_p_times_int, _times)
            if len(_t[:-1]) > 0:
                int2 = 0.5 * integrate.trapezoid(integrate.cumulative_trapezoid(_m, _t), _t[:-1])
            else:
                int2 = 0
            int1 = 0.5 * integrate.trapezoid(integrate.cumulative_trapezoid(_rho_p_times_int, _times), _times[:-1])

            rho_t[tidx] = g1 * int1 + g2 * int2 + g3 * int3 + g4star * int4 + g5 * rho_p[_times_int[-1]]

        rho_t[rho_t < 0] = 0
        rho_t = Ichise2002VascModel.slide(rho_t, times, t_0, None)
        if data["rhoUsesBoxcar"]:
            rho = Boxcar.apply_boxcar(rho_t, data)
        else:
            rho = np.interp(timesMid, times, rho_t)
        return rho, timesMid, rho_t, times

    @staticmethod
    def volume_specific(data: dict):

        v = data["v"]
        ks = v[:4]

        K1 = ks[0]
        VP = v[4]
        Vstar = v[5]  # total volume of distribution V^\star = V_P + V_N + V_S, per Ichise 2002 appendix
        g1 = Vstar * ks[1] * ks[3]
        g2 = -1 * ks[1] * ks[3]
        g3 = -(ks[1] + ks[2] + ks[3])
        g4star = K1
        g5 = VP

        numer = g1*(g1 + g3*g4star) + g2*(g4star + g3*g5)^2
        denom = g2*(g1 + g3*g4star)
        return numer/denom

    @staticmethod
    def volume_nonspecific(data: dict):

        v = data["v"]
        VP = v[4]
        Vstar = v[5]
        VS = Ichise2002VascModel.volume_specific(data)

        return Vstar - VP - VS

    @staticmethod
    def volume_plasma(data: dict):

        v = data["v"]
        return v[4]

    @staticmethod
    def volume_total(data: dict):

        v = data["v"]
        return v[5]

