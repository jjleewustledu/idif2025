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

# general & system functions
import os

# basic numeric setup
import numpy as np
import scipy.integrate as integrate


class Ichise2002Model(TCModel):

    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag=""):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         tag=tag)

    @property
    def labels(self):
        return [
            r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$V_P$", r"$V_N + V_S$", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):

        rho = data["rho"]
        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        v1 = data["martinv1"]
        v = data["v"]
        n = len(input_func_interp)
        times = np.arange(n)

        ks = v[:6]
        VP = ks[4]
        K1 = ks[0] * VP
        Vstar = VP + ks[5]
        g1 = Vstar * ks[1] * ks[3]
        g2 = -1 * ks[1] * ks[3]
        g3 = -(ks[1] + ks[2] + ks[3])
        g4star = K1
        g5 = VP
        t_0 = v[6]
        tau_a = v[7]

        _rho = np.zeros(len(timesMid))
        timesMid1 = timesMid[1:]
        _rho_p = input_func_interp * min(VP / v1, 1)
        _rho_p = Ichise2002Model.slide(_rho_p, times, tau_a, None)
        for tidx, tMid in enumerate(timesMid1):
            _m = rho[:tidx]
            _t = timesMid[:tidx]
            _times = np.arange(min(tMid, n - 1))  # integration interval
            _times_int = _times.astype(int)
            _rho_p_times_int = _rho_p[_times_int]  # integration interval

            int3 = integrate.trapezoid(_m, _t)
            int4 = integrate.trapezoid(_rho_p_times_int, _times)
            if len(_t[:-1]) > 0:
                int2 = 0.5 * integrate.trapezoid(integrate.cumulative_trapezoid(_m, _t), _t[:-1])
            else:
                int2 = 0
            int1 = 0.5 * integrate.trapezoid(integrate.cumulative_trapezoid(_rho_p_times_int, _times), _times[:-1])

            _rho[tidx] = g1 * int1 + g2 * int2 + g3 * int3 + g4star * int4 + g5 * _rho_p[_times_int[-1]]

        _rho[_rho < 0] = 0
        _rho = Ichise2002Model.slide(_rho, timesMid, t_0, None)
        rho = _rho
        rho_t = np.interp(times, timesMid, _rho)
        return rho, timesMid, rho_t, times
