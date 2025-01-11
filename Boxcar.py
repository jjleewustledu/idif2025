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
from Artery import Artery
from IOImplementations import BaseIO

# basic numeric setup
import numpy as np
from numba import njit
#from six.moves import zip


class Boxcar(Artery):
    """
    Boxcar class extends the Artery class and represents a boxcar signal model.

    Parameters
    ----------
    input_func_measurement (dict): The measurement data for the input function.
    tracer (Trajectory, optional): The tracer trajectory.
    truths (dict, optional): The true values for the parameters.
    sample (str, optional): The sampling method. Default is "rslice".
    nlive (int, optional): The number of live points for nested sampling. Default is 1000.
    rstate (numpy.random.SeedSequence, optional): The random state for sampling. Default is np.random.default_rng(916301).
    tag (str, optional): The tag for the Boxcar instance. Default is "".

    Methods
    -------
    signalmodel(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Computes the signal and ideal values of the model.

    apply_boxcar(vec, data: dict) -> np.ndarray
        Applies the boxcar transformation to the vector.

    """
    def __init__(self, input_func_measurement,
                 tracer=None,
                 truths=None,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 times_last=None,
                 tag=""):
        super().__init__(input_func_measurement,
                         tracer=tracer,
                         truths=truths,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         times_last=times_last,
                         tag=tag)

        Artery.sigma = 0.01

    @staticmethod
    def signalmodel(data: dict):
        t_interp = BaseIO.data2tinterp(data)
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

        # rho_ = A * Boxcar.solution_1bolus(t_interp, t_0, a, b, p)
        # rho_ = A * Boxcar.solution_2bolus(t_interp, t_0, a, b, p, g, f_ss)
        # rho_ = A * Boxcar.solution_3bolus(t_interp, t_0, tau_2, a, b, p, dp_2, g, f_2, f_ss)
        # rho_ = A * Boxcar.solution_4bolus(t_interp, t_0, tau_2, tau_3, a, b, p, dp_2, dp_3, g, f_2, f_3, f_ss)
        rho_ = A * Boxcar.solution_3bolus_series(t_interp, t_0, tau_2, tau_3, a, b, p, dp_2, dp_3, g, f_2, f_3)
        rho = Boxcar.apply_boxcar(rho_, data)
        A_qs = 1 / max(rho)
        signal = A_qs * rho
        ideal = A_qs * rho_
        return signal, ideal, t_interp

    @staticmethod
    def apply_boxcar(vec, data: dict):
        times0_int = (data["timesMid"] - data["taus"] / 2).astype(int)
        timesF_int = (data["timesMid"] + data["taus"] / 2).astype(int)

        # Original implementation with loop
        # vec_sampled = np.full(times0.shape, np.nan)
        # for idx, (t0, tF) in enumerate(zip(times0, timesF)):
        #     vec_sampled[idx] = np.mean(vec[int(t0):int(tF)])        
        # return np.nan_to_num(vec_sampled, 0)

        # Optimized implementation using cumsum, padding vec with 0 at beginning
        cumsum = np.cumsum(np.pad(vec, (1, 0)))
        vec_sampled = (cumsum[timesF_int + 1] - cumsum[times0_int]) / data["taus"]
        return np.nan_to_num(vec_sampled, 0)
