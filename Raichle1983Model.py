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
from TCModel import TCModel
from Boxcar import Boxcar

# general & system functions
import os

# basic numeric setup
import numpy as np


class Raichle1983Model(TCModel):
    """
    The Raichle1983Model class is a subclass of the TCModel class that implements the signal model proposed by Raichle et al. in 1983.

    Args:
        input_function (function): The input function that describes the tracer kinetics.
        pet_measurement (dict): The PET measurement data.
        truths (dict, optional): The ground truth values for the parameters. Defaults to None.
        home (str, optional): The HOME directory. Defaults to the current working directory.
        sample (str, optional): The sampling method. Defaults to "rslice".
        nlive (int, optional): The number of live points for sampling. Defaults to 1000.
        rstate (numpy.random.Generator, optional): The random state. Defaults to np.random.default_rng(916301).
        tag (str, optional): A tag for identification. Defaults to "".

    Attributes:
        labels (list): The labels for the parameters.
    """
    def __init__(self, input_function, pet_measurement, **kwargs):
        super().__init__(input_function, pet_measurement, **kwargs)

    @property
    def labels(self):
        return [
            r"$f$", r"$\lambda$", r"ps", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):

        hl = data["halflife"]
        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        # v1 = data["martinv1"]  # inconsistent with data?
        v = data["v"]
        f = v[0]
        lamb = v[1]
        ps = v[2]
        t_0 = v[3]
        tau_a = v[4]
        E = 1 - np.exp(-ps / f)
        n = len(input_func_interp) * data["delta_time"]
        tf_interp = n * data["delta_time"]
        times = np.arange(0, tf_interp, data["delta_time"])
        kernel = np.exp(-E * f * times / lamb - 0.005670305 * times)
        input_func_interp = Raichle1983Model.slide(input_func_interp, times, tau_a, hl)
        rho_t = E * f * np.convolve(kernel, input_func_interp, mode="full")
        rho_t = rho_t[:input_func_interp.size]  # + v1 * input_func_interp  # inconsistent with data?
        rho_t = Raichle1983Model.slide(rho_t, times, t_0, hl)
        if data["rho_experiences_boxcar"]:
            rho = Boxcar.apply_boxcar(rho_t, data)
        else:
            rho = np.interp(timesMid, times, rho_t)
        return rho, timesMid, rho_t, times
