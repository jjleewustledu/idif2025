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


class Huang1980Model(TCModel):
    """

    The `Huang1980Model` class is a subclass of `TCModel` and represents a model for PET (Positron Emission Tomography) data based on the Huang et al. (1980) model.

    Attributes:
        - `input_function`: The input function used in the model.
        - `pet_measurement`: The PET measurement data.
        - `truths`: An optional argument representing the ground truth data for comparison.
        - `HOME`: The HOME directory path.
        - `sample`: The sampling method. Default is "rslice".
        - `nlive`: The number of live points. Default is 1000.
        - `rstate`: The random state used for sampling. Default is np.random.default_rng(916301).
        - `tag`: An optional tag for identifying the model.

    Properties:
        - `labels`: Returns a list of labels for the parameters in the model.

    Methods:
        - `signalmodel(data: dict)`: Calculates and returns the signal model for the given data.

    Note: This class does not define any additional methods or attributes beyond those inherited from the `TCModel` class.

    """
    def __init__(self, input_function, pet_measurement, **kwargs):
        kwargs.setdefault('time_last', None)
        super().__init__(input_function, pet_measurement, **kwargs)

    @property
    def labels(self):
        return [
            r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict, verbose=False):

        timesMid = data["timesMid"]
        input_func_interp = data["rhoInputFuncInterp"]
        v1 = data["martinv1"]
        v = data["v"]

        k1 = v[0]
        k2 = v[1]
        k3 = v[2]
        k4 = v[3]
        t_0 = v[4]
        tau_a = v[5]
        n = len(input_func_interp)
        tf_interp = n * data["delta_time"]
        times = np.arange(0, tf_interp, data["delta_time"])
        input_func_interp = Huang1980Model.slide(input_func_interp, times, tau_a, None)

        # use k1:k4
        k234 = k2 + k3 + k4
        bminusa = np.sqrt(np.power(k234, 2) - 4 * k2 * k4)
        alpha = 0.5 * (k234 - bminusa)
        beta = 0.5 * (k234 + bminusa)
        conva = np.convolve(np.exp(-alpha * times), input_func_interp, mode="full")
        convb = np.convolve(np.exp(-beta * times), input_func_interp, mode="full")
        conva = conva[:n]
        convb = convb[:n]
        conv2 = (k4 - alpha) * conva + (beta - k4) * convb
        conv3 = conva - convb
        q2 = (k1 / bminusa) * conv2
        q3 = (k3 * k1 / bminusa) * conv3

        # rho_t is the inferred source signal

        rho_t = v1 * (input_func_interp + q2 + q3)
        rho_t = Huang1980Model.slide(rho_t, times, t_0, None)
        if data["rho_experiences_boxcar"]:
            rho = Boxcar.apply_boxcar(rho_t, data)
        else:
            rho = np.interp(timesMid, times, rho_t)
        if verbose:
            return rho, timesMid, rho_t, times, v1 * (q2 + q3), v1 * input_func_interp
        else:
            return rho, timesMid, rho_t, times
