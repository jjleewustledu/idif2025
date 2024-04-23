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


class Huang1980ModelVenous(TCModel):
    """
    This class implements the Huang1980 model for venous blood region in PET analysis.

    It inherits from the `TCModel` class.

    Attributes:
    - input_function (function): The input function used in the model.
    - pet_measurement (array-like): The PET measurement data.
    - truths (dict, optional): The ground truth values for the model parameters. Default is None.
    - home (str, optional): The directory to save the analysis results. Default is the current working directory.
    - sample (str, optional): The sample name for saving files. Default is "rslice".
    - nlive (int, optional): The number of live points for the nested sampling algorithm. Default is 1000.
    - rstate (MersenneTwister or PCG64, optional): The random number generator state. Default is numpy's default RNG with seed 916301.
    - tag (str, optional): Additional tag for saving files. Default is empty string.
    - venous_recovery_coefficient (int or float, optional): Recovery coefficient for the venous blood region. Default is 1.

    Methods:
    - labels() -> list: Returns the labels for the model parameters.
    - signalmodel(data: dict) -> tuple: Computes the model signal using the given data and returns the results.

    """
    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag="",
                 venous_recovery_coefficient=1):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         time_last=None,
                         tag=tag)

        # self.RECOVERY_COEFFICIENT = self.RECOVERY_COEFFICIENT * venous_recovery_coefficient

    @property
    def labels(self):
        return [
            r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):

        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        v1 = data["martinv1"]
        v = data["v"]

        k1 = v[0]
        k2 = v[1]
        k3 = v[2]
        k4 = v[3]
        t_0 = v[4]
        tau_a = v[5]
        n = len(input_func_interp)
        times = np.arange(n)
        input_func_interp = Huang1980ModelVenous.slide(input_func_interp, times, tau_a, None)

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

        rho_t = v1 * (input_func_interp + q2 + q3)
        rho_t = Huang1980ModelVenous.slide(rho_t, times, t_0, None)
        rho = np.interp(timesMid, times, rho_t)
        return rho, timesMid, rho_t, times
