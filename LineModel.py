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

# general & system functions
import os

# basic numeric setup
import numpy as np


class LineModel(TCModel):
    """ """
    def __init__(self, input_function, pet_measurement, **kwargs):
        kwargs["time_last"] = None
        super().__init__(input_function, pet_measurement, **kwargs)

    @property
    def labels(self):
        return [
            r"intercept", r"slope", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict, verbose=False):

        timesMid = data["timesMid"]
        input_func_interp = data["rhoInputFuncInterp"]
        v = data["v"]

        intercept = v[0]
        slope = v[1]
        n = len(input_func_interp)
        tf_interp = n * data["delta_time"]
        times = np.arange(0, tf_interp, data["delta_time"])

        # rho_t is the inferred source signal

        rho_t = intercept + slope * times
        rho = np.interp(timesMid, times, rho_t)
        if verbose:
            return rho, timesMid, rho_t, times, input_func_interp, input_func_interp
        else:
            return rho, timesMid, rho_t, times

