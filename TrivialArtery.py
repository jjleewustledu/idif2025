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
from __future__ import print_function
from abc import ABC
from PETModel import PETModel
from dynesty import utils as dyutils

# general & system functions
import re
import os
from copy import deepcopy
import pickle

# basic numeric setup
import numpy as np
import pandas as pd


class TrivialArtery(PETModel, ABC):
    """

    """
    def __init__(self,
                 input_func_measurement,
                 kernel_measurement=None,
                 remove_baseline=False,
                 tracer=None,
                 truths=None,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 times_last=None,
                 tag=""):
        super().__init__(sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         time_last=times_last,
                         tag=tag)

        self._truths_internal = truths

        self.__input_func_measurement = input_func_measurement  # set with fqfn
        ifm = self.input_func_measurement  # get dict conforming to nibabel
        self.HALFLIFE = ifm["halflife"]
        self.RHO = ifm["img"] / np.max(ifm["img"])
        self.TAUS = ifm["taus"]
        self.TIMES_MID = ifm["timesMid"]

        self.tracer = tracer
        if not self.tracer:
            regex = r"_trc-(.*?)_"
            matches = re.findall(regex, input_func_measurement)
            assert matches, "no tracer information in input_func_measurement"
            self.tracer = matches[0]
            print(f"{self.__class__.__name__}: found data for tracer {self.tracer}")

        self.__remove_baseline = remove_baseline

    @property
    def fqfp(self):
        return self.input_func_measurement["fqfp"]

    @property
    def fqfp_results(self):
        return self.fqfp + "-" + self.__class__.__name__

    @property
    def input_func_measurement(self):
        if isinstance(self.__input_func_measurement, dict):
            return deepcopy(self.__input_func_measurement)

        assert os.path.isfile(self.__input_func_measurement), f"{self.__input_func_measurement} was not found."
        fqfn = self.__input_func_measurement
        self.__input_func_measurement = self.load_nii(fqfn)
        return deepcopy(self.__input_func_measurement)

    @property
    def labels(self):
        return None

    @property
    def ndim(self):
        return None

    @property
    def truths(self):
        return self._truths_internal

    def data(self, v):
        pass

    def loglike(self, v):
        pass

    def plot_truths(self, truths, parc_index):
        pass

    def plot_variations(self, tindex0, tmin, tmax, truths):
        pass

    def run_nested(self, checkpoint_file):
        pass

    def save_results(self, res, tag):
        pass

    @staticmethod
    def prior_transform(tag):
        pass

    @staticmethod
    def signalmodel(data: dict):
        pass
