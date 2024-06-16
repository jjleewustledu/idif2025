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

from SpectralAnalysis import SpectralAnalysis
from Huang1980Model import Huang1980Model
from Ichise2002Model import Ichise2002Model
from Ichise2002PosthocModel import Ichise2002PosthocModel
from Ichise2002VascModel import Ichise2002VascModel
from LineModel import LineModel

# general & system functions
import os

# basic numeric setup
import numpy as np


class TZ3108:
    """  """

    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag="",
                 model=None,
                 delta_time=4,
                 M=1):
        if "SpectralAnalysis" in model:
            self._strategy = SpectralAnalysis(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=delta_time,
                M=M)
        elif "Ichise2002PosthocModel" in model:
            self._strategy = Ichise2002PosthocModel(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=delta_time)
        elif "Ichise2002VascModel" in model:
            self._strategy = Ichise2002VascModel(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=delta_time)
        elif "Ichise2002Model" in model:
            self._strategy = Ichise2002Model(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=delta_time)
        elif "Huang1980Model" in model:
            self._strategy = Huang1980Model(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=1)
        elif "LineModel" in model:
            self._strategy = LineModel(
                input_function,
                pet_measurement,
                truths=truths,
                home=home,
                sample=sample,
                nlive=nlive,
                rstate=rstate,
                tag=tag,
                delta_time=delta_time)
        else:
            raise RuntimeError(self.__class__.__name__ + ".__init__() does not support model -> " + model)

    @property
    def fqfp(self):
        return self._strategy.fqfp

    @property
    def fqfp_results(self):
        return self._strategy.fqfp_results

    @property
    def labels(self):
        return self._strategy.labels

    @property
    def ndim(self):
        return self._strategy.ndim

    @property
    def pet_measurement(self):
        return self._strategy.adjusted_pet_measurement

    @property
    def sigma(self):
        return self._strategy.sigma
 
    @property
    def solver(self):
        return self._strategy.solver

    @property
    def truths(self):
        return self._strategy.truths

    def data(self, v):
        return self._strategy.data(v)

    def data2t(self, data):
        return self._strategy.data2t(data)

    def data2taus(self, data):
        return self._strategy.data2taus(data)

    def data2timesMid(self, data):
        return self._strategy.data2timesMid(data)

    def input_function(self):
        return self._strategy.adjusted_input_function()

    def load_nii(self, fqfn):
        return self._strategy.load_nii(fqfn)

    def loglike(self, v):
        return self._strategy.loglike(v)

    def parse_halflife(self, fqfp):
        return self._strategy.parse_halflife(fqfp)

    def parse_isotope(self, fqfp):
        return self._strategy.parse_isotope(fqfp)

    def plot_results(self, *args, **kwargs):
        return self._strategy.plot_results(*args, **kwargs)

    def plot_truths(self, *args, activity_units="kBq/cm$^3$", **kwargs):
        return self._strategy.plot_truths(*args, activity_units=activity_units, **kwargs)

    def plot_variations(self, *args, **kwargs):
        return self._strategy.plot_variations(*args, **kwargs)

    def run_nested(self, *args, **kwargs):
        return self._strategy.run_nested(*args, **kwargs)

    def run_nested_for_indexed_tac(self, *args, **kwargs):
        return self._strategy.run_nested_for_indexed_tac(*args, **kwargs)

    def save_csv(self, *args, **kwargs):
        return self._strategy.save_csv(*args, **kwargs)

    def save_nii(self, *args, **kwargs):
        return self._strategy.save_nii(*args, **kwargs)

    def save_res_dict(self, *args, **kwargs):
        return self._strategy.pickle_results(*args, **kwargs)

    def save_results(self, *args, **kwargs):
        return self._strategy.save_results(*args, **kwargs)

    def signalmodel(self, *args, **kwargs):
        return self._strategy.signalmodel(*args, **kwargs)
