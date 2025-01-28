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

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from DynestyData import DynestyData
from PETUtilities import PETUtilities

class InputFuncData(DynestyData):
    """ input function data assumed to be decay-corrected, consistent with tomography """
    
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)

        assert "input_func_fqfn" in self.data_dict, "data_dict missing required key 'input_func_fqfn'"
      
        niid = self.nii_load(self.data_dict["input_func_fqfn"])
        self._data_dict["halflife"] = niid["halflife"]
        self._data_dict["rho"] = niid["img"] / np.max(niid["img"])
        self._data_dict["timesMid"] = niid["timesMid"]
        self._data_dict["taus"] = niid["taus"]
        self._data_dict["times"] = niid["times"]
        self._data_dict["sigma"] = 0.1

        if "sample" not in self._data_dict:
            self._data_dict["sample"] = "rslice"
        if "nlive" not in self._data_dict:
            self._data_dict["nlive"] = 300
        if "rstate" not in self._data_dict:
            self._data_dict["rstate"] = np.random.default_rng(916301)
        if "tag" not in self._data_dict:
            self._data_dict["tag"] = ""
        if "pfrac" not in self._data_dict:
            self._data_dict["pfrac"] = 1.0

    @property
    def input_func_fqfn(self) -> str:
        return self.data_dict["input_func_fqfn"]

    @property
    def input_func_measurement(self) -> dict:
        if hasattr(self, "__input_func_measurement"):
            return deepcopy(self.__input_func_measurement)

        self.__input_func_measurement = self.context.io.nii_load(self.input_func_fqfn)
        return deepcopy(self.__input_func_measurement)

    @property
    def halflife(self) -> float:
        return self.data_dict["halflife"]
    
    @property
    def nlive(self) -> int:
        return self.data_dict["nlive"]
    
    @property
    def pfrac(self) -> float:
        return self.data_dict["pfrac"]
    
    @property
    def rstate(self) -> np.random.Generator:
        return self.data_dict["rstate"]
    
    @property
    def sample(self) -> str:
        return self.data_dict["sample"]

    @property
    def rho(self) -> NDArray:
        return self.data_dict["rho"].copy()

    @property
    def sigma(self) -> float:
        return self.data_dict["sigma"]

    @property
    def tag(self) -> str:
        return self.data_dict["tag"]
    
    @tag.setter
    def tag(self, tag: str):
        self._data_dict["tag"] = tag

    @property
    def taus(self) -> NDArray:
        return self.data_dict["taus"].copy()

    @property
    def timesIdeal(self) -> NDArray:
        if hasattr(self, "__timesIdeal"):
            return deepcopy(self.__timesIdeal)
        
        self.__timesIdeal = PETUtilities.data2timesInterp(self.data_dict)
        return deepcopy(self.__timesIdeal)

    @property
    def timesMid(self) -> NDArray:
        return self.data_dict["timesMid"].copy()
