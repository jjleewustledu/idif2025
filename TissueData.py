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
from IOImplementations import TissueIO
from PETUtilities import PETUtilities

class TissueData(DynestyData):
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)

        assert "input_func_fqfn" in self.data_dict, "data_dict missing required key 'input_func_fqfn'"
        assert "tissue_fqfn" in self.data_dict, "data_dict missing required key 'tissue_fqfn'"
        assert "v1_fqfn" in self.data_dict, "data_dict missing required key 'v1_fqfn'"
        assert "ks_fqfn" in self.data_dict, "data_dict missing required key 'ks_fqfn'"
   
        niid = self.context.io.load_nii(self.data_dict["tissue_fqfn"])
        self.data_dict["halflife"] = niid["halflife"]
        self.data_dict["timesMid"] = niid["timesMid"]
        self.data_dict["taus"] = niid["taus"]
        self.data_dict["times"] = niid["times"]
        self.data_dict["sigma"] = 0.1

        if "sample" not in self._data_dict:
            self._data_dict["sample"] = "rslice"
        if "nlive" not in self._data_dict:
            self._data_dict["nlive"] = 300
        if "rstate" not in self._data_dict:
            self._data_dict["rstate"] = np.random.default_rng(916301)
        if "tag" not in self._data_dict:
            self._data_dict["tag"] = ""
        if "recovery_coefficient" not in self._data_dict:
            self._data_dict["recovery_coefficient"] = 1.8509
        if "delta_time" not in self._data_dict:
            self._data_dict["delta_time"] = 1

    @property
    def adjusted_input_func_measurement(self) -> dict:
        if hasattr(self._data_dict, "adjusted_input_func_measurement"):
            return deepcopy(self._data_dict["adjusted_input_func_measurement"])

        # normalize to max tissue measurement
        adj_input_func_meas = self.input_func_measurement
        adj_input_func_meas["img"] = adj_input_func_meas["img"] / self.max_tissue_measurement
        
        # apply recovery coefficient to Boxcar
        if self.input_func_type == "Boxcar":
            adj_input_func_meas["img"] = \
                adj_input_func_meas["img"] * self.recovery_coefficient

        # interpolate to tissue times
        adj_input_func_meas = PETUtilities.interpdata(
            data=adj_input_func_meas,
            ref=self.__adjusted_tissue_measurement)

        self._data_dict["adjusted_input_func_measurement"] = adj_input_func_meas
        return deepcopy(self._data_dict["adjusted_input_func_measurement"])
    
    @property
    def adjusted_tissue_measurement(self) -> dict:
        if hasattr(self._data_dict, "adjusted_tissue_measurement"):
            return deepcopy(self._data_dict["adjusted_tissue_measurement"])

        # normalize to max tissue measurement
        adj_tiss_meas = self.tissue_measurement
        adj_tiss_meas["img"] = adj_tiss_meas["img"] / self.max_tissue_measurement
        
        self._data_dict["adjusted_tissue_measurement"] = adj_tiss_meas
        return deepcopy(self._data_dict["adjusted_tissue_measurement"])

    @property
    def delta_time(self) -> float:
        return self.data_dict["delta_time"]

    @property
    def halflife(self) -> float:
        return self.data_dict["halflife"]

    @property
    def input_func_fqfn(self) -> str:
        return self.data_dict["input_func_fqfn"]

    @property
    def input_func_interp(self) -> NDArray:
        ifm = PETUtilities.interpdata(
            data=self.input_func_measurement,
            ref=self.adjusted_tissue_measurement)
        return ifm["img"].copy()
    
    @property
    def input_func_measurement(self) -> dict:
        """ assumed to be decay-corrected, which is hereby uncorrected for[15O] """
        if hasattr(self._data_dict, "input_func_measurement"):
            return deepcopy(self._data_dict["input_func_measurement"])

        input_func_meas = self.load_nii(self.input_func_fqfn)
        if PETUtilities.parse_isotope(self.input_func_fqfn) == "15O":
            input_func_meas = self.decay_uncorrect(input_func_meas)
        self._data_dict["input_func_measurement"] = input_func_meas
        return deepcopy(self._data_dict["input_func_measurement"])

    @property
    def input_func_type(self) -> str:
        _, bname = TissueIO.fileparts(self.input_func_fqfn)
        if "MipIdif" in bname or "Boxcar" in bname:
            return "Boxcar"
        if "TwiliteKit" in bname or "RadiaArtery" in bname:
            return "RadialArtery"        
        raise ValueError(f"input_func_fqfn {self.input_func_fqfn} not recognized as input function")

    @property
    def isidif(self) -> bool:
        return self.input_func_type == "Boxcar"
    
    @property   
    def ks(self) -> NDArray:
        return self.ks_measurement["img"].copy()

    @property
    def ks_measurement(self) -> dict:
        """ adjusts for recovery coefficient if RadialArtery used """
        if hasattr(self._data_dict, "ks_measurement"):
            return deepcopy(self._data_dict["ks_measurement"])

        self._data_dict["ks_measurement"] = self.load_nii(self._data_dict["ks_fqfn"])
        return deepcopy(self._data_dict["ks_measurement"])

    @property
    def max_tissue_measurement(self) -> float:
        """ normalized measured activities to rho """
        return np.max(self.tissue_measurement["img"])
    
    @property
    def nlive(self) -> int:
        return self.data_dict["nlive"]

    @property
    def recovery_coefficient(self) -> float:
        return self.data_dict["recovery_coefficient"]
    
    @property
    def rho(self) -> NDArray:
        return self.adjusted_tissue_measurement["img"].copy()
    
    @property
    def rstate(self) -> np.random.Generator:
        return self.data_dict["rstate"]
    
    @property
    def sample(self) -> str:
        return self.data_dict["sample"]
    
    @property
    def sigma(self) -> float:
        return self.data_dict["sigma"]
    
    @property
    def tag(self) -> str:
        return self.data_dict["tag"]
    
    @tag.setter
    def tag(self, tag: str):
        self.data_dict["tag"] = tag

    @property
    def taus(self) -> NDArray:
        return self.data_dict["taus"].copy()

    @property
    def times(self) -> NDArray:
        return self.data_dict["times"].copy()

    @property
    def timesIdeal(self) -> NDArray:
        if hasattr(self, "__timesIdeal"):
            return deepcopy(self.__timesIdeal)
        
        self.__timesIdeal = PETUtilities.data2timesInterp(self.data_dict)
        return deepcopy(self.__timesIdeal)

    @property
    def timesMid(self) -> NDArray:
        return self.data_dict["timesMid"].copy()
    
    @property
    def tissue_fqfn(self) -> str:
        return self.data_dict["tissue_fqfn"]

    @property
    def tissue_measurement(self) -> dict:
        """ assumed to be decay-corrected, which is hereby uncorrected for [15O] """
        if hasattr(self._data_dict, "tissue_measurement"):
            return deepcopy(self._data_dict["tissue_measurement"])

        tiss_meas = self.load_nii(self.tissue_fqfn)
        if PETUtilities.parse_isotope(self.tissue_fqfn) == "15O":
            tiss_meas = self.decay_uncorrect(tiss_meas)
        self._data_dict["tissue_measurement"] = tiss_meas
        return deepcopy(self._data_dict["tissue_measurement"])
    
    @property
    def v1(self) -> NDArray:
        return self.v1_measurement["img"].copy()

    @property
    def v1_measurement(self) -> dict:
        """ adjusts for recovery coefficient if RadialArtery used """
        if hasattr(self._data_dict, "v1_measurement"):
            return deepcopy(self._data_dict["v1_measurement"])

        self._data_dict["v1_measurement"] = self.load_nii(self._data_dict["v1_fqfn"])
        if self.input_func_type == "RadialArtery":
            self._data_dict["v1_measurement"]["img"] = self._data_dict["v1_measurement"]["img"] / self.recovery_coefficient
        return deepcopy(self._data_dict["v1_measurement"])
