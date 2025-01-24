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
import nibabel as nib
from DynestyData import DynestyData
from IOImplementations import TissueIO
from PETUtilities import PETUtilities

class TissueData(DynestyData):
    """ Requires all incoming PET and input function data to be decay corrected,
        and immediately decay uncorrects [15O] data on loading so that kinetics and decays will commute. """
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)

        assert "input_func_fqfn" in self.data_dict, "data_dict missing required key 'input_func_fqfn'"
        assert "tissue_fqfn" in self.data_dict, "data_dict missing required key 'tissue_fqfn'"
   
        niid = self.nii_load(self.data_dict["tissue_fqfn"])
        self._data_dict["halflife"] = niid["halflife"]
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
        if "recovery_coefficient" not in self._data_dict:
            self._data_dict["recovery_coefficient"] = 1.8509
        if "delta_time" not in self._data_dict:
            self._data_dict["delta_time"] = 1
        if "nparcels" not in self._data_dict:
            self._data_dict["nparcels"] = 1

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
        """ interpolated to 1-sec times, timesMid """
        ifm = PETUtilities.interpdata(
            data=self.input_func_measurement,
            ref=self.tissue_measurement)
        return ifm["img"].copy()
    
    @property
    def input_func_measurement(self) -> dict:
        """ assumed to be decay-corrected, which is hereby uncorrected for[15O];
            preserves timings of input_func_fqn """
        if hasattr(self._data_dict, "input_func_measurement"):
            return deepcopy(self._data_dict["input_func_measurement"])

        input_func_meas = self.nii_load(self.input_func_fqfn)
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
        """ normalized by self.max_tissue_measurement """
        return self.rho_tissue_measurement["img"].copy()

    @property
    def rho_input_func_interp(self) -> NDArray:
        """ normalized by self.max_tissue_measurement; interpolated to 1-sec times, timesMid """
        ifm = PETUtilities.interpdata(
            data=self.rho_input_func_measurement,
            ref=self.rho_tissue_measurement)
        return ifm["img"].copy()
    
    @property
    def rho_input_func_measurement(self) -> dict:
        """ normalized by self.max_tissue_measurement and self.recovery_coefficient;
            preserves timings of input_func_measurement """
        if hasattr(self._data_dict, "rho_input_func_measurement"):
            return deepcopy(self._data_dict["rho_input_func_measurement"])

        # normalize to max tissue measurement
        rho_input_func_meas = self.input_func_measurement
        rho_input_func_meas["img"] = rho_input_func_meas["img"] / self.max_tissue_measurement
        
        # apply recovery coefficient to Boxcar
        if self.input_func_type == "Boxcar":
            rho_input_func_meas["img"] = \
                rho_input_func_meas["img"] * self.recovery_coefficient

        self._data_dict["rho_input_func_measurement"] = rho_input_func_meas
        return deepcopy(self._data_dict["rho_input_func_measurement"])
    
    @property
    def rho_tissue_measurement(self) -> dict:
        """ normalized by self.max_tissue_measurement """
        if hasattr(self._data_dict, "rho_tissue_measurement"):
            return deepcopy(self._data_dict["rho_tissue_measurement"])

        # normalize to max tissue measurement
        rho_tiss_meas = self.tissue_measurement
        rho_tiss_meas["img"] = rho_tiss_meas["img"] / self.max_tissue_measurement
        
        self._data_dict["rho_tissue_measurement"] = rho_tiss_meas
        return deepcopy(self._data_dict["rho_tissue_measurement"])
    
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
        self._data_dict["tag"] = tag

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

        tiss_meas = self.nii_load(self.tissue_fqfn)
        if PETUtilities.parse_isotope(self.tissue_fqfn) == "15O":
            tiss_meas = self.decay_uncorrect(tiss_meas)
        self._data_dict["tissue_measurement"] = tiss_meas
        return deepcopy(self._data_dict["tissue_measurement"])

    def decay_correct(self, niid: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """ correct for decay of [15O] """
        return PETUtilities.decay_correct(niid)

    def decay_uncorrect(self, niid: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """ uncorrect for decay of [15O] """
        return PETUtilities.decay_uncorrect(niid)
