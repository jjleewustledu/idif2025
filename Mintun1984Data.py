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
from numpy.typing import NDArray

from IOImplementations import TissueIO
from TissueData import TissueData


class Mintun1984Data(TissueData):
    """Data class for handling Mintun 1984 PET data.

    This class extends TissueData to provide specific functionality for processing and analyzing
    PET data according to the Mintun 1984 model. It handles data loading, measurement adjustments,
    and provides access to volume of distribution (v1) and binding site density (ks) measurements.

    Args:
        context: The analysis context object containing configuration and utilities
        data_dict (dict, optional): Dictionary containing input data. Defaults to empty dict.
            Required keys:
                - v1_fqfn: Fully qualified filename for v1 measurements
                - ks_fqfn: Fully qualified filename for ks measurements

    Attributes:
        v1 (NDArray): Volume of distribution measurements as a numpy array
        v1_measurement (dict): Raw volume of distribution measurements with metadata
        ks (NDArray): Binding site density measurements as a numpy array
        ks_measurement (dict): Raw binding site density measurements with metadata

    Note:
        When using RadialArtery input function type, v1 measurements are automatically
        adjusted using the recovery coefficient.
    """
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)
        assert "v1_fqfn" in self.data_dict, "data_dict missing required key 'v1_fqfn'"
        assert "ks_fqfn" in self.data_dict, "data_dict missing required key 'ks_fqfn'"

    @property
    def ks(self) -> NDArray:
        return self.ks_measurement["img"].copy()

    @property
    def ks_measurement(self) -> dict:
        """ adjusts for recovery coefficient if RadialArtery used """

        if hasattr(self._data_dict, "ks_measurement"):
            return deepcopy(self._data_dict["ks_measurement"])

        self._data_dict["ks_measurement"] = self.nii_load(self._data_dict["ks_fqfn"])
        return deepcopy(self._data_dict["ks_measurement"])

    @property
    def v1(self) -> NDArray:
        return self.v1_measurement["img"].copy()

    @property
    def v1_fqfn(self) -> str:
        return self.data_dict["v1_fqfn"]

    @property
    def v1_measurement(self) -> dict:
        """ adjusts for recovery coefficient if RadialArtery used """

        if hasattr(self._data_dict, "v1_measurement"):
            return deepcopy(self._data_dict["v1_measurement"])

        self._data_dict["v1_measurement"] = self.nii_load(self.v1_fqfn)
        if self.v1_type == "Boxcar":
            img = self._data_dict["v1_measurement"]["img"]
            self._data_dict["v1_measurement"]["img"] = img / self.recovery_coefficient
        return deepcopy(self._data_dict["v1_measurement"])

    @property
    def v1_type(self) -> str:
        _, bname = TissueIO.fileparts(self.v1_fqfn)
        if "idif" in bname or "MipIdif" in bname or "Boxcar" in bname:
            return "Boxcar"
        if "twilite" in bname or "TwiliteKit" in bname or "RadiaArtery" in bname:
            return "RadialArtery"
        raise ValueError(f"v1_fqfn {self.v1_fqfn} not recognizably associated with an input function")
