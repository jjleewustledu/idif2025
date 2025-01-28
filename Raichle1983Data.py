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

from TissueData import TissueData
from IOImplementations import TissueIO
from PETUtilities import PETUtilities

class Raichle1983Data(TissueData):
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)
    
    @property
    def v1(self) -> NDArray:
        return self.v1_measurement["img"].copy()

    @property
    def v1_measurement(self) -> dict:
        """ adjusts for recovery coefficient if RadialArtery used """
        if hasattr(self._data_dict, "v1_measurement"):
            return deepcopy(self._data_dict["v1_measurement"])

        self._data_dict["v1_measurement"] = self.nii_load(self._data_dict["v1_fqfn"])
        if self.input_func_type == "RadialArtery":
            self._data_dict["v1_measurement"]["img"] = self._data_dict["v1_measurement"]["img"] / self.recovery_coefficient
        return deepcopy(self._data_dict["v1_measurement"])
