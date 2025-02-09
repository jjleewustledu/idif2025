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


from TissueData import TissueData


class SpectralData(TissueData):
    """Data class for handling spectral data.

    This class extends TissueData to provide specific functionality for processing and analyzing
    PET data according to spectral methods. It handles data loading, measurement adjustments,
    and provides access to the number of spectral components, M.

    Args:
        context: The analysis context object containing configuration and utilities
        data_dict (dict, optional): Dictionary containing input data. Defaults to empty dict.

    Attributes:
        M (int): Number of spectral components
    """

    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)
        if "M" not in self._data_dict:
            self._data_dict["M"] = 3 
    
    @property
    def corner_title_fmt(self):
        return ".4f"

    @property
    def M(self) -> int:
        return self.data_dict["M"]
    
    @M.setter
    def M(self, M: int):
        self._data_dict["M"] = M
