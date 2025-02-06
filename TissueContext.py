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

from DynestyContext import DynestyContext
from IOImplementations import TissueIO
from TissuePlotting import TissuePlotting


class TissueContext(DynestyContext):
    """Base context class for tissue model implementations.

    This class provides the base context for analyzing PET data using various tissue models.
    It coordinates the data handling, solver, I/O operations, and plotting functionality
    common across different tissue model implementations.

    Args:
        data_dict (dict): Dictionary containing input data including:
            - input_func_fqfn (str): Fully qualified filename for input function data
            - tissue_fqfn (str): Fully qualified filename for tissue data

    Attributes:
        io (TissueIO): I/O handler for tissue data
        plotting (TissuePlotting): Plotting utilities for tissue data
        input_func_type (str): Type of input function used

    Example:
        >>> data = {"input_func_fqfn": "input.csv", "tissue_fqfn": "tissue.csv"}
        >>> context = TissueContext(data)
        >>> context.io.load_data()
        >>> context.plotting.plot_results()

    Notes:
        This is an abstract base class that should be inherited by specific tissue model
        implementations like Mintun1984Context or Raichle1983Context.

        Requires all incoming PET and input function data to be decay corrected.
    """

    def __init__(self, data_dict: dict):
        super().__init__()
        self._io = TissueIO(self)
        self._plotting = TissuePlotting(self)

    # @property
    # def data(self):
    #     return self._data

    @property
    def input_func_type(self):
        return self.data.input_func_type

    @property
    def io(self):
        return self._io

    @property
    def plotting(self):
        return self._plotting

    # @property
    # def solver(self):
    #     return self._solver

    # @property
    # def tag(self):
    #     return self.data.tag

    # @tag.setter
    # def tag(self, tag):
    #     self._data.tag = tag
