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


import sys
import logging
import matplotlib

from Huang1980Data import Huang1980Data
from Huang1980Solver import Huang1980Solver
from TissueContext import TissueContext


class Huang1980Context(TissueContext):
    """Context class for the Huang 1980 tissue model implementation.

    This class provides the context for analyzing PET data using the Huang 1980 tissue model.
    It coordinates the data handling, solver, and I/O operations specific to this model.

    Args:
        data_dict (dict): Dictionary containing input data including:
            - input_func_fqfn (str): Fully qualified filename for input function data
            - tissue_fqfn (str): Fully qualified filename for tissue data

    Attributes:
        data (Huang1980Data): Data handler for the Huang 1980 model
        solver (Huang1980Solver): Solver implementing the Huang 1980 model
        tag (str): Identifier tag for the analysis
        input_func_type (str): Type of input function used

    Example:
        >>> data = {"input_func_fqfn": "input.csv", "tissue_fqfn": "tissue.csv"}
        >>> context = Huang1980Context(data)
        >>> context()  # Run the analysis
        >>> results = context.solver.results_load()

    Notes:
        The Huang 1980 model is described in:
        Huang MA, Raichle ME, Martin WR, Herscovitch P.
        Brain oxygen utilization measured with O-15 radiotracers and positron emission tomography.
        J Nucl Med. 1980 Feb;25(2):177-87. PMID: 6610032.

        Requires all incoming PET and input function data to be decay corrected.
    """

    def __call__(self) -> None:
        logging.basicConfig(
            filename=self.data.results_fqfp + ".log",
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s")
        self.solver.run_nested(print_progress=False)
        self.solver.results_save()

    def __init__(self, data_dict: dict):
        super().__init__(data_dict)
        # self._io = TissueIO(self)
        self._data = Huang1980Data(self, data_dict)
        self._solver = Huang1980Solver(self)
        # self._plotting = TissuePlotting(self)
        self.tag += "-Huang1980"

    @property
    def data(self):
        return self._data

    @property
    def input_func_type(self):
        return self.data.input_func_type

    # @property
    # def io(self):
    #     return self._io

    # @property
    # def plotting(self):
    #     return self._plotting

    @property
    def solver(self):
        return self._solver

    @property
    def tag(self):
        return self.data.tag

    @tag.setter
    def tag(self, tag):
        self._data.tag = tag


if __name__ == "__main__":
    matplotlib.use('Agg')  # disable interactive plotting

    data_dict = {
        "input_func_fqfn": sys.argv[1],
        "tissue_fqfn": sys.argv[2],
        "v1_fqfn": sys.argv[3],
        "ks_fqfn": sys.argv[4],
        "nlive": int(sys.argv[5])
    }
    h = Huang1980Context(data_dict)
    h()
