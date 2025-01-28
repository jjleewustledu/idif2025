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


import os
import sys
import logging
import matplotlib

from DynestyContext import DynestyContext
from IOImplementations import BoxcarIO
from BoxcarData import BoxcarData
from BoxcarSolver import BoxcarSolver
from InputFuncPlotting import InputFuncPlotting


class BoxcarContext(DynestyContext):
    """Context class for analyzing input functions with boxcar averaging.

    This class provides the context for analyzing PET input functions using boxcar averaging.
    It coordinates the data handling, solver, and I/O operations specific to boxcar analysis.

    Args:
        data_dict (dict): Dictionary containing input data including:
            - input_func_fqfn (str): Fully qualified filename for input function data
            - nlive (int): Number of live points for nested sampling

    Attributes:
        data (BoxcarData): Data handler for boxcar analysis
        solver (BoxcarSolver): Solver implementing boxcar averaging
        io (BoxcarIO): I/O handler for boxcar data
        plotting (InputFuncPlotting): Plotting utilities for input functions
        tag (str): Identifier tag for the analysis

    Example:
        >>> data = {"input_func_fqfn": "input.csv", "nlive": 1000}
        >>> context = BoxcarContext(data)
        >>> context()  # Run the analysis
        >>> results = context.solver.results_load()

    Notes:
        Requires all incoming PET and input function data to be decay corrected.
    """
    def __call__(self) -> None:
        logging.basicConfig(
            filename=self.data.results_fqfp + ".log",
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s")        
        self.solver.run_nested(print_progress=False)
        self.solver.results_save()
        self.plotting.results_plot(do_save=True)

    def __init__(self, data_dict: dict):
        super().__init__()
        self._io = BoxcarIO(self)
        self._data = BoxcarData(self, data_dict)
        self._solver = BoxcarSolver(self)
        self._plotting = InputFuncPlotting(self)
               
    @property
    def data(self):
        return self._data
    
    @property
    def io(self):
        return self._io
        
    @property
    def solver(self):
        return self._solver
    
    @property
    def plotting(self):
        return self._plotting

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
        "nlive": int(sys.argv[2])
    }
    bc = BoxcarContext(data_dict)
    bc()
