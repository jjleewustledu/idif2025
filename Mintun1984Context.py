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
from IOImplementations import TissueIO
from Mintun1984Data import Mintun1984Data
from Mintun1984Solver import Mintun1984Solver
from TissueContext import TissueContext
from TissuePlotting import TissuePlotting


class Mintun1984Context(TissueContext):
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
        self._data = Mintun1984Data(self, data_dict)
        self._solver = Mintun1984Solver(self)
        # self._plotting = TissuePlotting(self)
               
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
    m = Mintun1984Context(data_dict)
    m()
