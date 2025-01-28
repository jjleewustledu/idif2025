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


from __future__ import absolute_import
from __future__ import print_function
from abc import ABC, abstractmethod
from dynesty import utils as dyutils


class DynestyContext(ABC):
    """Base context class for dynesty-based model implementations.

    This abstract base class provides the foundation for implementing models that use dynesty
    for nested sampling analysis. It defines the required interface and common functionality
    for data handling, I/O operations, and result visualization.

    All PET and input function data must be decay corrected before use.

    Attributes:
        data: Abstract property for accessing model data.
        fqfp (str): Fully qualified file path for I/O operations.
        io: Abstract property for I/O handler.
        plotting: Abstract property for plotting utilities.
        results_fqfp (str): Fully qualified file path for results.
        solver: Abstract property for model solver.

    Example:
        >>> class MyModel(DynestyContext):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._data = ModelData()
        ...         self._io = ModelIO()
        ...         self._solver = ModelSolver()
        ...         self._plotting = ModelPlotting()

    Note:
        Subclasses must implement the abstract properties:
        - data
        - io
        - plotting
        - solver
        
        Requires all incoming PET and input function data to be decay corrected. 
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass
    
    @property
    def fqfp(self):
        return self.io.fqfp

    @property
    @abstractmethod
    def io(self):
        pass

    @property
    @abstractmethod
    def plotting(self):
        pass
    
    @property
    def results_fqfp(self):
        return self.io.results_fqfp

    @property
    @abstractmethod
    def solver(self):
        pass

    def pickle_load(self, fqfn: str) -> dyutils.Results | list[dyutils.Results]:
        return self.solver.pickle_load(fqfn)
    
    def pickle_dump(self, tag: str) -> str:
        return self.solver.pickle_dump(tag)
