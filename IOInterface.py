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
from abc import ABC, abstractmethod
import os
import json
from typing import Any
import numpy as np
import nibabel as nib
import pandas as pd
from copy import deepcopy


class IOInterface(ABC):
    """Abstract interface class for filesystem I/O operations.

    This class defines the interface for file system input/output operations used across
    the PET data analysis pipeline. It provides abstract methods that must be implemented
    by concrete subclasses to handle various I/O tasks.

    The interface includes methods for:
        - Loading and saving NIfTI files and associated metadata
        - Extracting timing information from data dictionaries
        - File path manipulation and validation
        - Data interpolation and time series operations

    Attributes:
        fqfp: Fully qualified file prefix property
        results_fqfp: Results file prefix property

    Note:
        This is an abstract base class that cannot be instantiated directly.
        Concrete implementations must override all abstract methods.
    """

    @property
    @abstractmethod
    def fqfp(self):
        pass

    @property
    @abstractmethod
    def results_fqfp(self):
        pass

    @staticmethod
    @abstractmethod
    def data2taus(data: dict) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def data2times(data: dict) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def data2timesInterp(data: dict) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def data2timesMid(data: dict) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def fileparts(fqfn: str) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def fqfileprefix(fqfn: str) -> str:
        pass

    @abstractmethod
    def nii_load(self, fqfn: str) -> dict:
        """Load a NIfTI file and associated metadata."""
        pass

    @abstractmethod
    def nii_save(self, data: dict, fqfn: str = None):
        """Save data to a NIfTI file and associated metadata."""
        pass

    @abstractmethod
    def pickle_dump(self, data: Any, fqfn: str | None = None) -> None:
        """Save by pickling."""        
        pass

    @abstractmethod
    def pickle_load(self, fqfn: str) -> Any:
        """Load data from a pickle file."""
        pass

    @abstractmethod
    def to_csv(self, data: dict, fqfn: str = None):
        """Save data to a CSV file."""
        pass
