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
import numpy as np
import nibabel as nib
import pandas as pd
from copy import deepcopy


class IOInterface(ABC):
    """Abstract interface class for filesystem I/O operations."""

    @abstractmethod
    def load_nii(self, fqfn: str) -> dict:
        """Load a NIfTI file and associated metadata."""
        pass

    @abstractmethod
    def save_nii(self, data: dict, fqfn: str = None):
        """Save data to a NIfTI file and associated metadata."""
        pass

    @abstractmethod
    def save_csv(self, data: dict, fqfn: str = None):
        """Save data to a CSV file."""
        pass