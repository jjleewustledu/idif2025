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

from DynestyModel import DynestyModel

# general & system functions
from abc import ABC
import os
import sys
from copy import deepcopy
import inspect
import warnings

# basic numeric setup
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# NIfTI support
import json
import nibabel as nib

from IOImplementations import BaseIO
from PETData import PETData


class PETModel(DynestyModel, ABC):
    """
    """

    def __init__(self, home=os.getcwd(), time_last=None, **kwargs):
        super().__init__(**kwargs)
        self.home = home
        self.time_last = time_last
        self.io = BaseIO()

    @staticmethod
    def decay_correct(tac: dict):
        return PETData.decay_correct(tac)

    @staticmethod
    def decay_uncorrect(tac: dict):
        return PETData.decay_uncorrect(tac)
    
    @staticmethod
    def interpimg(timesNew: NDArray, times: NDArray, img: NDArray, kind: str="linear") -> NDArray:
        return PETData.interpimg(timesNew, times, img, kind)

    def load_nii(self, fqfn):
        return self.io.load_nii(fqfn)

    @staticmethod
    def parse_halflife(fqfp: str):
        return PETData.parse_halflife(fileprefix=fqfp)

    @staticmethod
    def parse_isotope(name: str):
        return PETData.parse_isotope(fileprefix=name)

    def save_csv(self, data: dict, fqfn=None):
        self.io.save_csv(data, fqfn)

    def save_nii(self, data: dict, fqfn=None):
        self.io.save_nii(data, fqfn)

    @staticmethod
    def slide(rho, t, dt, halflife=None):
        return PETData.slide(rho, t, dt, halflife)
    