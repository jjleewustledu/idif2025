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

# general & system functions
from __future__ import absolute_import
from abc import ABC, abstractmethod


class DynestyInterface(ABC):
    """
    """

    @property
    @abstractmethod
    def fqfp(self):
        pass

    @property
    @abstractmethod
    def fqfp_results(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def truths(self):
        pass

    @staticmethod
    @abstractmethod
    def data(v):
        pass

    @staticmethod
    @abstractmethod
    def loglike(v):
        pass

    @abstractmethod
    def plot_results(self, res, parc_index):
        pass

    @abstractmethod
    def plot_truths(self, truths, parc_index):
        pass

    @abstractmethod
    def plot_variations(self, tindex0, tmin, tmax, truths):
        pass

    @staticmethod
    @abstractmethod
    def prior_transform(tag):
        pass

    @abstractmethod
    def run_nested(self, checkpoint_file):
        pass

    @abstractmethod
    def save_results(self, res, tag):
        pass

    @staticmethod
    @abstractmethod
    def signalmodel(data: dict):
        pass
