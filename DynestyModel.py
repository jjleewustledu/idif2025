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
from abc import ABC, abstractmethod


class DynestyModel(ABC):
    """

    :class: DynestyModel

    Abstract Base Class for implementing a Dynesty model.

    .. attribute:: fqfp

        Full qualified file prefix (fqfp) of the model.

    .. attribute:: fqfp_results

        Full qualified file prefix (fqfp) of the results.

    .. attribute:: labels

        List of parameter labels.

    .. attribute:: truths

        List of parameter truths.

    .. method:: plot_results(res, parc_index)

        Plots the results of the model.

        :param res: The results of the model.
        :type res: unknown

        :param parc_index: The parameter index.
        :type parc_index: int

    .. method:: plot_truths(truths, parc_index)

        Plots the parameter truths.

        :param truths: The parameter truths.
        :type truths: unknown

        :param parc_index: The parameter index.
        :type parc_index: int

    .. method:: plot_variations(tindex0, tmin, tmax, truths)

        Plots the variations of parameters.

        :param tindex0: Unknown parameter.
        :type tindex0: unknown

        :param tmin: Minimum parameter value.
        :type tmin: float

        :param tmax: Maximum parameter value.
        :type tmax: float

        :param truths: The parameter truths.
        :type truths: unknown

    .. method:: run_nested(checkpoint_file)

        Runs the nested sampler.

        :param checkpoint_file: The checkpoint file.
        :type checkpoint_file: str

    .. method:: save_results(res, tag)

        Saves the results of the model.

        :param res: The results of the model.
        :type res: unknown

        :param tag: The results tag.
        :type tag: str

    .. staticmethod:: data(v)

        Unknown method.

        :param v: Unknown parameter.
        :type v: unknown

    .. staticmethod:: loglike(v)

        Unknown method.

        :param v: Unknown parameter.
        :type v: unknown

    .. staticmethod:: prior_transform(tag)

        Unknown method.

        :param tag: The prior transform tag.
        :type tag: unknown

    .. staticmethod:: signalmodel(data)

        Creates a signal model based on the given data.

        :param data: The data to create the signal model.
        :type data: dict

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

    @abstractmethod
    def plot_results(self, res, parc_index):
        pass

    @abstractmethod
    def plot_truths(self, truths, parc_index):
        pass

    @abstractmethod
    def plot_variations(self, tindex0, tmin, tmax, truths):
        pass

    @abstractmethod
    def run_nested(self, checkpoint_file):
        pass

    @abstractmethod
    def save_results(self, res, tag):
        pass

    @staticmethod
    @abstractmethod
    def data(v):
        pass

    @staticmethod
    @abstractmethod
    def loglike(v):
        pass

    @staticmethod
    @abstractmethod
    def prior_transform(tag):
        pass

    @staticmethod
    @abstractmethod
    def signalmodel(data: dict):
        pass
