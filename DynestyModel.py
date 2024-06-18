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


from DynestyInterface import DynestyInterface
from DynestySolver import DynestySolver
from dynesty import utils as dyutils, plotting as dyplot

# general & system functions
from abc import abstractmethod
import traceback

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

# re-defining plotting defaults
matplotlib.use('Agg')
rcParams.update({"xtick.major.pad": "7.0"})
rcParams.update({"xtick.major.size": "7.5"})
rcParams.update({"xtick.major.width": "1.5"})
rcParams.update({"xtick.minor.pad": "7.0"})
rcParams.update({"xtick.minor.size": "3.5"})
rcParams.update({"xtick.minor.width": "1.0"})
rcParams.update({"ytick.major.pad": "7.0"})
rcParams.update({"ytick.major.size": "7.5"})
rcParams.update({"ytick.major.width": "1.5"})
rcParams.update({"ytick.minor.pad": "7.0"})
rcParams.update({"ytick.minor.size": "3.5"})
rcParams.update({"ytick.minor.width": "1.0"})
rcParams.update({"font.size": 30})


class DynestyModel(DynestyInterface):
    """
    """

    def __init__(self,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag=""):
        self.solver = DynestySolver(model=self,
                                    sample=sample,
                                    nlive=nlive,
                                    rstate=rstate)
        self.NLIVE = nlive
        self.TAG = tag

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
    def ndim(self):
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

    def plot_results(self, res: dyutils.Results, tag="", parc_index=None):

        if tag and "-" not in tag:
            tag = "-" + tag
        if parc_index and f"-parc{parc_index}" not in tag:
            tag = tag + f"-parc{parc_index}"
        fqfp1 = self.fqfp_results + tag
        qm, _, _ = self.solver.quantile(res)

        try:
            self.plot_truths(qm, parc_index=parc_index)
            plt.savefig(fqfp1 + "-results.png")
            plt.savefig(fqfp1 + "-results.svg")
        except Exception as e:
            print("PETModel.plot_results: caught an Exception: ", str(e))
            traceback.print_exc()

        try:
            dyplot.runplot(res)
            plt.tight_layout()
            plt.savefig(fqfp1 + "-runplot.png")
            plt.savefig(fqfp1 + "-runplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.runplot: caught a ValueError: {e}")

        try:
            fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm, title_fmt=".5f",
                                         fig=plt.subplots(self.ndim, 2, figsize=(16, 25)))
            plt.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
            fig.tight_layout()
            plt.savefig(fqfp1 + "-traceplot.png")
            plt.savefig(fqfp1 + "-traceplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.traceplot: caught a ValueError: {e}")

        try:
            dyplot.cornerplot(res, truths=qm, title_fmt=".5f", show_titles=True,
                              title_kwargs={"y": 1.04}, labels=self.labels,
                              fig=plt.subplots(self.ndim, self.ndim, figsize=(100, 100)))
            plt.savefig(fqfp1 + "-cornerplot.png")
            plt.savefig(fqfp1 + "-cornerplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.cornerplot: caught a ValueError: {e}")

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
