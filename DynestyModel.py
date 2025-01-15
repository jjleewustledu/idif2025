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
# matplotlib.use('Agg')
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
rcParams.update({"font.size": 24})


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
                                    rstate=rstate,
                                    tag=tag)

    @abstractmethod
    def data(self, v):
        pass

    @property
    @abstractmethod
    def fqfp(self):
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
    def results_fqfp(self):
        pass

    @property
    def tag(self):
        return self.solver.tag
    
    @tag.setter
    def tag(self, tag):
        self.solver.tag = tag

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
        fqfp1 = self.results_fqfp + tag
        qm, _, _ = self.solver.quantile(res)

        # truth plot ------------------------------------------------------------

        try:
            self.plot_truths(qm, parc_index=parc_index)
            plt.savefig(fqfp1 + "-results.png")
            plt.savefig(fqfp1 + "-results.svg")
        except Exception as e:
            print(("PETModel.plot_results: caught an Exception: ", str(e)))
            traceback.print_exc()

        # run plot --------------------------------------------------------------

        try:
            dyplot.runplot(
                res,
                label_kwargs={"fontsize": 30})
            plt.tight_layout()
            plt.savefig(fqfp1 + "-runplot.png")
            plt.savefig(fqfp1 + "-runplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.runplot: caught a ValueError: {e}")

        # trace plot ------------------------------------------------------------

        try:
            fig, axes = dyplot.traceplot(
                res,
                labels=self.labels,
                truths=qm,
                title_fmt=".2f",
                label_kwargs={"fontsize": 26},
                fig=plt.subplots(self.ndim, 2, figsize=(12, 12 * self.ndim / 2)))
            # plt.tick_params(axis='both', labelsize=9)  # Smaller tick labels
            # for ax in axes.flat:
            #     ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Larger axis labels 
            #     ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            plt.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
            fig.tight_layout()
            plt.savefig(fqfp1 + "-traceplot.png")
            plt.savefig(fqfp1 + "-traceplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.traceplot: caught a ValueError: {e}")

        # corner plot ----------------------------------------------------------

        try:
            dyplot.cornerplot(
                res,
                truths=qm,
                title_fmt=".1f",
                show_titles=True,
                title_kwargs={"y": 1.1},
                labels=self.labels,
                label_kwargs={"fontsize": 50},
                fig=plt.subplots(self.ndim, self.ndim, figsize=(8 * self.ndim / 2, 8 * self.ndim / 2)))
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

    def print_data(self):
        """ Print the data in a table format. """
        print("\nData:")
        print("-" * 40)
        for key, value in self.data(self.truths).items():
            if isinstance(value, np.ndarray):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  First few values: {value[:3]}")
                print(f"  Last few values: {value[-3:]}")
            else:
                print(f"{key}: {value}")
        print("-" * 40)

    def print_truths(self):
        """ Print the truths in a table format. """
        truths = self.truths
        labels = self.labels
        print("\nTruths:")
        print("-" * 40)
        print(f"{'Parameter':<25} {'Value':>12}")
        print("-" * 40)
        for label, value in zip(labels, truths):
            print(f"{label:<25} {value:>12.5f}")
        print("-" * 40)

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
