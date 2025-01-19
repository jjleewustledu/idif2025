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


import numpy as np
from numpy.typing import NDArray
import traceback

# plotting
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rcParams

# re-defining plotting defaults
# matplotlib.use('Agg')  # disable interactive plotting, to be placed in idif.py
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

from dynesty import utils as dyutils, plotting as dyplot


class DynestyPlotting:
    def __init__(self, context):
        self.context = context
    
    @property
    def fqfp(self):
        return self.context.io.fqfp 

    @property
    def labels(self):
        return self.context.solver.labels  
    
    @property
    def ndim(self):
        return self.context.solver.ndim
    
    @property
    def results_fqfp(self):
        return self.context.io.results_fqfp 
    
    def plot_results(
            self,
            tag: str = "",
            parc_index: int | None = None,
            do_save: bool = True
    ) -> None:

        if tag:
            tag = f"-{tag.lstrip('-')}"
        if parc_index is not None:
            if f"-parc{parc_index}" not in tag:
                tag = tag + f"-parc{parc_index}"
        fqfp1 = self.results_fqfp + tag
        res = self.context.solver.dynesty_results
        qm, _, _ = self.context.solver.quantile()

        # truth plot ------------------------------------------------------------

        try:
            self.plot_truths(qm, parc_index=parc_index)
            if do_save:
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
            if do_save:
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
            plt.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
            fig.tight_layout()
            if do_save:
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
                label_kwargs={"fontsize": 36},
                fig=plt.subplots(self.ndim, self.ndim, figsize=(8 * self.ndim / 2, 8 * self.ndim / 2)))
            if do_save:
                plt.savefig(fqfp1 + "-cornerplot.png")
                plt.savefig(fqfp1 + "-cornerplot.svg")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.cornerplot: caught a ValueError: {e}")

    def plot_variations(
            self,
            tindex: int = 0,
            tmin: float | None = None,
            tmax: float | None = None,
            truths: NDArray | None = None,
            tag: str = "",
            do_save: bool = True
    ) -> None:
        if truths is None:
            truths = self.context.solver.truths
        truths_ = truths.copy()

        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 = self.results_fqfp + tag + f"-v{tindex}-{tmin}-{tmax}"

        fig, ax = plt.subplots(figsize=(12, 0.618*12))  # Create figure and axes explicitly
        ncolors: int = 75
        viridis = cm.get_cmap("viridis", ncolors)
        dt = (tmax - tmin) / ncolors
        trange = np.arange(tmin, tmax, dt)
        for tidx, t in enumerate(trange):
            truths_[tindex] = t
            _, rho_ideal, timesIdeal = self.context.solver.signalmodel(truths_)
            ax.plot(timesIdeal, rho_ideal, color=viridis(tidx))

        ax.set_xlabel("time of mid-frame (s)")
        ax.set_ylabel("activity (arbitrary)")

        # Add a colorbar to understand colors
        sm = plt.cm.ScalarMappable(cmap=viridis)
        sm.set_array(trange)
        plt.colorbar(sm, ax=ax, label="Varying " + self.labels[tindex])  # Pass the ax argument
        plt.tight_layout()

        if do_save:
            plt.savefig(fqfp1 + "-variations.png")
            plt.savefig(fqfp1 + "-variations.svg")
 
