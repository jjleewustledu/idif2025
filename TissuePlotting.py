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

# plotting
from matplotlib import cm, pyplot as plt

from DynestyPlotting import DynestyPlotting


class TissuePlotting(DynestyPlotting):
    """Plotting utilities for tissue time-activity curves and model predictions.

    This class extends DynestyPlotting to provide specialized plotting methods for tissue
    time-activity curves, including measured data, model predictions, and input functions.

    Args:
        context: The context object containing solver, data, and IO information.

    Attributes:
        Inherits all attributes from DynestyPlotting parent class.

    Example:
        >>> plotter = TissuePlotting(context)
        >>> plotter.parcs_plot(activity_units="kBq/mL")
    """
    def __init__(self, context):
        super().__init__(context)

    def parcs_plot(
            self,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots from multiple selected parc. indices
            PET measurement, rho_pred from signal model, and also input function prediction"""
        truths = self.context.solver.truths        
        A_max = self.context.data.max_tissue_measurement

        # input function prediction
        if_meas = self.context.data.rho_input_func_measurement
        A_if = A_max * if_meas["img"]
        timesMid_if = if_meas["timesMid"]

        # PET measurement
        tiss_meas = self.context.data.rho_tissue_measurement
        
        parc_index = range(len(truths))
        timesMid_tiss = tiss_meas["timesMid"]

        # scalings
        if activity_units.startswith("k"):
            yscaling = 0.001
        elif activity_units.startswith("M"):
            yscaling = 1e-6
        else:
            yscaling = 1
        if_scaling = A_max / np.max(A_if)
        xwidth = np.min((timesMid_tiss[-1], timesMid_if[-1]))

        fig, ax = plt.subplots(figsize=(12, 0.618*12))
        ncolors = len(parc_index) 
        reds = cm.get_cmap("Reds", ncolors)
        greys = cm.get_cmap("Greys", ncolors)

        p3, = ax.plot(
            timesMid_if,
            yscaling * if_scaling * A_if,
            color="dodgerblue",
            linewidth=3,
            alpha=0.7,
            label=f"input function x {if_scaling:.3}")  

        for i, pidx in enumerate(parc_index):
            p1, = ax.plot(
                timesMid_tiss,
                yscaling * A_max * tiss_meas["img"][pidx],
                color=greys(i),
                marker="+", 
                ls="none",
                alpha=0.6,
                markersize=10,
                label=f"PET measured, parcel 0:{ncolors-1}")
            
            # signal model
            rho_pred, timesMid_pred, _, _ = self.context.solver.signalmodel(truths[i], parc_index=pidx)
            p2, = ax.plot(
                timesMid_pred,
                yscaling * A_max * rho_pred,
                color=reds(i),
                linewidth=2,
                alpha=0.6,
                label=f"PET predicted, parcel 0:{ncolors-1}")
        
        plt.xlim((-0.1 * xwidth, 1.1 * xwidth))
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel(f"activity ({activity_units})")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()

    def truths_plot(
            self,
            truths: list[float] | tuple[float, ...] | NDArray | None = None,
            parc_index: int | None = None,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots PET measurement, rho_pred from signal model, and input function prediction """
        if truths is None:
            truths = self.context.solver.truths

        if isinstance(truths, np.ndarray) and truths.ndim == 2:
            if parc_index is None:
                raise ValueError("parc_index must not be None when truths is 2-dimensional")
            truths = truths[parc_index]

        A_max = self.context.data.max_tissue_measurement

        # input function prediction
        if_meas = self.context.data.rho_input_func_measurement
        A_if = A_max * if_meas["img"]
        timesMid_if = if_meas["timesMid"]

        # PET measurement
        tiss_meas = self.context.data.rho_tissue_measurement
        timesMid_tiss = tiss_meas["timesMid"]

        # signal model
        rho_pred, timesMid_pred, _, _ = self.context.solver.signalmodel(truths, parc_index=parc_index)

        # scalings
        if activity_units.startswith("k"):
            yscaling = 0.001
        elif activity_units.startswith("M"):
            yscaling = 1e-6
        else:
            yscaling = 1
        if_scaling = A_max / np.max(A_if)
        xwidth = np.min((timesMid_tiss[-1], timesMid_if[-1]))

        plt.figure(figsize=(12, 0.618*12))

        p3, = plt.plot(
            timesMid_if,
            yscaling * if_scaling * A_if,
            color="dodgerblue",
            linewidth=3,
            alpha=0.7,
            label=f"input function x {if_scaling:.3}")  
        p1, = plt.plot(
            timesMid_tiss,
            yscaling * A_max * tiss_meas["img"][parc_index],
            color="black",
            marker="+", 
            ls="none",
            alpha=0.9,
            markersize=16,
            label=f"PET measured, parcel {parc_index}")
        p2, = plt.plot(
            timesMid_pred,
            yscaling * A_max * rho_pred,
            marker="o",
            color="red",
            ls="none", 
            alpha=0.8,
            markersize=6,
            label=f"PET predicted, parcel {parc_index}")
        
        plt.xlim((-0.1 * xwidth, 1.1 * xwidth))
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel(f"activity ({activity_units})")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()
