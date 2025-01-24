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
from matplotlib import pyplot as plt

from DynestyPlotting import DynestyPlotting


class TissuePlotting(DynestyPlotting):
    def __init__(self, context):
        super().__init__(context)

    def parcsplot(
            self,
            truths: NDArray | None = None,
            parc_index: int | list[int] | tuple[int, ...] | NDArray = 0,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots from multiple selected parc. indices
            PET measurement, rho_pred from signal model, and also input function prediction"""
        pass

    def truthplot(
            self,
            truths: NDArray | None = None,
            parc_index: int = 0,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots PET measurement, rho_pred from signal model, and input function prediction """
        if truths is None:
            truths = self.context.solver.truths

        if truths.ndim == 2:
            truths = truths[parc_index]

        A_max = self.context.data.max_tissue_measurement

        # PET measurement
        tiss_meas = self.context.data.rho_tissue_measurement
        A_tiss = A_max * tiss_meas["img"][parc_index]
        timesMid_tiss = tiss_meas["timesMid"]

        # signal model
        rho_pred, timesMid_pred, _, _ = self.context.solver.signalmodel(truths, parc_index=parc_index)
        A_pred = A_max * rho_pred

        # input function prediction
        if_meas = self.context.data.rho_input_func_measurement
        A_if = A_max * if_meas["img"]
        timesMid_if = if_meas["timesMid"]

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
        p1, = plt.plot(
            timesMid_tiss,
            yscaling * A_tiss,
            color="black",
            marker="+", 
            ls="none",
            alpha=0.9,
            markersize=16,
            label=f"PET measured, parcel {parc_index}")
        p2, = plt.plot(
            timesMid_pred,
            yscaling * A_pred,
            marker="o",
            color="red",
            ls="none", 
            alpha=0.8,
            markersize=6,
            label=f"PET predicted, parcel {parc_index}")
        p3, =plt.plot(
            timesMid_if,
            yscaling * if_scaling * A_if,
            color="dodgerblue",
            linewidth=3,
            alpha=0.7,
            label=f"input function x {if_scaling:.3}")        
        plt.xlim((-0.1 * xwidth, 1.1 * xwidth))
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel(f"activity ({activity_units})")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()
