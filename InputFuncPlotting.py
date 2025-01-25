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


class InputFuncPlotting(DynestyPlotting):
    def __init__(self, context):
        super().__init__(context)

    def truths_plot(
            self,
            truths: NDArray | None = None,
            parc_index: int | None = None,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots input function measurement and rho_pred, with rho_ideal, from signalmodel """
        if truths is None:
            truths = self.context.solver.truths

        # input function measurement
        ifm = self.context.data.input_func_measurement
        rho = ifm["img"]
        M0 = np.max(rho)
        tM = ifm["timesMid"]

        # signal model
        rho_pred, rho_ideal, timesIdeal = self.context.solver.signalmodel(truths)

        # scalings        
        if activity_units.startswith("k"):
            scaling = 0.001
        elif activity_units.startswith("M"):
            scaling = 1e-6
        else:
            scaling = 1

        plt.figure(figsize=(12, 0.618*12))
        plt.plot(
            tM,
            scaling * rho,
            color="black",
            marker="+", 
            ls="none",
            alpha=0.9,
            markersize=16)
        plt.plot(
            tM,
            scaling * M0 * rho_pred,
            marker="o",
            color="red",
            ls="none", 
            alpha=0.8,
            markersize=6)
        plt.plot(
            timesIdeal,
            scaling * M0 * rho_ideal,
            color="dodgerblue",
            linewidth=3,
            alpha=0.7)
        plt.xlim([-0.1, 1.1 * np.max(tM)])
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel(f"activity ({activity_units})")
        plt.tight_layout()
