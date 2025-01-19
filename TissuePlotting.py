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

    def plot_truths(
            self,
            truths: NDArray | None = None,
            parc_index: int | None = None,
            activity_units: str = "kBq/mL"
    ) -> None:
        """ plots PET measurement, rho_pred from signal model, and input function prediction """
        if truths is None:
            truths = self.context.solver.truths

        # PET measurement
        petm = self.context.data.adjusted_pet_measurement
        if parc_index:
            A_pet = petm["img"][parc_index]
        else:
            # evaluates median of petm["img"] over spatial dimensions only
            assert petm["img"].ndim == 2, f"Expected petm['img'] to be 2D but got {petm['img'].ndim}D"
            A_pet = np.median(petm["img"], axis=0)
        t_pet = petm["timesMid"]

        # signal model
        rho_pred, t_pred, _, _ = self.context.solver.signalmodel(truths)

        # input function prediction
        ifm = self.context.data.adjusted_input_function
        A_if_hat = ifm["img"]
        t_if = ifm["timesMid"]

        # scalings
        A0 = np.max(petm["img"])
        if_scaling = A0 / np.max(ifm["img"])
        if activity_units.startswith("k"):
            yscaling = 0.001
        elif activity_units.startswith("M"):
            yscaling = 1e-6
        else:
            yscaling = 1
        xwidth = np.max((t_pet[-1], t_if[-1]))

        plt.figure(figsize=(12, 0.618*12))
        p1, = plt.plot(
            t_pet,
            yscaling * A_pet,
            color="black",
            marker="+", 
            ls="none",
            alpha=0.9,
            markersize=16,
            label=f"PET measured, parcel {parc_index}")
        p2, = plt.plot(
            t_pred,
            yscaling * A0 * rho_pred,
            marker="o",
            color="red",
            ls="none", 
            alpha=0.8,
            markersize=6,
            label=f"PET predicted, parcel {parc_index}")
        p3, =plt.plot(
            t_if,
            yscaling * if_scaling * A_if_hat,
            color="dodgerblue",
            linewidth=3,
            alpha=0.7,
            label=f"input function x {if_scaling:.3}")        
        plt.xlim((-0.1 * xwidth, 1.1 * xwidth))
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel(f"activity ({activity_units})")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()
