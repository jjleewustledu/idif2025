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
from scipy import stats
import matplotlib.pyplot as plt

from IOImplementations import BaseIO


class RBCPartition:
    """ RBCPartition models fig. 8 of Phelps, Huang, Hoffman, et al.
        Tomographic measurement of local cerebral glucose metabolic rate in humans with
        (F‐18)2‐fluoro‐2‐deoxy‐D‐glucose: Validation of method.
        Ann Neurol 6:371--388, 1979.
        See also https://github.com/jjleewustledu/mlraichle/blob/master/src/%2Bmlraichle/RBCPartition.m """

    def __init__(self, hct: float, t_units: str = "s"):

        hct = float(hct)  # ensure float
        if hct > 1:
            hct = hct / 100  # convert from percentage to fraction
        if not 0 < hct < 1:
            raise ValueError("hct must be between 0 and 1")
        self.hct = hct

        if t_units not in ["s", "m", "h"]:
            raise ValueError("t_units must be one of 's', 'm', 'h'")
        self.t_units = t_units

        self.fit_params()

    def fit_params(self):
        """Fit piecewise line and quadratic to RBC/plasma data."""

        # data digitized by pixel coordinates from fig. 8 of Phelps, Huang, Hoffman, et al.

        # Original data
        t_unsorted = np.array([
            0.551, 1.444, 2.082, 2.858, 2.01, 1.248, 0.559, 0.755, 1.074, 1.654, 1.973, 2.481, 2.611, 3.192, 3.961,
            3.772, 4.867, 4.867, 6.942, 9.792, 12.084, 14.848, 19.786, 25.129, 29.684, 40.119, 49.846, 59.311,
            65.98, 69.421, 79.62, 99.048, 109.949, 119.793, 139.749, 145.454, 159.177, 179.486, 191.654, 200.437,
            205.729, 219.901, 245.03, 238.684, 272.48, 258.255, 279.179, 287.906, 285.555, 298.016
        ])
        rbc_unsorted = np.array([
            0.852, 0.852, 0.838, 0.787, 0.79, 0.795, 0.799, 0.812, 0.812, 0.812, 0.812, 0.807, 0.811, 0.821,
            0.808, 0.838, 0.826, 0.845, 0.845, 0.853, 0.861, 0.87, 0.915, 0.875, 0.908, 0.869, 0.892, 0.865,
            0.939, 0.909, 0.946, 0.999, 0.942, 1.015, 0.912, 0.971, 1.062, 1.002, 1.066, 0.98, 1.03, 1.075,
            1.063, 1.139, 1.079, 1.152, 1.029, 1.105, 1.149, 1.208
        ])

        # Sort both arrays based on tData
        sort_indices = np.argsort(t_unsorted)
        self.tData = t_unsorted[sort_indices]
        self.rbcData = rbc_unsorted[sort_indices]
        # Split data into two regions
        self.t_crossover = 35
        mask_early = self.tData <= self.t_crossover
        mask_late = self.tData > self.t_crossover

        t_early = self.tData[mask_early]
        rbc_early = self.rbcData[mask_early]
        t_late = self.tData[mask_late]
        rbc_late = self.rbcData[mask_late]

        # Fit line to late data
        slope, intercept, _, _, _ = stats.linregress(t_late, rbc_late)

        # Get line value at t=20 for continuity constraint
        y_crossover = slope * self.t_crossover + intercept

        # Solve for a,b in ax^2 + bx + c = y where c = y_crossover - 20b - 400a
        # Rearranging: ax^2 + bx + (y_crossover - 20b - 400a) = y
        # ax^2 + bx - 20b - 400a = y - y_crossover
        # (x^2 - 400)a + (x - 20)b = y - y_crossover
        A_constrained = np.column_stack([(t_early**2 - self.t_crossover**2), (t_early - self.t_crossover)])
        b_constrained = rbc_early - y_crossover
        quad_params = np.linalg.lstsq(A_constrained, b_constrained, rcond=None)[0]
        a, b = quad_params
        c = y_crossover - self.t_crossover*b - (self.t_crossover**2)*a

        # Store fitted parameters
        self.early_quad = (a, b, c)  # y = ax^2 + bx + c
        self.late_line = (slope, intercept)  # y = mx + b

        # Create fitted values array
        self.rbcFitted = np.zeros_like(self.tData)
        early_indices = self.tData <= self.t_crossover
        late_indices = self.tData > self.t_crossover

        # Apply quadratic fit for early times
        self.rbcFitted[early_indices] = (a * self.tData[early_indices]**2 +
                                         b * self.tData[early_indices] + c)

        # Apply linear fit for late times
        self.rbcFitted[late_indices] = (slope * self.tData[late_indices] +
                                        intercept)

    def nii_wb2plasma(
        self,
        fqfn: str,
        do_trim: bool = True,
        time_last: float | None = None,
        check_validity: bool = True,
        output_format: str = "niid"
    ) -> dict | str:

        io = BaseIO()
        niid = io.nii_load(fqfn, do_trim, time_last, check_validity)
        rho_wb = niid["img"]
        t = niid["timesMid"]
        niid["img"] = self.wb2plasma(rho_wb, t)

        if output_format == "niid":
            return niid
        elif output_format == "fqfn":
            fqfn = niid["fqfp"] + "-wb2plasma.nii.gz"
            io.nii_save(niid, fqfn, check_validity)
            return fqfn
        else:
            raise ValueError(f"Invalid output format: {output_format}")

    def plot(self):
        """ Plot the fitted data. """

        plt.figure(figsize=(10, 6))
        plt.plot(self.tData, self.rbcData, 'c:', marker='o', mfc='none', alpha=1, label='Phelps 1979')
        plt.plot(self.tData, self.rbcFitted, 'b-', linewidth=3, alpha=0.5, label='Fitted')
        plt.xlabel('Time (min)')
        plt.ylabel('RBC/plasma ratio')
        # plt.xlim(0, 40)
        # plt.ylim(0.75, 0.95)
        plt.grid(True)
        plt.legend()
        plt.show()

    def rbc_over_plasma_5param(self, t: NDArray) -> NDArray:
        """ t (NDArray) in seconds
            returns rbc / plasma """

        t = self.t_in_minutes(t)
        lambda_t = np.interp(t, self.tData, self.rbcFitted)
        if lambda_t.ndim > 1:
            raise ValueError("lambda_t must be scalar or 1D array")
        return lambda_t

    def t_in_minutes(self, t: NDArray) -> NDArray:
        """ Phelps' models used minutes for [18F]FDG. """

        t = np.asarray(t)  # ensure NDArray
        if t.ndim > 1:
            raise ValueError("t must be scalar or 1D array")
        if self.t_units == "s":
            return t / 60.0  # convert to minutes
        elif self.t_units == "m":
            return t
        elif self.t_units == "h":
            return t * 60.0  # convert to minutes
        else:
            raise ValueError("t_units must be one of 's', 'm', 'h'")

    def wb2plasma(self, rho_wb: NDArray, t: NDArray) -> NDArray:
        """ t (NDArray): may be start of frame or mid-frame.
            units (str):  provide API consistency. """

        lambda_t = self.rbc_over_plasma_5param(t)
        return rho_wb / (1 + self.hct * (lambda_t - 1))
