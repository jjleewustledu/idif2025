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

from PETModel import PETModel
from dynesty import utils as dyutils
from dynesty import plotting as dyplot

# general & system functions
import re
import os
from copy import deepcopy

# basic numeric setup
import numpy as np
import pandas as pd

# NIfTI support
import json
import nibabel as nib

# plotting
from matplotlib import pyplot as plt
from matplotlib import cm

# re-defining plotting defaults
from matplotlib import rcParams

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

SIGMA = 0.05


class TCModel(PETModel):

    def __init__(self,
                 input_function,
                 pet_measurement,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        super().__init__(home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate)
        self.__input_function = input_function
        self.__pet_measurement = pet_measurement

    @property
    def fqfp(self):
        return self.pet_measurement["fqfp"]

    @property
    def input_function(self):
        if isinstance(self.__input_function, dict):
            return self.__input_function

        assert os.path.isfile(self.__input_function), f"{self.__input_function} was not found."
        fqfn = self.__input_function
        if self.parse_isotope(fqfn) == "15O":
            niid = self.decay_uncorrect(self.load_nii(fqfn))
        else:
            niid = self.load_nii(fqfn)

        # interpolate to timing domain of pet_measurements
        petm = self.pet_measurement
        tMI = np.arange(0, round(petm["timesMid"][-1]))
        niid["img"] = np.interp(tMI, niid["timesMid"], niid["img"])
        niid["timesMid"] = tMI
        self.__input_function = niid
        return self.__input_function

    @property
    def pet_measurement(self):
        if isinstance(self.__pet_measurement, dict):
            return self.__pet_measurement

        assert os.path.isfile(self.__pet_measurement), f"{self.__pet_measurement} was not found."
        fqfn = self.__pet_measurement
        if self.parse_isotope(fqfn) == "15O":
            self.__pet_measurement = self.decay_uncorrect(self.load_nii(fqfn))
        else:
            self.__pet_measurement = self.load_nii(fqfn)
        return self.__pet_measurement

    def plot_truths(self, truths=None):
        if truths is None:
            truths = self.truths
        data = self.data(truths)
        rho_pred, t_pred, _, _ = self.signalmodel(data)

        petm = self.pet_measurement
        t_petm = petm["timesMid"]
        rho_petm = np.mean(petm["img"], axis=0)
        M0 = np.max(rho_petm)

        inputf = self.input_function
        t_inputf = inputf["timesMid"]
        rho_inputf = inputf["img"]
        I0 = M0 / np.max(rho_inputf)

        plt.figure(figsize=(12, 8))
        p1, = plt.plot(t_inputf, I0 * rho_inputf, color="black", linewidth=2, alpha=0.7,
                 label=f"input function x {I0:.3}")
        p2, = plt.plot(t_petm, rho_petm, color="black", marker="+", ls="none", alpha=0.9, markersize=16,
                 label="measured TAC")
        p3, = plt.plot(t_pred, M0 * rho_pred, marker="o", color="red", ls="none", alpha=0.8,
                 label="predicted TAC")
        plt.xlim([-0.1, 1.1 * np.max(t_petm)])
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel("activity (Bq/mL)")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()

    def plot_variations(self, tindex=0, tmin=None, tmax=None, truths=None):
        if truths is None:
            truths = self.truths

        plt.figure(figsize=(12, 7.4))

        ncolors: int = 100
        viridis = cm.get_cmap("viridis", ncolors)
        dt = (tmax - tmin) / ncolors
        trange = np.arange(tmin, tmax, dt)
        for tidx, t in enumerate(trange):
            truths[tindex] = t
            data = self.data(truths)
            rho, timesMid, _, _ = self.signalmodel(data)
            plt.plot(timesMid, rho, color=viridis(tidx))

        # plt.xlim([-0.1,])
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel("activity (arbitrary)")

        # Add a colorbar to understand colors
        # First create a mappable object with the same colormap
        sm = plt.cm.ScalarMappable(cmap=viridis)
        sm.set_array(trange)
        plt.colorbar(sm, label="Varying " + self.labels[tindex])

        plt.tight_layout()

    def prior_transform(self, prior_tag):
        return {
            "Raichle1983Model": self.prior_transform_raichle,
            "Mintun1984Model": self.prior_transform_mintun,
            "Huang1980Model": self.prior_transform_huang
        }.get(prior_tag, self.prior_transform_ichise)

    def run_nested(self, checkpoint_file=None):
        """ checkpoint_file=self.fqfp+"_dynesty-RadialArtery.save");
        default implementation expects vector for RHO """

        class_name = self.__class__.__name__
        return self.solver.run_nested(prior_tag=class_name,
                                      ndim=self.ndim,
                                      checkpoint_file=checkpoint_file)

    def save_results(self, res: dyutils.Results):
        """ default implementation expects vector for RHO """

        petm = self.pet_measurement
        M0 = np.max(petm["img"])
        qm, ql, qh = self.solver.quantile(res)
        data = self.data(qm)
        rho_pred, _, _, _ = self.signalmodel(data)

        product = deepcopy(petm)
        product["img"] = M0*rho_pred
        fqfp1 = self.fqfp + "_dynesty-" + self.__class__.__name__
        self.save_nii(product, fqfp1 + "-signal.nii.gz")

        d_quantiles = {
            "label": self.labels,
            "qm": qm,
            "ql": ql,
            "qh": qh}
        df = pd.DataFrame(d_quantiles)
        df.to_csv(fqfp1 + "-quantiles.csv")

    @staticmethod
    def decay_correct(tac: dict):
        img = tac["img"] * np.power(2, tac["timesMid"]/tac["halflife"])
        tac["img"] = img
        return tac

    @staticmethod
    def decay_uncorrect(tac: dict):
        img = tac["img"] * np.power(2, -tac["timesMid"]/tac["halflife"])
        tac["img"] = img
        return tac

    @staticmethod
    def prior_transform_raichle(u):
        v = u
        v[0] = u[0] * 0.0149 + 0.0022  # f (1/s)
        v[1] = u[1] + 0.5  # \lambda (cm^3/mL)
        v[2] = u[2] * 0.0212 + 0.0081  # ps (mL cm^{-3}s^{-1})
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * SIGMA  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_mintun(u):
        v = u
        v[0] = u[0] * 0.6 + 0.14  # OEF
        v[1] = u[1] * 0.6 + 0.2  # frac. water of metab. at 90 s
        v[2] = u[2] * 0.5 + 0.5  # v_{post} + 0.5 v_{cap}
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * SIGMA  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_huang(u):
        v = u
        v[0] = u[0] * 0.5  # K_1 (mL cm^{-3}s^{-1})
        v[1] = u[1] * 0.02  # k_2 (1/s)
        v[2] = u[2] * 0.01  # k_3 (1/s)
        v[3] = u[3] * 0.001  # k_4 (1/s)
        v[4] = u[4] * 20  # t_0 (s)
        v[5] = u[5] * SIGMA  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_ichise(u):
        v = u
        v[0] = u[0] * 0.5  # K_1 (mL/cm^{-3}s^{-1})
        v[1] = v[1] * 0.02  # k_2 (1/s)
        v[2] = v[2] * 0.01  # k_3 (1/s)
        v[3] = v[3] * 0.001  # k_4 (1/s)
        v[4] = v[4] * 9.9 + 0.1  # V_P (mL/cm^{-3})
        v[5] = v[5] * 49 + 1  # V_N + V_S (mL/cm^{-3})
        v[6] = u[6] * 20  # t_0 (s)
        v[7] = u[7] * SIGMA  # sigma ~ fraction of M0
        return v
