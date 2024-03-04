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

from abc import ABC

from TCModel import TCModel
from Artery import Artery

# general & system functions
import os
from copy import deepcopy
from pprint import pprint

# basic numeric setup
import numpy as np

# plotting
from matplotlib import pyplot as plt


class TCModelAndArtery(TCModel, ABC):

    def __init__(self,
                 input_function,
                 pet_measurement,
                 kernel=None,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag=""):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate)

        pprint(self.input_function())

    def data(self, v):
        artery_data = self.ARTERY.data(v[:14])
        inputf_interp = self.INPUTF_INTERP.copy()
        _, ARTERY_ideal, t_ideal = self.ARTERY.signalmodel(artery_data)
        if not np.any(np.isnan(ARTERY_ideal)):
            artery_ideal = (ARTERY_ideal.copy() *
                            np.max(self.ARTERY.input_func_measurement["img"]) /
                            np.max(self.pet_measurement["img"]))
            artery_timesMid = self.TIMES_MID[self.TIMES_MID <= t_ideal[-1]]
            inputf_interp[:len(artery_timesMid)] = np.interp(artery_timesMid, t_ideal, artery_ideal)

        return deepcopy({
            "halflife": self.HALFLIFE,
            "rho": self.RHO, "rhos": self.RHOS, "timesMid": self.TIMES_MID, "taus": self.TAUS,
            "times": (self.TIMES_MID - self.TAUS / 2), "inputFuncInterp": inputf_interp,
            "martinv1": self.MARTIN_V1, "raichleks": self.RAICHLE_KS,
            "v": v})

    def loglike(self, v: np.array):
        return super().loglike(v) + self.ARTERY.loglike(v[:14])

    def plot_truths(self, truths=None):
        self.plot_truths_with_ARTERY(truths=truths)
        self.ARTERY.plot_truths(truths=truths)

    def plot_truths_with_ARTERY(self, truths=None):
        if truths is None:
            truths = self.truths
        data = self.data(truths)
        rho_pred, t_pred, _, _ = self.signalmodel(data)

        petm = self.pet_measurement
        t_petm = petm["timesMid"]
        selected_axes = tuple(np.arange(petm["img"].ndim - 1))
        rho_petm = np.mean(petm["img"], axis=selected_axes)  # evaluates mean of petm["img"] for spatial dimensions only
        M0 = np.max(rho_petm)

        data_for_ARTERY = self.ARTERY.data(truths)
        _, rho_inputf, t_inputf = self.ARTERY.signalmodel(data_for_ARTERY)
        I0 = M0 / np.max(rho_inputf)

        plt.figure(figsize=(12, 8))
        p1, = plt.plot(t_inputf, I0 * rho_inputf, color="black", linewidth=2, alpha=0.7,
                       label=f"input function x {I0:.3}")
        p2, = plt.plot(t_petm, rho_petm, color="black", marker="+", ls="none", alpha=0.9, markersize=16,
                       label="measured TAC")
        p3, = plt.plot(t_pred, M0 * rho_pred, marker="o", color="red", ls="none", alpha=0.8,
                       label="predicted TAC")
        plt.xlim([-0.1, 1.1 * np.max([np.max(t_petm), np.max(t_inputf)])])
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel("activity (Bq/mL)")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()

    def prior_transform(self):
        return {
            "Martin1987ModelAndArtery": self.prior_transform_martin,
            "Raichle1983ModelAndArtery": self.prior_transform_raichle,
            "Mintun1984ModelAndArtery": self.prior_transform_mintun,
            "Huang1980ModelAndArtery": self.prior_transform_huang
        }.get(self.__class__.__name__, self.prior_transform_ichise)

    # def save_results(self, res_dict: dict):
    #     super().save_results(res_dict)
    #     self.ARTERY.save_results(res_dict["res"][:14])

    @staticmethod
    def prior_transform_martin(u):
        v = u
        v[:14] = Artery.prior_transform_co(u[:14])
        v[14:] = TCModel.prior_transform_martin(u[14:])
        return v

    @staticmethod
    def prior_transform_raichle(u):
        v = u
        v[:14] = Artery.prior_transform_default(u[:14])
        v[14:] = TCModel.prior_transform_raichle(u[14:])
        return v

    @staticmethod
    def prior_transform_mintun(u):
        v = u
        v[:14] = Artery.prior_transform_oo(u[:14])
        v[14:] = TCModel.prior_transform_mintun(u[14:])
        return v

    @staticmethod
    def prior_transform_huang(u):
        v = u
        v[:14] = Artery.prior_transform_default(u[:14])
        v[14:] = TCModel.prior_transform_huang(u[14:])
        return v

    @staticmethod
    def prior_transform_ichise(u):
        v = u
        v[:14] = Artery.prior_transform_default(u[:14])
        v[14:] = TCModel.prior_transform_ichise(u[14:])
        return v
