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
from Boxcar import Boxcar
from RadialArtery import RadialArtery

# general & system functions
import os
from copy import deepcopy

# basic numeric setup
import numpy as np


def kernel_fqfn(artery_fqfn: str):
    sourcedata = os.path.join(os.getenv("SINGULARITY_HOME"), "CCIR_01211", "sourcedata",)
    if "sub-108293" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=46.8.nii.gz")
    if "sub-108237" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=43.9.nii.gz")
    if "sub-108254" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=37.9.nii.gz")
    if "sub-108250" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=42.8.nii.gz")
    if "sub-108284" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=39.7.nii.gz")
    if "sub-108306" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=41.1.nii.gz")

    # mean hct for females and males
    return os.path.join(sourcedata, "kernel_hct=44.5.nii.gz")


class TCModelAndArtery(TCModel, ABC):

    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate)

        if "Mipidif_idif" in input_function:
            self.ARTERY = Boxcar(
                input_function,
                truths=self.truths[:14],
                nlive=self.NLIVE)
        elif "TwliiteKit-do-make-input-func-nomodel_inputfunc" in input_function:
            self.ARTERY = RadialArtery(
                input_function,
                kernel_fqfn(input_function),
                truths=self.truths[:14],
                nlive=self.NLIVE)
        else:
            raise RuntimeError(self.__class__.__name__ + ": does not yet support " + input_function)

    @property
    def truths(self):
        return np.concatenate((self.ARTERY.truths, super().truths[14:]))

    def data(self, v):
        artery_data = self.ARTERY.data(v[:14])
        artery_signal = self.ARTERY.signalmodel(artery_data)
        inputf_interp = self.INPUTF_INTERP.copy()
        inputf_interp[:len(artery_signal)] = artery_signal
        return deepcopy({
            "rho": self.RHO, "rhos": self.RHOS, "timesMid": self.TIMES_MID, "taus": self.TAUS,
            "times": (self.TIMES_MID - self.TAUS / 2), "inputFuncInterp": inputf_interp,
            "martinv1": self.MARTIN_V1, "raichleks": self.RAICHLE_KS,
            "v": v})

    def input_function(self):
        """input function read from filesystem, never updated during dynesty operations"""
        
        if isinstance(self.__input_function, dict):
            return deepcopy(self.__input_function)

        assert os.path.isfile(self.__input_function), f"{self.__input_function} was not found."
        fqfn = self.__input_function

        nii = self.ARTERY.input_func_measurement
        if self.parse_isotope(fqfn) == "15O":
            niid = self.decay_uncorrect(nii)
        else:
            niid = nii

        # interpolate to timing domain of pet_measurements
        petm = self.pet_measurement
        tMI = np.arange(0, round(petm["timesMid"][-1]))
        niid["img"] = np.interp(tMI, niid["timesMid"], niid["img"])
        niid["timesMid"] = tMI
        self.__input_function = niid
        return deepcopy(self.__input_function)

    def loglike(self, v: np.array):
        return super().loglike(v[14:]) + self.ARTERY.loglike(v[:14])

    def plot_truths(self, truths=None):
        if truths is None:
            super().plot_truths()
            self.ARTERY.plot_truths()
        else:
            super().plot_truths(truths=truths[14:])
            self.ARTERY.plot_truths(truths=truths[:14])

    def prior_transform(self):
        return {
            "Martin1987ModelAndArtery": self.prior_transform_martin,
            "Raichle1983ModelAndArtery": self.prior_transform_raichle,
            "Mintun1984ModelAndArtery": self.prior_transform_mintun,
            "Huang1980ModelAndArtery": self.prior_transform_huang
        }.get(self.__class__.__name__, self.prior_transform_ichise)

    def save_results(self, res_dict: dict):
        super().save_results(res_dict)
        self.ARTERY.save_results(res_dict["res"][:14])

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
