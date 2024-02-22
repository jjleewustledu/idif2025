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

from TCModel import TCModel
from dynesty import utils as dyutils

# general & system functions
import os
from copy import deepcopy
import pickle
from typing import List

# basic numeric setup
import numpy as np
import pandas as pd

RHOS = []
RHO = np.full(110, np.nan)
TAUS = np.full(110, np.nan)
TIMES_MID = np.full(110, np.nan)
MARTIN_V1 = 0.05
INPUTF_INTERP = np.full(119, np.nan)  # uniformly sampled


class Raichle1983Model(TCModel):

    def __init__(self,
                 input_function,
                 pet_measurement,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        super().__init__(input_function,
                         pet_measurement,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate)

        global RHOS, RHO, TAUS, TIMES_MID, MARTIN_V1, INPUTF_INTERP
        petm = self.pet_measurement
        RHOS = petm["img"] / np.max(petm["img"])
        TAUS = petm["taus"]
        TIMES_MID = petm["timesMid"]
        MARTIN_V1 = petm["martinv1"]
        inputf = self.input_function
        INPUTF_INTERP = inputf["img"] / np.max(petm["img"])

    @property
    def labels(self):
        return [
            r"$f$", r"$\lambda$", r"ps", r"$t_0$", r"$\sigma$"]

    @property
    def ndim(self):
        return 5

    @property
    def truths(self):
        return [0.012, 1.192, 0.020, 8.311, 0.027]

    def run_nested(self, checkpoint_file=None, print_progress=False):
        """ checkpoint_file=self.fqfp+"_dynesty-RadialArtery.save") """

        global RHOS, RHO
        res = []
        logz = []
        information = []
        qm = []
        ql = []
        qh = []
        rho_pred = []
        resid = []
        for tac in RHOS:
            RHO = tac
            class_name = self.__class__.__name__
            res__ = self.solver.run_nested_for_list(prior_tag=class_name,
                                                    ndim=self.ndim,
                                                    checkpoint_file=checkpoint_file,
                                                    print_progress=print_progress)
            res.append(res__)

            rd = res__.asdict()
            logz.append(rd["logz"][-1])
            information.append(rd["information"][-1])

            qm__, ql__, qh__ = self.solver.quantile(res__)
            qm.append(qm__)
            ql.append(ql__)
            qh.append(qh__)

            rho_pred__, _, _, _ = self.signalmodel(self.data(qm__))
            rho_pred.append(rho_pred__)
            resid.append(np.sum(rho_pred__ - tac) / np.sum(tac))

        return {"res": res,
                "logz": np.array(logz),
                "information": np.array(information),
                "qm": np.vstack(qm),
                "ql": np.vstack(ql),
                "qh": np.vstack(qh),
                "rho_pred": np.vstack(rho_pred),
                "resid": np.array(resid)}

    def save_results(self, res_dict: dict):
        """"""

        fqfp1 = self.fqfp + "_dynesty-" + self.__class__.__name__
        petm = self.pet_measurement
        M0 = np.max(petm["img"])

        product = deepcopy(petm)
        product["img"] = res_dict["logz"]
        self.save_nii(product, fqfp1 + "-logz.nii.gz")

        product = deepcopy(petm)
        product["img"] = res_dict["information"]
        self.save_nii(product, fqfp1 + "-information.nii.gz")

        product = deepcopy(petm)
        product["img"] = res_dict["qm"]
        self.save_nii(product, fqfp1 + "-qm.nii.gz")

        product = deepcopy(petm)
        product["img"] = res_dict["ql"]
        self.save_nii(product, fqfp1 + "-ql.nii.gz")

        product = deepcopy(petm)
        product["img"] = res_dict["qh"]
        self.save_nii(product, fqfp1 + "-qh.nii.gz")

        product = deepcopy(petm)
        product["img"] = M0 * res_dict["rho_pred"]
        self.save_nii(product, fqfp1 + "-rho-pred.nii.gz")

        product = deepcopy(petm)
        product["img"] = res_dict["resid"]
        self.save_nii(product, fqfp1 + "-resid.nii.gz")

        with open(fqfp1 + "-res.pickle", 'wb') as f:
            pickle.dump(res_dict["res"], f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def data(v):
        return {"rho": RHO, "timesMid": TIMES_MID, "taus": TAUS, "inputFuncInterp": INPUTF_INTERP, "martinv1": MARTIN_V1, "v": v}

    @staticmethod
    def loglike(v):
        data = Raichle1983Model.data(v)
        rho_pred, _, _, _ = Raichle1983Model.signalmodel(data)
        sigma = v[-1]
        residsq = (rho_pred - data["rho"]) ** 2 / sigma ** 2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))

        if not np.isfinite(loglike):
            loglike = -1e300

        return loglike

    @staticmethod
    def signalmodel(data: dict):
        timesMid = data["timesMid"]
        ifi = data["inputFuncInterp"]
        v = data["v"]
        f = v[0]
        lamb = v[1]
        ps = v[2]
        t_0 = v[3]
        E = 1 - np.exp(-ps / f)
        times = np.arange(0, len(ifi))
        kernel = np.exp(-E * f * times / lamb - 0.005670305 * times)
        rho_t = E * f * np.convolve(kernel, ifi, mode="full")
        rho_t = rho_t[:ifi.size] + data["martinv1"] * ifi
        rho_t = Raichle1983Model.slide(rho_t, times, t_0)
        rho = np.interp(timesMid, times, rho_t)
        return rho, timesMid, rho_t, times
