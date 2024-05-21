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

from DynestyModel import DynestyModel
from DynestySolver import DynestySolver
from dynesty import utils as dyutils
from dynesty import plotting as dyplot

# general & system functions
from abc import abstractmethod
import os
import sys
from copy import deepcopy
import traceback
import inspect

# basic numeric setup
import numpy as np
import pandas as pd

# NIfTI support
import json
import nibabel as nib

# plotting
from matplotlib import pyplot as plt

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


class PETModel(DynestyModel):
    """

    """
    def __init__(self,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 time_last=None,
                 tag=""):
        self.home = home
        self.solver = DynestySolver(model=self,
                                    sample=sample,
                                    nlive=nlive,
                                    rstate=rstate)
        self.NLIVE = nlive
        self.TIME_LAST = time_last
        self.TAG = tag

    @property
    @abstractmethod
    def ndim(self):
        pass

    def load_nii(self, fqfn):
        if not os.path.isfile(fqfn):
            return {}

        # load img
        nii = nib.load(fqfn)
        img = nii.get_fdata()

        # find json fields of interest
        base, _ = os.path.splitext(fqfn)
        fqfp, _ = os.path.splitext(base)
        jfile = fqfp + ".json"
        with open(jfile, "r") as f:
            j = json.load(f)

        # assemble dict
        if img.shape[0] == 1:
            img = np.array(img, dtype=float).ravel()
        niid = {
            "fqfp": fqfp,
            "nii": nii,
            "img": img,
            "halflife": self.parse_halflife(fqfp)}
        if "timesMid" in j:
            niid["timesMid"] = np.array(j["timesMid"], dtype=float).ravel()
        if "taus" in j:
            niid["taus"] = np.array(j["taus"], dtype=float).ravel()
        if "times" in j:
            niid["times"] = np.array(j["times"], dtype=float).ravel()
        if "martinv1" in j:
            niid["martinv1"] = np.array(j["martinv1"])
        if "raichleks" in j:
            niid["raichleks"] = np.array(j["raichleks"])
        if self.TIME_LAST:
            niid = self.trim_nii_dict(niid, self.TIME_LAST)
        return deepcopy(niid)

    def plot_results(self, res: dyutils.Results, tag="", parc_index=None):

        if not tag and parc_index:
            tag = f"parc{parc_index}"
        if tag:
            tag = "-" + tag
        fqfp1 = self.fqfp_results + tag
        qm, _, _ = self.solver.quantile(res)

        try:
            self.plot_truths(qm, parc_index=parc_index)
            plt.savefig(fqfp1 + "-results.png")
        except Exception as e:
            print("PETModel.plot_results: caught an Exception: ", str(e))
            traceback.print_exc()

        try:
            dyplot.runplot(res)
            plt.tight_layout()
            plt.savefig(fqfp1 + "-runplot.png")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.runplot: caught a ValueError: {e}")

        try:
            fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm,
                                         fig=plt.subplots(self.ndim, 2, figsize=(16, 50)))
            fig.tight_layout()
            plt.savefig(fqfp1 + "-traceplot.png")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.traceplot: caught a ValueError: {e}")

        try:
            dyplot.cornerplot(res, truths=qm, show_titles=True,
                              title_kwargs={"y": 1.04}, labels=self.labels,
                              fig=plt.subplots(self.ndim, self.ndim, figsize=(100, 100)))
            plt.savefig(fqfp1 + "-cornerplot.png")
        except ValueError as e:
            print(f"PETModel.plot_results.dyplot.cornerplot: caught a ValueError: {e}")

    def save_csv(self, data: dict, fqfn=None):
        """ """
        if not fqfn:
            fqfn = self.fqfp + "_dynesty-" + self.__class__.__name__ + ".csv"
        d_nii = {
            "taus": data["taus"],
            "timesMid": data["timesMid"],
            "times": data["times"],
            "img": data["img"]}
        df = pd.DataFrame(d_nii)
        df.to_csv(fqfn)

    def save_nii(self, data: dict, fqfn=None):
        if not fqfn:
            fqfn = self.fqfp + "_dynesty-" + self.__class__.__name__ + ".nii.gz"

        # useful for  warnings, exceptions
        cname = self.__class__.__name__
        mname = inspect.currentframe().f_code.co_name

        try:
            # load img
            _data = deepcopy(data)
            nii = _data["nii"]  # paranoia
            nii = nib.Nifti1Image(_data["img"], nii.affine, nii.header)
            nib.save(nii, fqfn)

            # find json fields of interest
            jfile = _data["fqfp"] + ".json"  # from previously loaded tindices
            with open(jfile, "r") as f:
                j = json.load(f)
            if "timesMid" in _data:
                j["timesMid"] = _data["timesMid"].tolist()
            else:
                j["timesMid"] = self.data2timesMid(_data).tolist()
            if "taus" in _data:
                j["taus"] = _data["taus"].tolist()
            else:
                j["taus"] = self.data2taus(_data).tolist()
            if "times" in _data:
                j["times"] = _data["times"].tolist()
            else:
                j["times"] = self.data2t(_data).tolist()
            base, _ = os.path.splitext(fqfn)  # remove .nii.gz
            fqfp, _ = os.path.splitext(base)
            jfile1 = fqfp + ".json"
            with open(jfile1, "w") as f:
                json.dump(j, f, indent=4)
        except Exception as e:
            # catch any error to enable graceful exit while sequentially writing NIfTI files
            print(f"{cname}.{mname}: caught Exception {e}, but proceeding", file=sys.stderr)

    @staticmethod
    def data2t(data: dict):
        timesMid = data["timesMid"]
        taus = data["taus"]
        t0 = timesMid[0] - taus[0] / 2
        tF = timesMid[-1] + taus[-1] / 2
        t = np.arange(t0, tF)
        return t

    @staticmethod
    def data2taus(data: dict):
        timesMid = data["timesMid"]
        times = data["times"]
        return 2 * (timesMid - times)

    @staticmethod
    def data2timesMid(data: dict):
        times = data["times"]
        taus = data["taus"]
        return times + taus / 2

    @staticmethod
    @abstractmethod
    def parse_halflife(fqfp: str):
        pass

    @staticmethod
    @abstractmethod
    def parse_isotope(name: str):
        pass

    @staticmethod
    def slide(rho, t, dt, halflife=None):
        if abs(dt) < 0.1:
            return rho
        rho = np.interp(t - dt, t, rho)
        if halflife:
            return rho * np.power(2, -dt / halflife)
        else:
            return rho

    @staticmethod
    def trim_nii_dict(niid: dict, time_last=None):
        if not isinstance(niid, dict):
            raise TypeError(f"niid must be a dict but had type {type(niid)}.")

        _niid = deepcopy(niid)
        img = _niid["img"]
        timesMid = _niid["timesMid"]
        taus = _niid["taus"]
        times = _niid["times"]
        viable = ~np.isnan(timesMid)
        if time_last is not None:
            selected = viable * (timesMid <= time_last)
        else:
            selected = viable
        if img.ndim == 1:
            _niid.update({"img": img[selected], "timesMid": timesMid[selected], "taus": taus[selected],
                         "times": times[selected]})
        else:
            _niid.update({"img": img[:, selected], "timesMid": timesMid[selected], "taus": taus[selected],
                         "times": times[selected]})
        return _niid
