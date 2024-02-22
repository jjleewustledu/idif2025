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
import copy

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


class PETModel(DynestyModel):
    def __init__(self,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        self.home = home
        self.solver = DynestySolver(model=self,
                                    sample=sample,
                                    nlive=nlive,
                                    rstate=rstate)

    @property
    @abstractmethod
    def ndim(self):
        pass

    def load_nii(self, fqfn):
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
            "timesMid": np.array(j["timesMid"], dtype=float).ravel(),
            "taus": np.array(j["taus"], dtype=float).ravel(),
            "times": np.array(j["times"], dtype=float).ravel(),
            "halflife": self.parse_halflife(fqfp)}
        if "martinv1" in j:
            niid["martinv1"] = j["martinv1"]
        if "raichleks" in j:
            niid["raichleks"] = j["raichleks"]
        niid = self._trim_nii_dict(niid)
        return niid

    def plot_results(self, res: dyutils.Results):
        class_name = self.__class__.__name__

        qm, _, _ = self.solver.quantile(res)
        self.plot_truths(qm)
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-results.png")

        dyplot.runplot(res)
        plt.tight_layout()
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-runplot.png")

        fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm,
                                     fig=plt.subplots(self.ndim, 2, figsize=(16, 25)))
        fig.tight_layout()
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-traceplot.png")

        dyplot.cornerplot(res, truths=qm, show_titles=True,
                          title_kwargs={"y": 1.04}, labels=self.labels,
                          fig=plt.subplots(self.ndim, self.ndim, figsize=(100, 100)))
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-cornerplot.png")

    def save_csv(self, data: dict, fqfn=None):
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

        # load img
        nii = copy.deepcopy(data["nii"])  # paranoia
        nii = nib.Nifti1Image(data["img"], nii.affine, nii.header)
        nib.save(nii, fqfn)

        # find json fields of interest
        jfile = data["fqfp"] + ".json"  # from previously loaded data
        with open(jfile, "r") as f:
            j = json.load(f)
        if "timesMid" in data:
            j["timesMid"] = data["timesMid"].tolist()
        else:
            j["timesMid"] = self.data2timesMid(data).tolist()
        if "taus" in data:
            j["taus"] = data["taus"].tolist()
        else:
            j["taus"] = self.data2taus(data).tolist()
        if "times" in data:
            j["times"] = data["times"].tolist()
        else:
            j["times"] = self.data2t(data).tolist()
        base, _ = os.path.splitext(fqfn)  # remove .nii.gz
        fqfp, _ = os.path.splitext(base)
        jfile1 = fqfp + ".json"
        with open(jfile1, "w") as f:
            json.dump(j, f, indent=4)

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
    def parse_halflife(fqfp):
        iso = PETModel.parse_isotope(fqfp)
        if iso == "15O":
            return 122.2416  # sec
        if iso == "18F":
            return 1.82951 * 3600  # sec
        raise ValueError(f"tracer and halflife not identifiable from fqfp {fqfp}")

    @staticmethod
    def parse_isotope(name: str):
        if "trc-co" in name or "trc-oc" in name or "trc-oo" in name or "trc-ho" in name:
            return "15O"
        if "trc-fdg" in name:
            return "18F"
        raise ValueError(f"tracer and halflife not identifiable from name {name}")

    @staticmethod
    def slide(rho, t, dt):
        if dt < 0.1:
            return rho
        return np.interp(t - dt, t, rho)

    @staticmethod
    def _trim_nii_dict(niid: dict):
        if not isinstance(niid, dict):
            raise TypeError(f"niid must be a dict but had type {type(niid)}.")
        img = niid["img"]
        timesMid = niid["timesMid"]
        taus = niid["taus"]
        times = niid["times"]
        viable = ~np.isnan(timesMid)
        early = timesMid <= 180
        selected = viable * early
        if img.ndim == 1:
            niid.update({"img": img[selected], "timesMid": timesMid[selected], "taus": taus[selected],
                         "times": times[selected]})
        else:
            niid.update({"img": img[:, selected], "timesMid": timesMid[selected], "taus": taus[selected], "times": times[selected]})
        return niid
