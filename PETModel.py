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

# general & system functions
from abc import ABC
import os
import sys
from copy import deepcopy
import inspect
import warnings

# basic numeric setup
import numpy as np
import pandas as pd

# NIfTI support
import json
import nibabel as nib


class PETModel(DynestyModel, ABC):
    """
    """

    def __init__(self,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 time_last=None,
                 tag=""):
        super().__init__(sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         tag=tag)
        self.HOME = home
        self.TIME_LAST = time_last

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
    def decay_correct(tac: dict):
        _tac = deepcopy(tac)
        img = _tac["img"] * np.power(2, _tac["timesMid"] / _tac["halflife"])
        _tac["img"] = img
        return _tac

    @staticmethod
    def decay_uncorrect(tac: dict):
        _tac = deepcopy(tac)
        img = _tac["img"] * np.power(2, -_tac["timesMid"] / _tac["halflife"])
        _tac["img"] = img
        return _tac

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
        img = np.squeeze(img)  # all singleton dimensions removed
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
        niid = self.trim_nii_dict(niid, self.TIME_LAST)
        return niid

    @staticmethod
    def parse_halflife(fqfp: str):
        iso = PETModel.parse_isotope(fqfp)
        if iso == "15O":
            return 122.2416  # sec
        if iso == "11C":
            return 20.340253 * 60  # sec
        if iso == "18F":
            return 1.82951 * 3600  # sec
        raise ValueError(f"tracer and halflife not identifiable from fqfp {fqfp}")

    @staticmethod
    def parse_isotope(name: str):
        if "trc-co" in name or "trc-oc" in name or "trc-oo" in name or "trc-ho" in name:
            return "15O"
        if "trc-cglc" in name or "trc-cs1p1" in name:
            return "11C"
        if "trc-fdg" in name or "trc-tz3108" in name or "trc-asem" in name or "trc-azan" in name or "trc-vat" in name:
            return "18F"
        warnings.warn(f"tracer and isotope not identifiable from name {name}", RuntimeWarning)
        return "18F"

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

    def slice_parc(self, img: np.array, xindex: int):
        img1 = img.copy()
        if img1.ndim == 1:
            return img1
        elif img1.ndim == 2:
            return img1[xindex]
        else:
            raise RuntimeError(self.__class__.__name__ + ".__slice_parc: img1.ndim -> " + img1.ndim)

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
        """ examines niid and
            (i) removes inviable temporal samples indicated by np.isnan(timesMid)
            (ii) removes temporal samples occurring after time_last
            (iii) trims all niid contents that appear to have temporal samples """

        if not isinstance(niid, dict):
            raise TypeError(f"Expected niid to be dict but it has type {type(niid)}.")

        img = niid["img"]
        timesMid = niid["timesMid"]
        taus = niid["taus"]
        times = niid["times"]
        viable = ~np.isnan(timesMid)
        
        if time_last is not None:
            viable = viable * (timesMid <= time_last)

        if img.ndim == 1 and len(img) == len(timesMid):
            niid.update({"img": img[viable], "timesMid": timesMid[viable], "taus": taus[viable],
                          "times": times[viable]})
        elif img.ndim == 2 and img.shape[1] == len(timesMid):
            niid.update({"img": img[:, viable], "timesMid": timesMid[viable], "taus": taus[viable],
                          "times": times[viable]})

        return niid
