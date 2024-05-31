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

from PETModel import PETModel
from Boxcar import Boxcar
from RadialArtery import RadialArtery
from TrivialArtery import TrivialArtery

# general & system functions
import glob
import os
import sys
import pickle
from copy import deepcopy
from pprint import pprint
import warnings
import inspect

# basic numeric setup
import numpy as np

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


def kernel_fqfn(artery_fqfn: str):
    """
    :param artery_fqfn: The fully qualified file name of an artery.
    :return: The corresponding fully qualified file name of the kernel to use for that artery.
    """
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


class TCModel(PETModel, ABC):
    """Class documentation for TCModel.

    :class: TCModel
    :inheritance: PETModel, ABC

    This class represents a TC Model used for PET imaging analysis.

    Attributes:
        _input_function (str): The fully qualified file path to the input function file.
        _pet_measurement (str or dict): The fully qualified file path to the PET measurement file or a dictionary containing the PET measurement data.
        truths (list, optional): A list of ground truth values.
        home (str): The home directory for the model (default: current working directory).
        sample (str): The sample type (default: "rslice").
        nlive (int): The number of live points for dynesty sampling (default: 1000).
        rstate (np.random.Generator): The random state for dynesty sampling (default: np.random.default_rng(916301)).
        tag (str): A tag for the model (default: "").

    """
    sigma = None

    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 recovery_coefficient=1.8509,
                 rstate=np.random.default_rng(916301),
                 time_last=None,
                 tag="",
                 delta_time=1):
        super().__init__(home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         time_last=time_last,
                         tag=tag)

        self._input_function = input_function  # fqfn to be converted to dict by property
        self._pet_measurement = pet_measurement  # fqfn to be converted to dict by property
        self._truths_internal = truths
        self.RECOVERY_COEFFICIENT = recovery_coefficient

        self.ARTERY = None
        self.DELTA_TIME = np.round(delta_time)
        petm = self.pet_measurement
        pprint(self.input_function())
        inputf_timesMid = self.input_function()["timesMid"]
        inputf_timesMidInterp = np.arange(0, petm["timesMid"][-1], self.DELTA_TIME)
        self.HALFLIFE = self.input_function()["halflife"]
        self.INPUTF_INTERP = self.input_function()["img"] / np.max(petm["img"])
        self.INPUTF_INTERP = np.interp(inputf_timesMidInterp, inputf_timesMid, self.INPUTF_INTERP)
        self.RHOS = petm["img"] / np.max(petm["img"])
        self.RHO = self.__slice_parc(self.RHOS, 0)
        TCModel.sigma = 0.2
        self.TAUS = petm["taus"]
        self.TIMES_MID = petm["timesMid"]
        self.V1_ASSUMED = np.array(0.05)
        try:
            self.MARTIN_V1 = self.__slice_parc(self.martin_v1_measurement["img"], 0)
        except (FileNotFoundError, TypeError, KeyError):
            self.MARTIN_V1 = self.V1_ASSUMED
        try:
            self.RAICHLE_KS = self.__slice_parc(self.raichle_ks_measurement["img"], 0)
        except (FileNotFoundError, TypeError, KeyError):
            self.RAICHLE_KS = None  # needed by implementations of Raichle1983Model

    @property
    def fqfp(self):
        return self.pet_measurement["fqfp"]

    @property
    def fqfp_results(self):
        fqfp1 = self.fqfp + "-" + self.__class__.__name__ + self.ARTERY.__class__.__name__
        fqfp1 = fqfp1.replace("ParcSchaeffer-reshape-to-schaeffer-", "")
        fqfp1 = fqfp1.replace("ModelAndArtery", "")
        fqfp1 = fqfp1.replace("Model", "")
        fqfp1 = fqfp1.replace("Radial", "")
        return fqfp1

    @property
    def martin_v1_measurement(self):
        subject_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(self._pet_measurement["fqfp"])))
        subject_path = subject_path.replace("sourcedata", "derivatives")

        # useful for  warnings, exceptions
        cname = self.__class__.__name__
        mname = inspect.currentframe().f_code.co_name

        # use existing twilite data
        if isinstance(self.ARTERY, RadialArtery):
            to_glob = subject_path + "/**/*-ParcSchaeffer-reshape-to-schaeffer-schaeffer-twilite_martinv1.nii.gz"
            matches = glob.glob(to_glob, recursive=True)
            if matches and matches[0]:
                return self.load_nii(matches[0])
            else:
                warnings.warn(
                    f"{cname}.{mname}: {to_glob} failed to produce matches",
                    UserWarning)
                warnings.warn(
                    f"{cname}.{mname}: self.ARTERY == {self.ARTERY}, but proceeding to use Boxcar data ",
                    UserWarning)

        # use existing idif data
        to_glob = subject_path + "/**/*-ParcSchaeffer-reshape-to-schaeffer-schaeffer-idif_martinv1.nii.gz"
        matches = glob.glob(to_glob, recursive=True)
        if matches and matches[0]:
            niid = self.load_nii(matches[0])
            niid["img"] = niid["img"] / self.RECOVERY_COEFFICIENT  # v1 has input func. in denom.
            return niid

        # raise FileNotFoundError(
        #     f"{cname}:{mname}: {to_glob} failed to match any usable data files for {type(self.ARTERY)}")

        return None

    @property
    def ndim(self):
        return len(self.labels)

    @property
    def pet_measurement(self):
        if isinstance(self._pet_measurement, dict):
            return deepcopy(self._pet_measurement)

        assert os.path.isfile(self._pet_measurement), f"{self._pet_measurement} was not found."
        fqfn = self._pet_measurement
        if self.parse_isotope(fqfn) == "15O":
            self._pet_measurement = self.decay_uncorrect(self.load_nii(fqfn))
        else:
            self._pet_measurement = self.load_nii(fqfn)
        return deepcopy(self._pet_measurement)

    @property
    def raichle_ks_measurement(self):
        subject_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(self._pet_measurement["fqfp"])))
        subject_path = subject_path.replace("sourcedata", "derivatives")

        # useful for  warnings, exceptions
        cname = self.__class__.__name__
        mname = inspect.currentframe().f_code.co_name

        # use existing twilite data
        if isinstance(self.ARTERY, RadialArtery):
            to_glob = subject_path + f"/**/*-createNiftiMovingAvgFrames-schaeffer-Raichle1983Artery-{self.TAG}-qm.nii.gz"
            matches = glob.glob(to_glob, recursive=True)
            if matches and matches[0]:
                return self.load_nii(matches[0])
            else:
                warnings.warn(
                    f"{cname}.{mname}: {to_glob} failed to produce matches",
                    UserWarning)
                warnings.warn(
                    f"{cname}.{mname}: self.ARTERY == {self.ARTERY}, but proceeding to use Boxcar data ",
                    UserWarning)

        # use existing idif data
        to_glob = subject_path + f"/**/*-createNiftiMovingAvgFrames-schaeffer-Raichle1983Boxcar-{self.TAG}-qm.nii.gz"
        matches = glob.glob(to_glob, recursive=True)
        if matches and matches[0]:
            niid = self.load_nii(matches[0])
            niid["img"] = niid["img"]
            return niid

        # raise FileNotFoundError(
        #    f"{cname}:{mname}: {to_glob} failed to match any usable data files for {type(self.ARTERY)}")

        return None

    @property
    def rho(self):
        return self.RHO

    @property
    def rho2(self):
        return self.RHOS

    @property
    def truths(self):
        return self._truths_internal.copy()

    def __slice_parc(self, img: np.array, xindex: int):
        img1 = img.copy()
        if img1.ndim == 1:
            return img1
        elif img1.ndim == 2:
            return img1[xindex]
        else:
            raise RuntimeError(self.__class__.__name__ + ".__slice_parc: img1.ndim -> " + img1.ndim)

    def data(self, v):
        rhoUsesBoxcar = self.TAUS[2] > self.TIMES_MID[2] - self.TIMES_MID[1]
        return deepcopy({
            "halflife": self.HALFLIFE,
            "rho": self.RHO, "rhos": self.RHOS, "timesMid": self.TIMES_MID, "taus": self.TAUS,
            "times": (self.TIMES_MID - self.TAUS / 2), "inputFuncInterp": self.INPUTF_INTERP,
            "martinv1": self.MARTIN_V1, "raichleks": self.RAICHLE_KS,
            "v": v,
            "rhoUsesBoxcar": rhoUsesBoxcar,
            "delta_time": self.DELTA_TIME})

    def input_function(self):
        """input function read from filesystem, never updated during dynesty operations"""

        if isinstance(self._input_function, dict):
            return deepcopy(self._input_function)

        assert os.path.isfile(self._input_function), f"{self._input_function} was not found."
        fqfn = self._input_function

        if "MipIdif_idif" in fqfn:
            self.ARTERY = Boxcar(
                fqfn,
                truths=self.truths[:14],
                nlive=self.NLIVE,
                times_last=self.TIME_LAST)
        elif "TwiliteKit-do-make-input-func-nomodel" in fqfn:
            self.ARTERY = RadialArtery(
                fqfn,
                kernel_fqfn(fqfn),
                truths=self.truths[:14],
                nlive=self.NLIVE,
                times_last=self.TIME_LAST)
        elif "_proc-" in fqfn and "-aif" in fqfn:
            self.ARTERY = TrivialArtery(
                fqfn,
                times_last=self.TIME_LAST)
        else:
            raise RuntimeError(self.__class__.__name__ + ": does not yet support " + fqfn)

        niid = self.ARTERY.input_func_measurement
        if self.parse_isotope(fqfn) == "15O":
            niid = self.decay_uncorrect(niid)
        if isinstance(self.ARTERY, Boxcar):
            niid["img"] = self.RECOVERY_COEFFICIENT * niid["img"]

        # interpolate to timing domain of pet_measurements
        petm = self.pet_measurement
        tMI = np.arange(0, round(petm["timesMid"][-1]), self.DELTA_TIME)
        niid["img"] = np.interp(tMI, niid["timesMid"], niid["img"])
        niid["timesMid"] = tMI
        self._input_function = niid
        return deepcopy(self._input_function)

    def loglike(self, v):
        data = self.data(v)
        rho_pred, _, _, _ = self.signalmodel(data)
        sigma = v[-1]
        residsq = (rho_pred - data["rho"]) ** 2 / sigma ** 2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))

        if not np.isfinite(loglike):
            loglike = -1e300

        return loglike

    def plot_truths(self, truths=None, parc_index=None, activity_units="Bq/cm^3"):
        if truths is None:
            truths = self.truths
        data = self.data(truths)
        rho_pred, t_pred, _, _ = self.signalmodel(data)

        petm = self.pet_measurement
        t_petm = petm["timesMid"]
        if parc_index:
            petm_hat = petm["img"][parc_index]
        else:
            selected_axes = tuple(np.arange(petm["img"].ndim - 1))
            petm_hat = np.median(petm["img"], axis=selected_axes)  # evaluates mean of petm["img"] for spatial dimensions only
        M0 = np.max(petm["img"])

        inputf = self.input_function()
        t_inputf = inputf["timesMid"]
        inputf_hat = inputf["img"]
        I0 = M0 / np.max(inputf_hat)

        plt.figure(figsize=(12, 8))
        p1, = plt.plot(t_inputf, I0 * inputf_hat, color="black", linewidth=2, alpha=0.7,
                       label=f"input function x {I0:.3}")
        p2, = plt.plot(t_petm, petm_hat, color="black", marker="+", ls="none", alpha=0.9, markersize=16,
                       label=f"measured TAC, parcel {parc_index}")
        p3, = plt.plot(t_pred, M0 * rho_pred, marker="o", color="red", ls="none", alpha=0.8,
                       label=f"predicted TAC, parcel {parc_index}")
        width = np.max((np.max(t_petm), np.max(t_inputf)))
        plt.xlim((-0.1 * width, 1.1 * width))
        plt.xlabel("time of mid-frame (s)")
        plt.ylabel("activity (" + activity_units + ")")
        plt.legend(handles=[p1, p2, p3], loc="right", fontsize=12)
        plt.tight_layout()

    def plot_variations(self, tindex=0, tmin=None, tmax=None, truths=None):
        if truths is None:
            truths = self.truths
        _truths = truths.copy()

        plt.figure(figsize=(12, 7.4))

        ncolors: int = 75
        viridis = cm.get_cmap("viridis", ncolors)
        dt = (tmax - tmin) / ncolors
        trange = np.arange(tmin, tmax, dt)
        for tidx, t in enumerate(trange):
            _truths[tindex] = t
            data = self.data(_truths)
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

    def prior_transform(self):
        return {
            "Martin1987Model": self.prior_transform_martin,
            "Raichle1983Model": self.prior_transform_raichle,
            "Mintun1984Model": self.prior_transform_mintun,
            "Huang1980ModelVenous": self.prior_transform_huang,
            "Huang1980Model": self.prior_transform_huang,
            "Ichise2002Model": self.prior_transform_ichise,
            "Ichise2002VascModel": self.prior_transform_ichise_vasc,
            "LineModel": self.prior_transform_test,
        }.get(self.__class__.__name__, self.prior_transform_huang)
        # default is self.prior_transform_huang for 2-tissue compartment models

    def run_nested(self, checkpoint_file=None, print_progress=False, resume=False):
        """ default: checkpoint_file=self.fqfp+"_dynesty-ModelClass-yyyyMMddHHmmss.save") """

        res = []
        logz = []
        information = []
        qm = []
        ql = []
        qh = []
        martinv1 = []
        raichleks = []
        rho_pred = []
        resid = []

        if self.RHOS.ndim == 1:
            self.RHO = self.RHOS
            tac = self.RHO
            self.MARTIN_V1 = self.V1_ASSUMED
            self.RAICHLE_KS = None

            _res = self.solver.run_nested_for_list(prior_tag=self.__class__.__name__,
                                                   ndim=self.ndim,
                                                   checkpoint_file=checkpoint_file,
                                                   print_progress=print_progress,
                                                   resume=resume)
            if print_progress:
                self.plot_results(_res)
            res.append(_res)
            rd = _res.asdict()
            logz.append(rd["logz"][-1])
            information.append(rd["information"][-1])
            _qm, _ql, _qh = self.solver.quantile(_res)
            qm.append(_qm)
            ql.append(_ql)
            qh.append(_qh)
            martinv1.append(self.MARTIN_V1)
            raichleks.append(self.RAICHLE_KS)
            _rho_pred, _, _, _ = self.signalmodel(self.data(_qm))
            rho_pred.append(_rho_pred)
            resid.append(np.sum(_rho_pred - tac) / np.sum(tac))

            package = {
                "res": res,
                "logz": np.array(logz),
                "information": np.array(information),
                "qm": np.array(qm),
                "ql": np.array(ql),
                "qh": np.array(qh),
                "martinv1": np.array(martinv1),
                "raichleks": np.array(raichleks),
                "rho_pred": np.array(rho_pred),
                "resid": np.array(resid)}

        elif self.RHOS.ndim == 2:
            for tidx, tac in enumerate(self.RHOS):
                self.RHO = tac
                try:
                    self.MARTIN_V1 = self.__slice_parc(self.martin_v1_measurement["img"], tidx)
                except (FileNotFoundError, TypeError, KeyError):
                    self.MARTIN_V1 = self.V1_ASSUMED
                try:
                    self.RAICHLE_KS = self.__slice_parc(self.raichle_ks_measurement["img"], tidx)
                except (FileNotFoundError, TypeError, KeyError):
                    self.RAICHLE_KS = None

                _res = self.solver.run_nested_for_list(prior_tag=self.__class__.__name__,
                                                       ndim=self.ndim,
                                                       checkpoint_file=checkpoint_file,
                                                       print_progress=print_progress,
                                                       resume=resume)
                if print_progress:
                    self.plot_results(_res, tag=f"parc{tidx}", parc_index=tidx)
                res.append(_res)
                rd = _res.asdict()
                logz.append(rd["logz"][-1])
                information.append(rd["information"][-1])
                _qm, _ql, _qh = self.solver.quantile(_res)
                qm.append(_qm)
                ql.append(_ql)
                qh.append(_qh)
                martinv1.append(self.MARTIN_V1)
                raichleks.append(self.RAICHLE_KS)
                _rho_pred, _, _, _ = self.signalmodel(self.data(_qm))
                rho_pred.append(_rho_pred)
                resid.append(np.sum(_rho_pred - tac) / np.sum(tac))

            package = {
                "res": res,
                "logz": np.array(logz),
                "information": np.array(information),
                "qm": np.vstack(qm),
                "ql": np.vstack(ql),
                "qh": np.vstack(qh),
                "martinv1": np.vstack(martinv1),
                "raichleks": np.vstack(raichleks),
                "rho_pred": np.vstack(rho_pred),
                "resid": np.array(resid)}

        else:
            raise RuntimeError(self.__class__.__name__ + ": self.RHOS.ndim -> " + self.RHOS.ndim)

        self.save_results(package, tag=self.TAG)
        return package

    def run_nested_for_indexed_tac(self, tidx: int, checkpoint_file=None, print_progress=False, resume=False):

        self.RHO = self.RHOS[tidx]
        try:
            self.MARTIN_V1 = self.__slice_parc(self.martin_v1_measurement["img"], tidx)
        except (FileNotFoundError, TypeError, KeyError):
            self.MARTIN_V1 = self.V1_ASSUMED
        try:
            self.RAICHLE_KS = self.__slice_parc(self.raichle_ks_measurement["img"], tidx)
        except (FileNotFoundError, TypeError, KeyError):
            self.RAICHLE_KS = None

        _res = self.solver.run_nested_for_list(
            prior_tag=self.__class__.__name__,
            ndim=self.ndim,
            checkpoint_file=checkpoint_file,
            print_progress=print_progress,
            resume=resume)

        if print_progress:
            self.plot_results(_res, tag=f"parc{tidx}", parc_index=tidx)

        _rd = _res.asdict()
        _qm, _ql, _qh = self.solver.quantile(_res)
        _rho_pred, _, _, _ = self.signalmodel(self.data(_qm))
        _resid = np.sum(_rho_pred - self.RHO) / np.sum(self.RHO)

        package = {
            "res": _res,
            "logz": np.array(_rd["logz"][-1]),
            "information": np.array(_rd["information"][-1]),
            "qm": np.array(_qm),
            "ql": np.array(_ql),
            "qh": np.array(_qh),
            "martinv1": np.array(self.MARTIN_V1),
            "raichleks": np.array(self.RAICHLE_KS),
            "rho_pred": np.array(_rho_pred),
            "resid": np.array(_resid)}

        # tag = f"{TCModel.run_nested_for_indexed_tac.__qualname__}_tidx{tidx}"
        # self.save_res_dict(package, tag=tag)
        return package

    def save_res_dict(self, res_dict: dict, tag=""):

        if tag:
            tag = "-" + tag
        fqfp1 = self.fqfp_results + tag

        with open(fqfp1 + "-res.pickle", 'wb') as f:
            pickle.dump(res_dict["res"], f, pickle.HIGHEST_PROTOCOL)

    def save_results(self, res_dict: dict, tag=""):
        """"""

        if tag:
            tag = "-" + tag
        fqfp1 = self.fqfp_results + tag

        with open(fqfp1 + "-res.pickle", 'wb') as f:
            pickle.dump(res_dict["res"], f, pickle.HIGHEST_PROTOCOL)

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

        try:
            # historically prone to fail

            if not TCModel.is_sequence(res_dict["rho_pred"]):
                product = deepcopy(petm)
                product["img"] = M0 * res_dict["rho_pred"]
                self.save_nii(product, fqfp1 + "-rho-pred.nii.gz")

            if not TCModel.is_sequence(res_dict["resid"]):
                product = deepcopy(petm)
                product["img"] = res_dict["resid"]
                self.save_nii(product, fqfp1 + "-resid.nii.gz")

            if not TCModel.is_sequence(res_dict["martinv1"]):
                product = deepcopy(petm)
                product["img"] = res_dict["martinv1"]
                self.save_nii(product, fqfp1 + "-martinv1.nii.gz")
        except Exception as e:
            # catch any error to enable graceful exit while sequentially writing NIfTI files
            print(f"{TCModel.save_results.__name__}: caught Exception {e}, but proceeding", file=sys.stderr)

        # product = deepcopy(petm)
        # product["img"] = res_dict["raichleks"]
        # self.save_nii(product, fqfp1 + "-raichleks.nii.gz")

    @staticmethod
    def is_sequence(obj):
        """ a sequence cannot be multiplied by floating point """
        return isinstance(obj, (list, tuple, type(None)))

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

    @staticmethod
    def prior_transform_martin(u):
        v = u
        return v

    @staticmethod
    def prior_transform_raichle(u):
        v = u
        v[0] = u[0] * 0.016 + 0.0011  # f (1/s)
        v[1] = u[1] * 1.95 + 0.05  # \lambda (cm^3/mL)
        v[2] = u[2] * 0.0272 + 0.0011  # ps (mL cm^{-3}s^{-1})
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * (-60) + 20  # \tau_a (s)
        v[5] = u[5] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_mintun(u):
        v = u
        v[0] = u[0] * 0.6 + 0.2  # OEF
        v[1] = u[1] * 1.8 + 0.1  # frac. water of metab. at 90 s
        v[2] = 0.835  # (v_{post} + 0.5 v_{cap}) / v_1
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * (-60) + 20  # \tau_a (s)
        v[5] = u[5] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_huang(u):
        v = u
        v[0] = u[0] * 2  # k_1 (1/s)
        v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
        v[2] = u[2] * 0.05 + 0.00001  # k_3 (1/s)
        v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
        v[4] = u[4] * 20  # t_0 (s)
        v[5] = u[5] * 120 - 60  # \tau_a (s)
        v[6] = u[6] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_ichise(u):
        v = u
        v[0] = u[0] * 2  # K_1 (mL/cm^{-3}s^{-1})
        v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
        v[2] = u[2] * 0.05 + 0.00001  # k_3 (1/s)
        v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
        v[4] = u[4] * 99.9 + 0.1  # V (mL/cm^{-3}) is total volume := V_N + V_S
        v[5] = u[5] * 120 - 60  # \tau_a (s)
        # v[5] = u[5] * 20  # t_0 (s)
        v[6] = u[6] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_ichise_vasc(u):
        v = u
        v[0] = u[0] * 2  # K_1 (mL/cm^{-3}s^{-1})
        v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
        v[2] = u[2] * 0.05 + 0.00001  # k_3 (1/s)
        v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
        v[4] = u[4] * 0.099 + 0.001  # V_P (mL/cm^{-3})
        v[5] = u[5] * 99.9 + 0.1  # V^\star (mL/cm^{-3}) is total volume := V_P + V_N + V_S
        v[6] = u[6] * 120 - 60  # \tau_a (s)
        # v[6] = u[6] * 20  # t_0 (s)
        v[7] = u[7] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_test(u):
        v = u
        v[0] = u[0] * 2  # intercept
        v[1] = u[1] * 0.5 + 0.00001  # slope
        v[2] = u[2] * TCModel.sigma  # sigma ~ fraction of M0
        return v
