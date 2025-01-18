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

from __future__ import absolute_import
from Boxcar import Boxcar
from TissueModel import TissueModel
from RadialArtery import RadialArtery
from PETUtilities import PETUtilities

# general & system functions
import glob
import inspect
import os
import warnings
from abc import ABC
from copy import deepcopy

# basic numeric setup
import numpy as np


class TCModel(TissueModel, ABC):
    """
    """

    sigma = None  # class attribute needed by dynesty

    def __init__(self, input_function, pet_measurement, **kwargs):
        super().__init__(input_function, pet_measurement, **kwargs)
        TCModel.sigma = 0.2
        self.__martin_v1_measurement = None
        self.__raichle_ks_measurement = None

    @property
    def martin_v1_measurement(self):
        if self.__martin_v1_measurement:
            return self.__martin_v1_measurement

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
                self.__martin_v1_measurement = self.load_nii(matches[0])
                return self.__martin_v1_measurement
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
            self.__martin_v1_measurement = self.load_nii(matches[0])
            self.__martin_v1_measurement["img"] = self.__martin_v1_measurement["img"] / self.RECOVERY_COEFFICIENT  # v1 has input func. in denom.
            return self.__martin_v1_measurement

        self.__martin_v1_measurement = None
        return self.__martin_v1_measurement

    @property
    def raichle_ks_measurement(self):
        if self.__raichle_ks_measurement:
            return self.__raichle_ks_measurement

        subject_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(self._pet_measurement["fqfp"])))
        subject_path = subject_path.replace("sourcedata", "derivatives")

        # useful for  warnings, exceptions
        cname = self.__class__.__name__
        mname = inspect.currentframe().f_code.co_name

        # use existing twilite data
        if isinstance(self.ARTERY, RadialArtery):
            to_glob = (subject_path +
                       f"/**/*-createNiftiMovingAvgFrames-schaeffer-Raichle1983Artery-{self.tag}-qm.nii.gz")
            matches = glob.glob(to_glob, recursive=True)
            if matches and matches[0]:
                self.__raichle_ks_measurement = self.load_nii(matches[0])
                return self.__raichle_ks_measurement
            else:
                warnings.warn(
                    f"{cname}.{mname}: {to_glob} failed to produce matches",
                    UserWarning)
                warnings.warn(
                    f"{cname}.{mname}: self.ARTERY == {self.ARTERY}, but proceeding to use Boxcar data ",
                    UserWarning)

        # use existing idif data
        to_glob = subject_path + f"/**/*-createNiftiMovingAvgFrames-schaeffer-Raichle1983Boxcar-{self.tag}-qm.nii.gz"
        matches = glob.glob(to_glob, recursive=True)
        if matches and matches[0]:
            self.__raichle_ks_measurement = self.load_nii(matches[0])
            return self.__raichle_ks_measurement

        # data not available
        self.__raichle_ks_measurement = None
        return self.__raichle_ks_measurement

    def data(self, v):
        rho_experiences_boxcar = isinstance(self.ARTERY, Boxcar)
        v1 = PETUtilities.slice_parc(self.martin_v1_measurement["img"], self._parc_index)
        ks = PETUtilities.slice_parc(self.raichle_ks_measurement["img"], self._parc_index)
        return deepcopy({
            "halflife": self.HALFLIFE,
            "rho": self.RHO, "rhos": self.rhos, "timesMid": self.TIMES_MID, "taus": self.TAUS,
            "times": (self.TIMES_MID - self.TAUS / 2), "inputFuncInterp": self.INPUTF_INTERP,
            "martinv1": v1, 
            "raichleks": ks,
            "v": v,
            "rho_experiences_boxcar": rho_experiences_boxcar,
            "parc_index": self._parc_index,
            "delta_time": self.DELTA_TIME})
    
    def martin_v1(self, parc_index: int = 0) -> np.ndarray:
        try:
            return PETUtilities.slice_parc(self.martin_v1_measurement["img"], parc_index)
        except (FileNotFoundError, TypeError, KeyError):
            return np.array(0.05)

    def raichle_ks(self, parc_index: int = 0) -> np.ndarray:
        try:
            return PETUtilities.slice_parc(self.raichle_ks_measurement["img"], parc_index)
        except (FileNotFoundError, TypeError, KeyError):
            return None

    def prior_transform(self):
        return {
            "Martin1987Model": TCModel.prior_transform_martin,
            "Raichle1983Model": TCModel.prior_transform_raichle,
            "Mintun1984Model": TCModel.prior_transform_mintun,
            "Huang1980ModelVenous": TCModel.prior_transform_huang,
            "Huang1980Model": TCModel.prior_transform_huang,
            "Ichise2002Model": TCModel.prior_transform_ichise,
            "Ichise2002VascModel": TCModel.prior_transform_ichise_vasc,
            "Ichise2002PosthocModel": TCModel.prior_transform_ichise_posthoc,
            "LineModel": TCModel.prior_transform_test,
        }.get(self.__class__.__name__, TCModel.prior_transform_huang)
        # default is self.prior_transform_huang for 2-tissue compartment models

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
        v[0] = u[0] * 2  # k_1 (1/s)
        v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
        v[2] = u[2] * 0.05 + 0.00001  # k_3 (1/s)
        v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
        v[4] = u[4] * 999.9 + 0.1  # V (mL/cm^{-3}) is total volume := V_N + V_S
        v[5] = u[5] * 120 - 60  # \tau_a (s)
        # v[5] = u[5] * 20  # t_0 (s)
        v[6] = u[6] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_ichise_vasc(u):
        v = u
        v[0] = u[0] * 2  # k_1 (1/s)
        v[1] = u[1] * 0.5 + 0.00001  # k_2 (1/s)
        v[2] = u[2] * 0.05 + 0.00001  # k_3 (1/s)
        v[3] = u[3] * 0.05 + 0.00001  # k_4 (1/s)
        v[4] = u[4] * 0.099 + 0.001  # V_P (mL/cm^{-3})
        v[5] = u[5] * 9999.9 + 0.1  # V^\star (mL/cm^{-3}) is total volume := V_P + V_N + V_S
        v[6] = u[6] * 120 - 60  # \tau_a (s)
        # v[6] = u[6] * 20  # t_0 (s)
        v[7] = u[7] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_ichise_posthoc(u):
        raise NotImplementedError(
            f"{TCModel.prior_transform_ichise_posthoc.__name__} requires overriding by class Ichise2002PosthocModel.")

    @staticmethod
    def prior_transform_martin(u):
        v = u
        return v

    @staticmethod
    def prior_transform_mintun(u):
        v = u
        v[0] = u[0] * 0.8 + 0.1  # OEF
        v[1] = u[1] * 1.8 + 0.1  # frac. water of metab. at 90 s
        v[2] = u[2] * 0.9 + 0.1  # {v_{post} + 0.5 v_{cap}} / v_1
        v[3] = u[3] * 20  # t_0 (s)
        v[4] = u[4] * (-60)  # \tau_a (s)
        v[5] = u[5] * 20  # \tau_d (s)
        v[6] = u[6] * TCModel.sigma  # sigma ~ fraction of M0
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
    def prior_transform_test(u):
        v = u
        v[0] = u[0] * 2  # intercept
        v[1] = u[1] * 0.5 + 0.00001  # slope
        v[2] = u[2] * TCModel.sigma  # sigma ~ fraction of M0
        return v

    def run_nested(self, checkpoint_file=None, print_progress=False, resume=False):
        """ default: checkpoint_file=self.fqfp+"_dynesty-ModelClass-yyyyMMddHHmmss.save") """

        res_ = []
        logz_ = []
        information_ = []
        qm_ = []
        ql_ = []
        qh_ = []
        v1_ = []
        ks_ = []
        rho_pred_ = []
        resid_ = []

        if self.rhos.ndim == 1:
            self.RHO = self.rhos
            tac = self.RHO
            _res_ = self.solver.run_nested_for_list(prior_tag=self.__class__.__name__,
                                                   ndim=self.ndim,
                                                   checkpoint_file=checkpoint_file,
                                                   print_progress=print_progress,
                                                   resume=resume)

            if print_progress:
                self.plot_results(_res_)

            res_.append(_res_)
            rd = _res_.asdict()
            logz_.append(rd["logz"][-1])
            information_.append(rd["information"][-1])
            _qm_, _ql_, _qh_ = self.solver.quantile(_res_)
            qm_.append(_qm_)
            ql_.append(_ql_)
            qh_.append(_qh_)
            v1_.append(self.martin_v1())
            ks_.append(self.raichle_ks())
            _rho_pred_, _, _, _ = self.signalmodel(self.data(_qm_))
            rho_pred_.append(_rho_pred_)
            resid_.append(np.sum(_rho_pred_ - tac) / np.sum(tac))
            package = {
                "res": res_,
                "logz": np.array(logz_),
                "information": np.array(information_),
                "qm": np.array(qm_),
                "ql": np.array(ql_),
                "qh": np.array(qh_),
                "martinv1": np.array(v1_),
                "raichleks": np.array(ks_),
                "rho_pred": np.array(rho_pred_),
                "resid": np.array(resid_)}

        elif self.rhos.ndim == 2:
            for tidx, tac in enumerate(self.rhos):
                self.RHO = tac
                _res_ = self.solver.run_nested_for_list(prior_tag=self.__class__.__name__,
                                                       ndim=self.ndim,
                                                       checkpoint_file=checkpoint_file,
                                                       print_progress=print_progress,
                                                       resume=resume)

                # if print_progress:
                self.plot_results(_res_, tag=f"parc{tidx}", parc_index=tidx)

                res_.append(_res_)
                rd = _res_.asdict()
                logz_.append(rd["logz"][-1])
                information_.append(rd["information"][-1])
                _qm_, _ql_, _qh_ = self.solver.quantile(_res_)
                qm_.append(_qm_)
                ql_.append(_ql_)
                qh_.append(_qh_)
                v1_.append(self.martin_v1(parc_index=tidx))
                ks_.append(self.raichle_ks(parc_index=tidx))
                _rho_pred_, _, _, _ = self.signalmodel(self.data(_qm_))
                rho_pred_.append(_rho_pred_)
                resid_.append(np.sum(_rho_pred_ - tac) / np.sum(tac))
            package = {
                "res": res_,
                "logz": np.array(logz_),
                "information": np.array(information_),
                "qm": np.vstack(qm_),
                "ql": np.vstack(ql_),
                "qh": np.vstack(qh_),
                "martinv1": np.vstack(v1_),
                "raichleks": np.vstack(ks_),
                "rho_pred": np.vstack(rho_pred_),
                "resid": np.array(resid_)}

        else:
            raise RuntimeError(self.__class__.__name__ + ": self.rhos.ndim -> " + self.rhos.ndim)

        self.save_results(package, tag=self.tag)
        return package

    def run_nested_for_indexed_tac(self, tidx: int, checkpoint_file=None, print_progress=False, resume=False):

        self.RHO = self.rhos[tidx]
        res_ = self.solver.run_nested_for_list(
            prior_tag=self.__class__.__name__,
            ndim=self.ndim,
            checkpoint_file=checkpoint_file,
            print_progress=print_progress,
            resume=resume)

        # if print_progress:
        self.plot_results(res_, tag=f"parc{tidx}", parc_index=tidx)

        rd_ = res_.asdict()
        qm_, ql_, qh_ = self.solver.quantile(res_)
        rho_pred_, _, _, _ = self.signalmodel(self.data(qm_))
        resid_ = np.sum(rho_pred_ - self.RHO) / np.sum(self.RHO)
        package = {
            "res": res_,
            "logz": np.array(rd_["logz"][-1]),
            "information": np.array(rd_["information"][-1]),
            "qm": np.array(qm_),
            "ql": np.array(ql_),
            "qh": np.array(qh_),
            "martinv1": np.array(self.martin_v1(parc_index=tidx)),
            "raichleks": np.array(self.raichle_ks(parc_index=tidx)),
            "rho_pred": np.array(rho_pred_),
            "resid": np.array(resid_)}
        return package
