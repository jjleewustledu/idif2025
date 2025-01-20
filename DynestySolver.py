# The MIT License (MIT)
#
# Copyright (c) 2024 - Present: John J. Lee.
# Copyright (c) 2017 - Present: Josh Speagle and contributors.
# Copyright (c) 2014 - 2017: Kyle Barbary and contributors.
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

# general & system functions
from __future__ import absolute_import
from __future__ import print_function
from abc import ABC, abstractmethod
from copy import deepcopy
import pickle

# basic numeric setup
import numpy as np

# dynesty
from dynesty import dynesty
from dynesty import utils as dyutils
import pandas as pd


class DynestySolver(ABC):

    def __init__(self, context):
        self.context = context

        # Set numpy error handling for numerical issues such as underflow/overflow/invalid
        np.seterr(under="ignore")
        np.seterr(over="ignore")
        np.seterr(invalid="ignore")
    
    @property
    def dynesty_results(self):
        if not hasattr(self, '_dynesty_results'):
            return None
        return self._dynesty_results  # large object should not be copied

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        pass

    @property
    def ndim(self):
        return len(self.labels)
    
    @property
    def truths(self):
        if not hasattr(self, '_dynesty_results'):
            return None        
        truths_, _, _ = self.quantile()
        return np.squeeze(truths_)
    
    def package_results(self, parc_index: int = 0) -> dict:
        """ provides a super dictionary also containing dynesty_results in entry "res" """
        resd = self._dynesty_results.asdict()
        logz = resd["logz"][-1]
        information = resd["information"][-1]
        qm, ql, qh = self.quantile()
        rho_pred, rho_ideal, timesIdeal = self.signalmodel(qm, parc_index)
        resid = rho_pred - self.data.rho
        return {
            "res": self._dynesty_results,
            "logz": logz,
            "information": information, 
            "qm": qm,
            "ql": ql,
            "qh": qh,
            "rho_pred": rho_pred,
            "rho_ideal": rho_ideal,
            "timesIdeal": timesIdeal,
            "resid": resid
        }

    def quantile(
            self,
            results: dyutils.Results | None = None,
            verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Requires results or that run_nested() has successfully completed. 
            Returns cached results, if possible, else builds new results and cache. 
            verbose == True prints results of building new results. """    
        if not results:
            if not hasattr(self, '_dynesty_results'):
                raise AssertionError("self._dynesty_results does not exist. Run run_nested() first.")
            results = self._dynesty_results

        # return cached results
        if hasattr(self, '__qm') and hasattr(self, '__ql') and hasattr(self, '__qh'):
            return self.__qm, self.__ql, self.__qh

        # build new results
        samples = results["samples"].T
        weights = results.importance_weights().T
        ql = np.zeros(len(samples))
        qm = np.zeros(len(samples))
        qh = np.zeros(len(samples))
        max_label_len = max(len(label) for label in self.labels)

        for i, x in enumerate(samples):
            ql[i], qm[i], qh[i] = dyutils.quantile(x, [0.025, 0.5, 0.975], weights=weights)
            if verbose:
                # print quantile results in tabular format
                qm_fmt = ".1f" if abs(qm[i]) >= 1000 else ".4f"
                ql_fmt = ".1f" if abs(ql[i]) >= 1000 else ".4f" 
                qh_fmt = ".1f" if abs(qh[i]) >= 1000 else ".4f"
                label_padded = f"{self.labels[i]:<{max_label_len}}"
                print(f"Parameter {label_padded}: {qm[i]:{qm_fmt}} [{ql[i]:{ql_fmt}}, {qh[i]:{qh_fmt}}]")

        # cache results
        self.__qm, self.__ql, self.__qh = qm, ql, qh
        return qm, ql, qh


    def results_save(self, tag: str = "", parc_index: int = 0) -> str:
        """ Saves .nii.gz and -quantiles.csv.  Returns f.q. fileprefix. """
        self.save_pickle(tag)

        fqfp1 = self.context.io.results_fqfp 
        if parc_index > 0:
            fqfp1 += f"-parc{parc_index}"
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 += tag

        # =========== save .nii.gz ===========

        ifm = self.context.data.input_func_measurement
        A0 = np.max(ifm["img"])
        fqfp = ifm["fqfp"]
        nii = ifm["nii"]
        timesMid = ifm["timesMid"]
        taus = ifm["taus"]
        json = ifm["json"]
        if not np.array_equal(json["timesMid"], timesMid):
            json["timesMid"] = timesMid.tolist()
        if not np.array_equal(json["taus"], taus):
            json["taus"] = taus.tolist()

        resd = self.package_results(parc_index=parc_index)
        rho_pred, rho_ideal, timesIdeal = resd["rho_pred"], resd["rho_ideal"], resd["timesIdeal"]

        self.context.io.nii_save({
            "timesMid": timesMid,
            "taus": taus,
            "img": A0 * rho_pred,
            "nii": nii,
            "fqfp": fqfp,
            "json": json
        }, fqfp1 + "-signal.nii.gz")

        self.context.io.nii_save({
            "times": timesIdeal,
            "taus": np.ones(timesIdeal.shape),
            "img": A0 * rho_ideal,
            "nii": nii,
            "fqfp": fqfp,
            "json": json
        }, fqfp1 + "-ideal.nii.gz")

        product = deepcopy(ifm)
        product["img"] = A0 * resd["rho_pred"]
        self.context.io.nii_save(product, fqfp1 + "-rho-pred.nii.gz")

        product = deepcopy(ifm)
        product["img"] = A0 * resd["rho_ideal"]
        self.context.io.nii_save(product, fqfp1 + "-rho-ideal.nii.gz")

        for key in ["logz", "information", "qm", "ql", "qh", "resid"]:
            product = deepcopy(ifm)
            product["img"] = resd[key]
            self.context.io.nii_save(product, fqfp1 + f"-{key}.nii.gz")

        # =========== save .csv ===========

        qm, ql, qh = self.quantile()
        df = {
            "label": self.labels,
            "qm": qm,
            "ql": ql,
            "qh": qh}
        df = pd.DataFrame(df)
        df.to_csv(fqfp1 + "-quantiles.csv")

        return fqfp1
    
    @abstractmethod
    def _run_nested(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: int = 0
    ) -> dyutils.Results:
        pass
    
    def run_nested(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: int = 0
    ) -> dyutils.Results:
        self._dynesty_results = self._run_nested(checkpoint_file, print_progress, resume, parc_index)
        return self._dynesty_results

    def save_pickle(self, tag: str = "") -> str:
        """ saves -res.pickle """
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 = self.context.io.results_fqfp + tag

        if not hasattr(self, '_dynesty_results'):
            raise AssertionError("self._dynesty_results does not exist. Run run_nested() first.")
        with open(fqfp1 + "-res.pickle", 'wb') as f:
            pickle.dump(self._dynesty_results, f, pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def signalmodel(self, v: np.ndarray, parc_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def loglike(self, v: np.ndarray, parc_index: int = 0) -> float:
        pass
