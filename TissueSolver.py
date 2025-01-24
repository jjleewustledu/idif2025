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


from abc import abstractmethod
from copy import deepcopy
from dynesty import utils as dyutils
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from DynestySolver import DynestySolver


class TissueSolver(DynestySolver):
    def __init__(self, context):
        super().__init__(context)

    def package_results(
            self, 
            results: dyutils.Results |list[dyutils.Results] | None = None,
            parc_index: int |list[int] | tuple[int, ...] | NDArray = 0
    ) -> dict:
        """ provides a super dictionary also containing dynesty_results in entry "res" """
        
        if not results:
            if not self._get_cached_dynesty_results():
                raise AssertionError("Cache of dynesty_results is empty. Run run_nested().")
            results = self._get_cached_dynesty_results()

        if (isinstance(parc_index, (list, tuple, np.ndarray)) or
            (isinstance(results, list) and all(isinstance(r, dyutils.Results) for r in results))):
            return self._package_results_pool(results=results, parc_index=parc_index)
        
        resd = results.asdict()
        logz = resd["logz"][-1]
        information = resd["information"][-1]
        qm, ql, qh = self.quantile(results=results)
        rho_pred, timesMid, rho_ideal, timesIdeal= self.signalmodel(v=qm, parc_index=parc_index)
        resid = rho_pred - self.data.rho
        return {
            "logz": logz,
            "information": information, 
            "qm": qm,
            "ql": ql,
            "qh": qh,
            "rho_pred": rho_pred,
            "timesMid": timesMid,
            "rho_ideal": rho_ideal,
            "timesIdeal": timesIdeal,
            "resid": resid
        }
    
    def _package_results_pool(
            self,
            results: list[dyutils.Results] | None = None,
            parc_index: list[int] | tuple[int, ...] | NDArray = 0
    ) -> dict:
        
        qms, qls, qhs = self.quantile(results=results)
        expected_shape = (len(parc_index), self.ndim)
        assert qms.shape == expected_shape, f"qms shape {qms.shape} != expected {expected_shape}"
        assert qls.shape == expected_shape, f"qls shape {qls.shape} != expected {expected_shape}"
        assert qhs.shape == expected_shape, f"qhs shape {qhs.shape} != expected {expected_shape}"

        for res in results:
            
            logzs = []
            informations = []
            rho_preds = []
            timesMids = []
            rho_ideals = []
            timesIdeals = []
            resids = []
            
            for i, res in enumerate(results):
                resd = res.asdict()
                logzs.append(resd["logz"][-1])
                informations.append(resd["information"][-1])
                rho_pred, timesMid, rho_ideal, timesIdeal = self.signalmodel(qms[i], parc_index[i])
                rho_preds.append(rho_pred)
                timesMids.append(timesMid)
                rho_ideals.append(rho_ideal) 
                timesIdeals.append(timesIdeal)
                resids.append(rho_pred - self.data.rho[i,])
            
            logzs = np.array(logzs)
            informations = np.array(informations)
            qms = np.vstack(qms)
            qls = np.vstack(qls)
            qhs = np.vstack(qhs)
            rho_preds = np.vstack(rho_preds)
            timesMids = np.vstack(timesMids)
            rho_ideals = np.vstack(rho_ideals)
            timesIdeals = np.vstack(timesIdeals)
            resids = np.vstack(resids)
        return {
            "logz": logzs,
            "information": informations, 
            "qm": qms,
            "ql": qls,
            "qh": qhs,
            "rho_pred": rho_preds,
            "timesMid": timesMids,
            "rho_ideal": rho_ideals,
            "timesIdeal": timesIdeals,
            "resid": resids
        }
    
    def _quantile_pool(
            self,
            results: list[dyutils.Results] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Get dimensions
        M = len(results)  # Number of results objects
        N = len(results[0]["samples"].T)  # Number of parameters

        # Initialize arrays
        ql = np.zeros((M, N))
        qm = np.zeros((M, N)) 
        qh = np.zeros((M, N))
        max_label_len = max(len(label) for label in self.labels)

        # Process each results object
        for i, res in enumerate(results):
            samples = res["samples"].T
            weights = res.importance_weights().T
            
            # Calculate quantiles for each parameter
            for j, x in enumerate(samples):
                ql[i,j], qm[i,j], qh[i,j] = dyutils.quantile(x, [0.025, 0.5, 0.975], weights=weights)

        # cache results
        self._set_cached_quantile(qm, ql, qh)
        return qm, ql, qh

    def results_save(
            self, 
            tag: str = "", 
            results: dyutils.Results | list[dyutils.Results] | None = None,
            parc_index: int | list[int] | tuple[int, ...] | NDArray = 0
    ) -> str:
        """ Saves .nii.gz and -quantiles.csv.  Returns f.q. fileprefix. """
        
        # add parc index and tag to results f.q. fileprefix

        fqfp1 = self.context.io.results_fqfp 
        if isinstance(parc_index, (int, np.integer)) and parc_index > 0:
            fqfp1 += f"-parc{parc_index}"
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 += tag

        # =========== pickle dynesty results ===========   

        self.pickle_dump(tag)

        # =========== save .nii.gz ===========

        tm = self.context.data.tissue_measurement
        
        A0 = self.context.data.max_tissue_measurement

        pkg = self.package_results(results=results, parc_index=parc_index)

        json = tm["json"]
        if not np.array_equal(json["timesMid"], tm["timesMid"]):
            json["timesMid"] = tm["timesMid"].tolist()
        if not np.array_equal(json["taus"], tm["taus"]):
            json["taus"] = tm["taus"].tolist()

        self.context.io.nii_save({
            "timesMid": pkg["timesMid"],
            "taus": tm["taus"],
            "img": A0 * pkg["rho_pred"],
            "nii": tm["nii"],
            "fqfp": tm["fqfp"],
            "json": json
        }, fqfp1 + "-signal.nii.gz")

        self.context.io.nii_save({
            "times": pkg["timesIdeal"],
            "taus": np.ones(pkg["timesIdeal"].shape),
            "img": A0 * pkg["rho_ideal"],
            "nii": tm["nii"],
            "fqfp": tm["fqfp"],
            "json": json
        }, fqfp1 + "-ideal.nii.gz")

        product = deepcopy(tm)
        product["img"] = A0 * pkg["rho_pred"]
        self.context.io.nii_save(product, fqfp1 + "-rho-pred.nii.gz")

        product = deepcopy(tm)
        product["img"] = A0 * pkg["rho_ideal"]
        self.context.io.nii_save(product, fqfp1 + "-rho-ideal.nii.gz")

        for key in ["logz", "information", "qm", "ql", "qh", "resid"]:
            product = deepcopy(tm)
            product["img"] = pkg[key]
            self.context.io.nii_save(product, fqfp1 + f"-{key}.nii.gz")

        # =========== save .csv ===========

        qm, ql, qh = self.quantile(results=results)
        df = {
            "label": self.labels,
            "qm": qm,
            "ql": ql,
            "qh": qh}
        df = pd.DataFrame(df)
        df.to_csv(fqfp1 + "-quantiles.csv")

        return fqfp1  
    
    def run_nested(
            self,
            checkpoint_file: str | list[str] | None = None,
            print_progress: bool = False,
            resume: bool | list[bool] = False,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> dyutils.Results | list[dyutils.Results]:
        
        self._clear_cache()

        if (isinstance(parc_index, (int, np.integer)) or 
            (isinstance(parc_index, np.ndarray) and parc_index.size == 1)):
            return self._run_nested_single(checkpoint_file, print_progress, resume, parc_index)
        else:
            return self._run_nested_pool(checkpoint_file, print_progress, resume, parc_index)

    @abstractmethod
    def _run_nested_pool(
            self,
            checkpoint_file: list[str] | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: list[int] | tuple[int, ...] | NDArray | None = None
    ) -> list[dyutils.Results]:
        pass
