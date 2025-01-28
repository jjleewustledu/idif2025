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
from PETUtilities import PETUtilities


class InputFuncSolver(DynestySolver):
    def __init__(self, context):
        super().__init__(context)

    def package_results(
            self, 
            results: dyutils.Results |list[dyutils.Results] | None = None
    ) -> dict:
        """ provides a super dictionary also containing dynesty_results in entry "res" """
        
        if not results:
            if not self._get_cached_dynesty_results():
                raise AssertionError("Cache of dynesty_results is empty. Run run_nested().")
            results = self._get_cached_dynesty_results()
        
        resd = results.asdict()
        logz = resd["logz"][-1]
        information = resd["information"][-1]
        qm, ql, qh = self.quantile(results=results)
        rho_pred, rho_ideal, timesIdeal= self.signalmodel(v=qm)
        resid = rho_pred - self.data.rho
        return {
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

    def results_save(
            self, 
            tag: str = "",
            results: dyutils.Results | None = None
    ) -> str:
        """ Saves .nii.gz and -quantiles.csv.  Returns f.q. fileprefix. """
        
        # add tag to results f.q. fileprefix
        
        fqfp1 = self.context.io.results_fqfp 
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 += tag

        # =========== pickle dynesty results ===========

        self.pickle_dump(tag)

        # =========== save .nii.gz ===========

        ifm = self.context.data.input_func_measurement
        A0 = np.max(ifm["img"])
        pkg = self.package_results(results=results)

        self.context.io.nii_save({
            "timesMid": ifm["timesMid"],
            "taus": ifm["taus"],
            "img": A0 * pkg["rho_pred"],
            "nii": ifm["nii"],
            "fqfp": fqfp1,
            "json": ifm["json"]
        }, fqfp1 + "-signal.nii.gz")

        json = deepcopy(ifm["json"])
        json["taus"] = np.ones(pkg["timesIdeal"].shape).tolist()
        json["times"] = pkg["timesIdeal"].tolist()
        json["timesMid"] = self.context.io.data2timesMid(json).tolist()
        self.context.io.nii_save({
            "taus": np.ones(pkg["timesIdeal"].shape),
            "times": pkg["timesIdeal"],
            "timesMid": self.context.io.data2timesMid(json),
            "img": A0 * pkg["rho_ideal"],
            "nii": ifm["nii"],
            "fqfp": fqfp1,
            "json": json
        }, fqfp1 + "-ideal.nii.gz")

        product = deepcopy(ifm)
        product["img"] = A0 * pkg["rho_pred"]
        self.context.io.nii_save(product, fqfp1 + "-rho-pred.nii.gz")

        product = deepcopy(ifm)
        product["img"] = A0 * pkg["rho_ideal"]
        self.context.io.nii_save(product, fqfp1 + "-rho-ideal.nii.gz")

        for key in ["logz", "information", "qm", "ql", "qh", "resid"]:
            product = deepcopy(ifm)
            product["img"] = pkg[key]
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
    
    def run_nested(
            self,
            checkpoint_file: str | list[str] | None = None,
            print_progress: bool = False,
            resume: bool | list[bool] = False
    ) -> dyutils.Results | list[dyutils.Results]:
        
        self._clear_cache()

        return self._run_nested_single(checkpoint_file, print_progress, resume)
