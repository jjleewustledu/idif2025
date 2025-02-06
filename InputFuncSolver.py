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


from copy import deepcopy
from dynesty import utils as dyutils
import numpy as np
import pandas as pd
from DynestySolver import DynestySolver


class InputFuncSolver(DynestySolver):
    """Base class for solvers that fit input functions to PET data.

    This abstract class provides common functionality for fitting input functions to PET data
    using dynamic nested sampling. Subclasses implement specific input function models.

    Args:
        context: The context object containing data and configuration for the solver.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object containing PET measurements and metadata.

    Example:
        >>> context = InputFuncContext(data_dict)
        >>> solver = ConcreteInputFuncSolver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    Notes:
        Subclasses must implement:
        - labels property: Parameter labels for plotting and output
        - _loglike method: Log-likelihood function for nested sampling
        - signalmodel method: Forward model for generating predicted signals
    """
    def __init__(self, context):
        super().__init__(context)

    def package_results(
            self, 
            results: dyutils.Results | list[dyutils.Results] | None = None
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
        rho_pred, rho_ideal, timesIdeal = self.signalmodel(v=qm)
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
            results: dyutils.Results | None = None,
            do_pickle: bool = True
    ) -> str:
        """ Saves .nii.gz and -quantiles.csv.  Returns f.q. fileprefix. 
            do_pickle (bool) is useful for efficient unit-testing. """
        
        # add tag to results f.q. fileprefix
        
        fqfp1 = self.context.io.results_fqfp 
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 += tag

        # =========== pickle dynesty results ===========

        if do_pickle:
            self.pickle_dump(tag)

        # =========== save .nii.gz ===========

        ifm = self.context.data.input_func_measurement
        A0 = np.max(ifm["img"])
        pkg = self.package_results(results=results)

        data_signal = {
            "timesMid": ifm["timesMid"],
            "taus": ifm["taus"],
            "times": self.context.io.data2times(ifm),
            "img": A0 * pkg["rho_pred"],
            "nii": ifm["nii"],
            "fqfp": fqfp1,
            "json": ifm["json"]
        }
        self.context.io.nii_save(
            data_signal, 
            fqfp1 + "-signal.nii.gz", 
            check_validity=True
        )

        product = deepcopy(data_signal)
        product["img"] = pkg["rho_pred"]
        self.context.io.nii_save(
            product, 
            fqfp1 + "-rho-pred.nii.gz",
            check_validity=True
        )

        json_ideal = deepcopy(ifm["json"])
        json_ideal["timesMid"] = self.context.io.data2timesMidInterp(json_ideal).tolist()
        json_ideal["taus"] = np.ones(pkg["timesIdeal"].shape).tolist()
        json_ideal["times"] = pkg["timesIdeal"].tolist()
        data_ideal = {
            "timesMid": self.context.io.data2timesMidInterp(json_ideal),
            "taus": np.ones(pkg["timesIdeal"].shape),
            "times": pkg["timesIdeal"],
            "img": A0 * pkg["rho_ideal"],
            "nii": ifm["nii"],
            "fqfp": fqfp1,
            "json": json_ideal
        }
        self.context.io.nii_save(
            data_ideal, 
            fqfp1 + "-ideal.nii.gz",
            check_validity=True
        )

        product = deepcopy(data_ideal)
        product["img"] = pkg["rho_ideal"]
        self.context.io.nii_save(
            product, 
            fqfp1 + "-rho-ideal.nii.gz",
            check_validity=True
        )

        for key in ["logz", "information", "qm", "ql", "qh", "resid"]:
            product = deepcopy(data_signal)
            product["img"] = pkg[key]
            self.context.io.nii_save(
                product, 
                fqfp1 + f"-{key}.nii.gz", 
                check_validity=False
            )

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
