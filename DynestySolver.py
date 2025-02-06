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

# basic numeric setup
import numpy as np

# dynesty
from dynesty import utils as dyutils
from numpy.typing import NDArray


class DynestySolver(ABC):
    """Abstract base class for implementing nested sampling solvers using dynesty.

    This class provides a framework for implementing nested sampling parameter estimation
    using the dynesty package. It handles caching of results and provides abstract methods
    that must be implemented by concrete subclasses.

    Args:
        context: Context object containing data and configuration for the solver.

    Attributes:
        context: Reference to the context object.
        data: Reference to the context's data object.
        _cache (dict): Cache for storing computed results.
            Keys:
                - "quantile": Tuple of (qm, ql, qh) arrays for parameter quantiles
                - "dynesty_results": Results object from dynesty run

    Properties:
        dynesty_results: Results from most recent dynesty run
        labels: List of parameter labels (must be implemented by subclass)
        ndim: Number of parameters being fit

    Example:
        >>> class MySolver(DynestySolver):
        ...     @property
        ...     def labels(self):
        ...         return ["param1", "param2"]
        >>> solver = MySolver(context)
        >>> results = solver.run_nested()
        >>> qm, ql, qh = solver.quantile(results)

    Notes:
        Subclasses must implement:
        - labels property: List of parameter labels
        - Any additional methods needed for the specific model
    """

    def __init__(self, context):
        self.context = context
        self.data = self.context.data
        self._cache = {
            "quantile": None,
            "dynesty_results": None
        }
        # Set numpy error handling for numerical issues such as underflow/overflow/invalid
        np.seterr(under="ignore", over="ignore", invalid="ignore")

    # cache managers

    def _get_cached_quantile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_tuple = self._cache["quantile"]
        if not a_tuple:
            return None
        qm, ql, qh = a_tuple
        return qm, ql, qh
    
    def _set_cached_quantile(self, qm: np.ndarray, ql: np.ndarray, qh: np.ndarray) -> None:
        self._cache["quantile"] = (qm, ql, qh)

    def _get_cached_dynesty_results(self) -> dyutils.Results | list[dyutils.Results] | None:
        return self._cache["dynesty_results"]
    
    def _set_cached_dynesty_results(self, results: dyutils.Results | list[dyutils.Results]) -> None:
        self._cache["dynesty_results"] = results

    def _clear_cache(self) -> None:
        self._cache = {k: None for k in self._cache}

    # properties
    
    @property
    def dynesty_results(self) -> dyutils.Results | list[dyutils.Results] | None:
        if self._get_cached_dynesty_results():
            return self._get_cached_dynesty_results()
        return None

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        pass

    @property
    def ndim(self):
        return len(self.labels)
    
    @property
    def truths(self) -> np.ndarray | None:
        if not self._get_cached_dynesty_results():
            return None        
        qm, _, _ = self.quantile(self._get_cached_dynesty_results())
        return qm
    
    # methods
        
    @abstractmethod
    def loglike(
            self,
            v: np.ndarray,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> float:
        pass

    @abstractmethod
    def package_results(
            self, 
            results: dyutils.Results | list[dyutils.Results] | None = None,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> dict:
        pass

    def quantile(
            self,
            results: dyutils.Results | None = None,
            verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Requires externalresults or completion of run_nested().  
            Returns cached results, if possible, else builds new results with caches. 
            verbose == True prints table of new results. """    

        # return cached results
        # if self._get_cached_quantile():
        #     return self._get_cached_quantile()
        
        # use self._get_cached_dynesty_results() if no results provided
        if not results:
            if not self._get_cached_dynesty_results():
                raise AssertionError("Cache of dynesty_results is empty. Call run_nested().")
            results = self._get_cached_dynesty_results()

        # Call self._quantile_pool() if results is a list of dyutils.Results
        if (isinstance(results, list) and 
                all(isinstance(r, dyutils.Results) for r in results)):
            return self._quantile_pool(results)

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
        # self._set_cached_quantile(qm, ql, qh)
        return qm, ql, qh

    @abstractmethod
    def results_save(
            self, 
            tag: str = "", 
            results: dyutils.Results | list[dyutils.Results] | None = None,
            parc_index: int | list[int] | tuple[int, ...] | NDArray = None
    ) -> str:
        pass
    
    @abstractmethod
    def run_nested(
            self,
            checkpoint_file: str | list[str] | None = None,
            print_progress: bool = False,
            resume: bool | list[bool] = False,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> dyutils.Results | list[dyutils.Results]:
        pass
    
    @abstractmethod
    def _run_nested_single(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> dyutils.Results:
        pass

    def pickle_dump(self, tag: str = "") -> str:
        """ pickles cached dynesty results """

        if not self._get_cached_dynesty_results():
            raise AssertionError("Cache of dynesty_results is empty. Call run_nested().")
        
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 = self.context.io.results_fqfp + tag  
        if fqfp1.endswith("-"):
            fqfp1 = fqfp1[:-1]      
        fqfn = self.context.io.pickle_dump(self._get_cached_dynesty_results(), fqfp1)
        return fqfn
    
    def pickle_load(self, fqfn: str) -> dyutils.Results:
        return self.context.io.pickle_load(fqfn)

    @abstractmethod
    def signalmodel(
            self,
            v: np.ndarray,
            parc_index: int | list[int] | tuple[int, ...] | NDArray | None = None
    ) -> tuple[np.ndarray, ...]:
        pass
