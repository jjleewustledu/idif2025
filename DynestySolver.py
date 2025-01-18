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

    @abstractmethod
    def package_results(self) -> dict:
        pass

    def quantile(
            self,
            verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Requires that run_nested() has successfully completed. 
            Returns cached results, if possible, else builds new results and cache. 
            verbose == True prints results of building new results. """    
        if not hasattr(self, '_dynesty_results'):
            raise AssertionError("self._dynesty_results does not exist. Run run_nested() first.")

        # return cached results
        if hasattr(self, '__qm') and hasattr(self, '__ql') and hasattr(self, '__qh'):
            return self.__qm, self.__ql, self.__qh

        # build new results
        samples = self._dynesty_results["samples"].T
        weights = self._dynesty_results.importance_weights().T
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
    
    @abstractmethod
    def _run_nested(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False
    ) -> dyutils.Results:
        pass
    
    def run_nested(
            self,
            checkpoint_file: str | None = None,
            print_progress: bool = False,
            resume: bool = False
    ) -> dyutils.Results:
        self._dynesty_results = self._run_nested(checkpoint_file, print_progress, resume)
        return self._dynesty_results

    def save_results(self, tag: str = "") -> str:
        """ saves -res.pickle """
        if tag:
            tag = f"-{tag.lstrip('-')}"
        fqfp1 = self.context.io.results_fqfp + tag

        if not hasattr(self, '_dynesty_results'):
            raise AssertionError("self._dynesty_results does not exist. Run run_nested() first.")
        with open(fqfp1 + "-res.pickle", 'wb') as f:
            pickle.dump(self._dynesty_results, f, pickle.HIGHEST_PROTOCOL)

