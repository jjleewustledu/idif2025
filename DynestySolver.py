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
from abc import ABC
import sys
from datetime import datetime

# basic numeric setup
import numpy as np

# dynesty
from dynesty import dynesty
from dynesty import utils as dyutils


class DynestySolver(ABC):

    def __init__(self,
                 model=None,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        self.model = model
        self.sample = sample
        self.nlive = nlive
        self.rstate = rstate

        # Set numpy error handling for numerical issues such as underflow/overflow/invalid
        np.seterr(under="ignore")
        np.seterr(over="ignore")
        np.seterr(invalid="ignore")

    def run_nested(self,
                   prior_tag=None,
                   ndim=None,
                   checkpoint_file=None,
                   print_progress=False,
                   resume=False):
        """ checkpoint_file=self.fqfp+"_dynesty-ModelClass-yyyyMMddHHmmss.save") """

        mdl = self.model

        if ndim is None:
            ndim = mdl.ndim

        if resume:
            sampler = dynesty.DynamicNestedSampler.restore(checkpoint_file)
        else:
            __prior_transform = mdl.prior_transform()
            sampler = dynesty.DynamicNestedSampler(mdl.loglike, __prior_transform, ndim,
                                                   sample=self.sample, nlive=self.nlive,
                                                   rstate=self.rstate)
        sampler.run_nested(checkpoint_file=checkpoint_file, print_progress=print_progress, resume=resume)
        # for posterior > evidence, use wt_kwargs={"pfrac": 1.0}
        res = sampler.results
        class_name = self.__class__.__name__
        fqfn = mdl.fqfp+"_dynesty-"+class_name+"-"+prior_tag+"-summary.txt"
        with open(fqfn, "w") as f:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f
            res.summary()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return res

    def run_nested_for_list(self,
                            prior_tag=None,
                            ndim=None,
                            checkpoint_file=None,
                            print_progress=False,
                            resume=False):
        """ default: checkpoint_file=self.fqfp+"_dynesty-ModelClass-yyyyMMddHHmmss.save") """

        mdl = self.model

        if ndim is None:
            ndim = mdl.ndim

        if resume:
            sampler = dynesty.DynamicNestedSampler.restore(checkpoint_file)
        else:
            __prior_transform = mdl.prior_transform()
            sampler = dynesty.DynamicNestedSampler(mdl.loglike, __prior_transform, ndim,
                                                   sample=self.sample, nlive=self.nlive,
                                                   rstate=self.rstate)
        sampler.run_nested(checkpoint_file=checkpoint_file, print_progress=print_progress, resume=resume)
        # for posterior > evidence, use wt_kwargs={"pfrac": 1.0}
        res = sampler.results
        return res

    @staticmethod
    def quantile(res: dyutils.Results):
        samples = res["samples"].T
        weights = res.importance_weights().T
        ql = np.zeros(len(samples))
        qm = np.zeros(len(samples))
        qh = np.zeros(len(samples))
        for i, x in enumerate(samples):
            ql[i], qm[i], qh[i] = dyutils.quantile(x, [0.025, 0.5, 0.975], weights=weights)
            print(f"Parameter {i}: {qm[i]:.6f} [{ql[i]:.6f}, {qh[i]:.6f}]")
        return qm, ql, qh
