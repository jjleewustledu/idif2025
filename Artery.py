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
from abc import ABC, abstractmethod
import time, sys, os
from datetime import datetime

# basic numeric setup
import numpy as np
import pandas as pd

import dynesty
from dynesty import dynesty
from dynesty import utils as dyutils
from dynesty import plotting as dyplot

# NIfTI support
import nibabel as nib
import json

# plotting
from matplotlib import pyplot as plt
# re-defining plotting defaults
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 30})

# seed the random number generator
#rstate = np.random.default_rng(916301)

__all__ = ["Artery"]


def _trim_input_func_measurement(ifm):
    assert isinstance(ifm, dict), f"{ifm} is a {type(ifm)}, but should be a dict."
    img = ifm['img']
    timesMid = ifm['timesMid']
    taus = ifm['taus']
    viable = ~np.isnan(timesMid)
    ifm.update({'img': img[viable], 'timesMid': timesMid[viable], 'taus': taus[viable]})
    return ifm


class Artery(ABC):

    def __init__(self,
                 input_func_measurement,
                 remove_baseline=False,
                 home=os.getcwd(),
                 sample='rslice',
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        self.__input_func_measurement = input_func_measurement
        self.__remove_baseline = remove_baseline
        self.home = home
        self.sample = sample
        self.nlive = nlive
        self.rstate = rstate

        # Set numpy error handling for numerical issues such as underflow/overflow/invalid
        np.seterr(under='ignore')
        np.seterr(over='ignore')
        np.seterr(invalid='ignore')

    @property
    def fqfp(self):
        return self.input_func_measurement['fqfp']

    @property
    def input_func_measurement(self):
        if isinstance(self.__input_func_measurement, dict):
            return self.__input_func_measurement

        assert os.path.isfile(self.__input_func_measurement), f"{self.__input_func_measurement} was not found."
        fqfn = self.__input_func_measurement

        # load img
        nii = nib.load(fqfn)
        img = nii.get_fdata()
        if self.__remove_baseline:
            img = img - img[1]
            img = img.clip(min=0)

        # find json fields of interest
        base, ext = os.path.splitext(fqfn)
        fqfp = os.path.splitext(base)[0]
        jfile = fqfp + ".json"
        with open(jfile, 'r') as f:
            j = json.load(f)

        # assemble dict
        ifm = {
            'fqfp': fqfp,
            'img': np.array(img, dtype=float).reshape(1, -1),
            'timesMid': np.array(j['timesMid'], dtype=float).reshape(1, -1),
            'taus': np.array(j['taus'], dtype=float).reshape(1, -1)}
        self.__input_func_measurement = _trim_input_func_measurement(ifm)
        return self.__input_func_measurement

    @property
    def labels(self):
        """"""

        return [
            r'$t_0$', r'$\tau_2$', r'$\tau_3$',
            r'$\alpha - 1$', r'$1/\beta$', r'$p$', r'$\delta p_2$', r'$\delta p_3$', r'$1/\gamma$',
            r'$f_2$', r'$f_3$', r'$f_{ss}$',
            r'$A$', r'$\sigma$']

    @property
    def truths(self):
        """/Volumes/PrecunealSSD/Singularity/CCIR_01211/derivatives/sub-108293/ses-20210421150523/pet for mipidif"""
        """/Volumes/PrecunealSSD/Singularity/CCIR_01211/sourcedata/sub-108293/ses-20210421/pet for twilite nomodel"""

        return [9.02, 1.67, 5.95,
                7.08, 3.00, 2.79, -1.00, -0.19, 1.54,
                0.48, 0.19, 0.082,
                2.47, 0.001]

    def plot_results(self, res: dyutils.Results):
        qm, _, _ = self.quantile(res)
        self.plot_truths(qm)

        dyplot.runplot(res)
        plt.tight_layout()

        fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm,
                                     fig=plt.subplots(14, 2, figsize=(16, 50)))
        fig.tight_layout()

        fig, axes = dyplot.cornerplot(res, truths=qm, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=self.labels,
                                      fig=plt.subplots(14, 14, figsize=(100, 100)))

    def plot_truths(self, truths=None):
        if truths is None:
            truths = self.truths
        data = self.data(truths)
        rho_pred, rho_ideal, t_ideal = self.signalmodel(data)

        ifm = self.input_func_measurement
        tM = ifm['timesMid']
        rho = ifm['img']
        M0 = np.max(rho)

        plt.figure(figsize=(12, 5))
        plt.plot(tM, rho, color='black', marker='+',
                 ls='none', alpha=0.9, markersize=12)
        plt.plot(tM, M0 * rho_pred, marker='o', color='red', ls='none', alpha=0.8)
        plt.plot(t_ideal, M0 * rho_ideal, color='dodgerblue', linewidth=2, alpha=0.7)
        plt.xlim([-0.1, 1.1*np.max(tM)])
        plt.xlabel('time of mid-frame (s)')
        plt.ylabel('activity (kBq/mL)')
        plt.tight_layout()

    def run_nested(self, checkpoint_file=None):
        """ checkpoint_file=self.fqfp+"_dynesty-RadialArtery.save") """

        if checkpoint_file is None:
            class_name = self.__class__.__name__
            now = datetime.now()
            checkpoint_file = self.fqfp+"_dynesty-"+class_name+"-"+now.strftime("%Y%m%d%H%M%S")+".save"

        sampler = dynesty.DynamicNestedSampler(self.loglike, self.prior_transform, 14,
                                               sample=self.sample, nlive=self.nlive,
                                               rstate=self.rstate)
        sampler.run_nested(checkpoint_file=checkpoint_file)
        # for posterior > evidence, use wt_kwargs={'pfrac': 1.0}
        return sampler.results

    def save_results(self, res: dyutils.Results):
        """"""

        ifm = self.input_func_measurement
        timesMid = ifm['timesMid']
        M0 = np.max(ifm['img'])
        qm, ql, qh = self.quantile(res)
        data = self.data(qm)
        rho_signal, rho_ideal, times = self.signalmodel(data)

        class_name = self.__class__.__name__
        d_signal = {
            "timesMid": timesMid,
            "signal": M0 * rho_signal}
        df = pd.DataFrame(d_signal)
        df.to_csv(self.fqfp+"_dynesty-"+class_name+"-signal.csv")
        d_ideal = {
            "times": times,
            "ideal": M0 * rho_ideal}
        df = pd.DataFrame(d_ideal)
        df.to_csv(self.fqfp+"_dynesty-"+class_name+"-ideal.csv")
        d_quantiles = {
            "label": self.labels,
            "qm": qm,
            "ql": ql,
            "qh": qh}
        df = pd.DataFrame(d_quantiles)
        df.to_csv(self.fqfp+"_dynesty-"+class_name+"-quantiles.csv")

    @staticmethod
    @abstractmethod
    def data(v):
        pass

    @staticmethod
    def data2t(data: dict):
        timesMid = data['timesMid']
        taus = data['taus']
        t0 = timesMid[0] - taus[0]/2
        tF = timesMid[-1] + taus[-1]/2
        t = np.arange(t0, tF)
        return t

    @staticmethod
    @abstractmethod
    def loglike(v):
        pass

    @staticmethod
    def prior_transform(u):
        v = u
        v[0] = u[0] * 20  # t_0
        v[1] = u[1] * 20  # \tau_2 ~ t_2 - t_0
        v[2] = u[2] * 20  # \tau_3 ~ t_3 - t_2
        v[3] = u[3] * 20  # \alpha - 1
        v[4] = u[4] * 30 + 1  # 1/\beta
        v[5] = u[5] * 4 + 0.25  # p
        v[6] = u[6] * 3 - 3  # \delta p_2 ~ p_2 - p
        v[7] = u[7] * 3 - 3  # \delta p_3 ~ p_3 - p_2
        v[8] = u[8] * 100 + 0.01  # 1/\gamma for s.s.
        v[9] = u[9] * 0.75 + 0.25  # f_2
        v[10] = u[10] * 0.5  # f_3
        v[11] = u[11] * 0.125  # f_{ss}
        v[12] = u[12] * 4 + 0.5  # A is amplitude adjustment
        v[13] = u[13] * 0.001  # sigma ~ fraction of M0
        return v

    @staticmethod
    def quantile(res: dyutils.Results):
        samples = res['samples'].T
        weights = res.importance_weights().T
        ql = np.zeros(len(samples))
        qm = np.zeros(len(samples))
        qh = np.zeros(len(samples))
        for i, x in enumerate(samples):
            ql[i], qm[i], qh[i] = dyutils.quantile(x, [0.025, 0.5, 0.975], weights=weights)
            print(f"Parameter {i}: {qm[i]:.3f} [{ql[i]:.3f}, {qh[i]:.3f}]")
        return qm, ql, qh

    @staticmethod
    @abstractmethod
    def signalmodel(data: dict):
        pass

    @staticmethod
    def slide(rho, t, dt):
        if dt < 0.1:
            return rho
        return np.interp(t - dt, t, rho)

    @staticmethod
    def solution_1bolus(t, t_0, a, b, p):
        """ generalized gamma distribution """

        t_ = np.array(t - t_0, dtype=complex)
        t_ = t_.clip(min=0)
        rho = np.power(t_, a) * np.exp(-np.power((b * t_), p))
        rho = np.real(rho)
        rho = rho.clip(min=0)
        rho = rho / np.max(rho)
        return rho

    @staticmethod
    def solution_2bolus(t, t_0, a, b, p, g, f_ss):
        """ generalized gamma distributions + rising exponential """
        f_1 = 1 - f_ss
        rho = (f_1 * Artery.solution_1bolus(t, t_0, a, b, p) +
               f_ss * Artery.solution_ss(t, t_0, g))
        return rho

    @staticmethod
    def solution_3bolus(t, t_0, tau_2, a, b, p, dp_2, g, f_2, f_ss):
        """ two sequential generalized gamma distributions + rising exponential """

        f_ss_ = f_ss * (1 - f_2)
        f_1_ = (1 - f_ss) * (1 - f_2)
        f_2_ = f_2
        rho = (f_1_ * Artery.solution_1bolus(t, t_0, a, b, p) +
               f_2_ * Artery.solution_1bolus(t, t_0 + tau_2, a, b, max(0.25, p + dp_2)) +
               f_ss_ * Artery.solution_ss(t, t_0, g))
        return rho

    @staticmethod
    def solution_4bolus(t, t_0, tau_2, tau_3, a, b, p, dp_2, dp_3, g, f_2, f_3, f_ss):
        """ three sequential generalized gamma distributions + rising exponential """

        f_ss_ = f_ss * (1 - f_2) * (1 - f_3)
        f_1_ = (1 - f_ss) * (1 - f_2) * (1 - f_3)
        f_2_ = f_2 * (1 - f_3)
        f_3_ = f_3
        rho = (f_1_ * Artery.solution_1bolus(t, t_0, a, b, p) +
               f_2_ * Artery.solution_1bolus(t, t_0 + tau_2, a, b, max(0.25, p + dp_2)) +
               f_3_ * Artery.solution_1bolus(t, t_0 + tau_2 + tau_3, a, b, max(0.25, p + dp_2 + dp_3)) +
               f_ss_ * Artery.solution_ss(t, t_0, g))
        return rho

    @staticmethod
    def solution_ss(t, t0, g):
        """ rising exponential coincident with first-appearing bolus """

        t_ = np.array(t - t0)
        rho = 1 - np.exp(-g * t_)
        rho = rho.clip(min=0)
        rho = rho / np.max(rho)
        return rho




