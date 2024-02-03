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

from DynestyModel import DynestyModel
from DynestySolver import DynestySolver

# general & system functions
import re
import time, sys, os
from datetime import datetime

# basic numeric setup
import numpy as np
import pandas as pd

# NIfTI support
import json
import nibabel as nib

# plotting
from matplotlib import pyplot as plt
from matplotlib import cm

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

# dynesty
from dynesty import utils as dyutils
from dynesty import plotting as dyplot

SIGMA = 0.001

__all__ = ["Artery"]


def _trim_input_func_measurement(ifm):
    assert isinstance(ifm, dict), f"{ifm} is a {type(ifm)}, but should be a dict."
    img = ifm['img']
    timesMid = ifm['timesMid']
    taus = ifm['taus']
    viable = ~np.isnan(timesMid)
    early = timesMid <= 180
    selected = viable * early
    ifm.update({'img': img[selected], 'timesMid': timesMid[selected], 'taus': taus[selected]})
    return ifm


class Artery(DynestyModel):

    def __init__(self,
                 input_func_measurement,
                 remove_baseline=False,
                 tracer=None,
                 home=os.getcwd(),
                 sample='rslice',
                 nlive=1000,
                 rstate=np.random.default_rng(916301)):
        self.__input_func_measurement = input_func_measurement
        self.__remove_baseline = remove_baseline
        self.tracer = tracer
        self.home = home

        if not self.tracer:
            regex = r"_trc-(.*?)_"
            matches = re.findall(regex, input_func_measurement)
            assert matches, "no tracer information in input_funct_measurement"
            self.tracer = matches[0]
            print(f"{self.__class__.__name__}: found data for tracer {self.tracer}")

        self.solver = DynestySolver(model=self,
                                    sample=sample,
                                    nlive=nlive,
                                    rstate=rstate)

    @property
    def fqfp(self):
        return self.input_func_measurement['fqfp']

    @property
    def input_func_measurement(self):
        if isinstance(self.__input_func_measurement, dict):
            return self.__input_func_measurement

        assert os.path.isfile(self.__input_func_measurement), f"{self.__input_func_measurement} was not found."
        fqfn = self.__input_func_measurement

        # manage SIGMA
        global SIGMA
        if "TwiliteKit" in fqfn:
            SIGMA = 0.1
        if "MipIdif" in fqfn:
            SIGMA = 0.001

        # load img
        nii = nib.load(fqfn)
        img = nii.get_fdata()
        if self.__remove_baseline:
            img = img - img[0, 0]
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
    def ndims(self):
        return 14

    @property
    def truths(self):
        """/Volumes/PrecunealSSD/Singularity/CCIR_01211/derivatives/sub-108293/ses-20210421150523/pet for mipidif"""
        """/Volumes/PrecunealSSD/Singularity/CCIR_01211/sourcedata/sub-108293/ses-20210421/pet for twilite nomodel"""

        return [8.95, 2.34, 6.05,
                4.43, 4.23, 4.24, -2.00, -0.36, 8.69,
                0.49, 0.19, 0.084,
                2.56, 0.001]

    def plot_results(self, res: dyutils.Results):
        class_name = self.__class__.__name__

        qm, _, _ = self.quantile(res)
        self.plot_truths(qm)
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-results.png")

        dyplot.runplot(res)
        plt.tight_layout()
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-runplot.png")

        fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm,
                                     fig=plt.subplots(14, 2, figsize=(16, 50)))
        fig.tight_layout()
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-traceplot.png")

        dyplot.cornerplot(res, truths=qm, show_titles=True,
                          title_kwargs={'y': 1.04}, labels=self.labels,
                          fig=plt.subplots(14, 14, figsize=(100, 100)))
        plt.savefig(self.fqfp + "_dynesty-" + class_name + "-cornerplot.png")

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
        plt.xlim([-0.1, 1.1 * np.max(tM)])
        plt.xlabel('time of mid-frame (s)')
        plt.ylabel('activity (Bq/mL)')
        plt.tight_layout()

    def plot_variations(self, tindex=0, tmin=None, tmax=None, truths=None):
        if truths is None:
            truths = self.truths

        plt.figure(figsize=(12, 7.4))

        ncolors: int = 75
        viridis = cm.get_cmap('viridis', ncolors)
        dt = (tmax - tmin) / ncolors
        trange = np.arange(tmin, tmax, dt)
        for tidx, t in enumerate(trange):
            truths[tindex] = t
            data = self.data(truths)
            _, rho_ideal, t_ideal = self.signalmodel(data)
            plt.plot(t_ideal, rho_ideal, color=viridis(tidx))

        # plt.xlim([-0.1,])
        plt.xlabel('time of mid-frame (s)')
        plt.ylabel('activity (arbitrary)')

        # Add a colorbar to understand colors
        # First create a mappable object with the same colormap
        sm = plt.cm.ScalarMappable(cmap=viridis)
        sm.set_array(trange)
        plt.colorbar(sm, label="Varying " + self.labels[tindex])

        plt.tight_layout()

    def prior_transform(self, tracer):
        return {
            'co': self.prior_transform_co,
            'oc': self.prior_transform_co,
            'oo': self.prior_transform_oo
        }.get(tracer, self.prior_transform_default)

    def run_nested(self, checkpoint_file=None):
        """ checkpoint_file=self.fqfp+"_dynesty-RadialArtery.save") """

        return self.solver.run_nested(prior_tag=self.tracer,
                                      ndim=14,
                                      checkpoint_file=checkpoint_file)

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
        df.to_csv(self.fqfp + "_dynesty-" + class_name + "-signal.csv")
        d_ideal = {
            "times": times,
            "ideal": M0 * rho_ideal}
        df = pd.DataFrame(d_ideal)
        df.to_csv(self.fqfp + "_dynesty-" + class_name + "-ideal.csv")
        d_quantiles = {
            "label": self.labels,
            "qm": qm,
            "ql": ql,
            "qh": qh}
        df = pd.DataFrame(d_quantiles)
        df.to_csv(self.fqfp + "_dynesty-" + class_name + "-quantiles.csv")

    @staticmethod
    def data2t(data: dict):
        timesMid = data['timesMid']
        taus = data['taus']
        t0 = timesMid[0] - taus[0] / 2
        tF = timesMid[-1] + taus[-1] / 2
        t = np.arange(t0, tF)
        return t

    @staticmethod
    def prior_transform_co(u):
        v = u
        v[0] = u[0] * 60  # t_0
        v[1] = u[1] * 60  # \tau_2 ~ t_2 - t_0
        v[2] = u[2] * 60  # \tau_3 ~ t_3 - t_2
        v[3] = u[3] * 20  # \alpha - 1
        v[4] = u[4] * 30 + 1  # 1/\beta
        v[5] = u[5] * 10 + 0.25  # p
        v[6] = u[6] * 10 - 10  # \delta p_2 ~ p_2 - p
        v[7] = u[7] * 10 - 10  # \delta p_3 ~ p_3 - p_2
        v[8] = u[8] * 300 + 0.01  # 1/\gamma for s.s.
        v[9] = u[9] * 0.75 + 0.25  # f_2
        v[10] = u[10] * 0.75  # f_3
        v[11] = u[11] * 0.75  # f_{ss}
        v[12] = u[12] * 4 + 0.5  # A is amplitude adjustment
        v[13] = u[13] * SIGMA  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_oo(u):
        v = u
        v[0] = u[0] * 60  # t_0
        v[1] = u[1] * 60  # \tau_2 ~ t_2 - t_0
        v[2] = u[2] * 60  # \tau_3 ~ t_3 - t_2
        v[3] = u[3] * 20  # \alpha - 1
        v[4] = u[4] * 30 + 1  # 1/\beta
        v[5] = u[5] * 10 + 0.25  # p
        v[6] = u[6] * 10 - 10  # \delta p_2 ~ p_2 - p
        v[7] = u[7] * 10 - 10  # \delta p_3 ~ p_3 - p_2
        v[8] = u[8] * 300 + 0.01  # 1/\gamma for s.s.
        v[9] = u[9] * 0.75 + 0.25  # f_2
        v[10] = u[10] * 0.5  # f_3
        v[11] = u[11] * 0.25  # f_{ss}
        v[12] = u[12] * 4 + 0.5  # A is amplitude adjustment
        v[13] = u[13] * SIGMA  # sigma ~ fraction of M0
        return v

    @staticmethod
    def prior_transform_default(u):
        v = u
        v[0] = u[0] * 60  # t_0
        v[1] = u[1] * 60  # \tau_2 ~ t_2 - t_0
        v[2] = u[2] * 60  # \tau_3 ~ t_3 - t_2
        v[3] = u[3] * 20  # \alpha - 1
        v[4] = u[4] * 30 + 1  # 1/\beta
        v[5] = u[5] * 10 + 0.25  # p
        v[6] = u[6] * 10 - 10  # \delta p_2 ~ p_2 - p
        v[7] = u[7] * 10 - 10  # \delta p_3 ~ p_3 - p_2
        v[8] = u[8] * 300 + 0.01  # 1/\gamma for s.s.
        v[9] = u[9] * 0.5  # f_2
        v[10] = u[10] * 0.25  # f_3
        v[11] = u[11] * 0.25  # f_{ss}
        v[12] = u[12] * 4 + 0.5  # A is amplitude adjustment
        v[13] = u[13] * SIGMA  # sigma ~ fraction of M0
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
