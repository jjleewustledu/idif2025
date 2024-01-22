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

# basic numeric setup
import numpy as np
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

from Artery import Artery
from dynesty import dynesty
from dynesty import plotting as dyplot

RHO = None
TAUS = None
TIMES_MID = None

__all__ = ["Boxcar"]


class Boxcar(Artery):

    def __init__(self, input_func_measurement,
                 sample='rslice',
                 nlive=55,
                 rstate=None):
        super().__init__(input_func_measurement)

        if not rstate:
            # seed the random number generator
            rstate = np.random.default_rng(916301)

        self.sample = sample
        self.nlive = nlive
        self.rstate = rstate

        global RHO, TAUS, TIMES_MID
        ifm = self.input_func_measurement
        RHO = ifm['img'] / np.max(ifm['img'])
        TAUS = ifm['taus']
        TIMES_MID = ifm['timesMid']

    def plot_results(self, res: dynesty.DynamicNestedSampler.results):
        qm, _, _ = self.quantile(res)
        self.plot_truths(qm)

        dyplot.runplot(res)
        plt.tight_layout()

        fig, axes = dyplot.traceplot(res, labels=self.labels, truths=qm,
                                     fig=plt.subplots(11, 2, figsize=(16, 40)))
        fig.tight_layout()

        fig, axes = dyplot.cornerplot(res, truths=qm, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=self.labels,
                                      fig=plt.subplots(11, 11, figsize=(100, 100)))

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
        plt.ylabel('activity (Bq/mL)')
        plt.tight_layout()

    def run_nested(self, checkpoint_file=None):
        """ checkpoint_file=self.fqfp+"_dynesty-Boxcar.save") """

        sampler = dynesty.DynamicNestedSampler(self.loglike, self.prior_transform, 11,
                                               sample=self.sample, nlive=self.nlive,
                                               rstate=self.rstate)
        sampler.run_nested(checkpoint_file)
        return sampler.results

    @property
    def labels(self):
        """"""

        return [
            r'$t_0$', r'$\tau_2$',
            r'$\alpha - 1$', r'$1/\beta$', r'$p$', r'$\delta p_2$', r'$1/\gamma$',
            r'$f_2$', r'$f_{ss}$',
            r'$A$', r'$\sigma$']

    @property
    def truths(self):
        """/Volumes/PrecunealSSD/Singularity/CCIR_01211/derivatives/sub-108293/ses-20210421150523/pet"""

        return [15.7, 8.73,
                0.29, 3.04, 2.23, 0, 53.11,
                0.31, 0.07,
                1.98, 0.01]

    @staticmethod
    def data(v):
        return {'timesMid': TIMES_MID, 'taus': TAUS, 'v': v}

    @staticmethod
    def loglike(v):
        data = Boxcar.data(v)
        rho_pred, _, _ = Boxcar.signalmodel(data)
        sigma = v[-1]
        residsq = (rho_pred - RHO) ** 2 / sigma ** 2
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma ** 2))

        if not np.isfinite(loglike):
            loglike = -1e300

        return loglike

    @staticmethod
    def prior_transform(u):
        v = u
        v[0] = u[0] * 30 + 5  # t_0
        v[1] = u[1] * 20 + 5  # \tau_2 ~ t_2 - t_0
        v[2] = u[2] * 5  # \alpha - 1
        v[3] = u[3] * 10 + 0.1  # 1/\beta
        v[4] = u[4] * 3 + 0.5  # p
        v[5] = u[5] * 0.05 - 0.025  # \delta p_2 ~ p_2 - p
        v[6] = u[6] * 100 + 5  # 1/\gamma for s.s.
        v[7] = u[7] * 0.75 + 0.25 # f_2
        v[8] = u[8] * 0.125  # f_{ss}
        v[9] = u[9] * 3 + 1 # A is amplitude adjustment
        v[10] = u[10] * 0.02  # sigma ~ fraction of M0
        return v

    @staticmethod
    def signalmodel(data: dict):
        t_ideal = Boxcar.data2t(data)
        v = data['v']
        t_0 = v[0]
        tau_2 = v[1]
        a = v[2]
        b = 1 / v[3]
        p = v[4]
        dp_2 = v[5]
        g = 1 / v[6]
        f_2 = v[7]
        f_ss = v[8]
        A = v[9]

        rho_ = A * Boxcar.solution_3bolus(t_ideal, t_0, tau_2, a, b, p, dp_2, g, f_2, f_ss)
        rho = Boxcar.apply_boxcar(rho_, data)
        A_qs = 1 / max(rho)
        signal = A_qs * rho
        ideal = A_qs * rho_
        return signal, ideal, t_ideal

    @staticmethod
    def apply_boxcar(vec, data):
        times0 = data['timesMid'] - data['taus'] / 2
        timesF = data['timesMid'] + data['taus'] / 2

        vec_sampled = np.full(times0.shape, np.nan)
        for idx, (t0, tF) in enumerate(zip(times0, timesF)):
            vec_sampled[idx] = np.mean(vec[int(t0):int(tF)])
        return vec_sampled
