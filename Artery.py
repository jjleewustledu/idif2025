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

# system functions
import time, sys, os

# basic numeric setup
import numpy as np
import dynesty
from dynesty import utils as dyutils

# NIfTI support
import nibabel as nib
import json

# plotting
import matplotlib
from matplotlib import pyplot as plt

# seed the random number generator
rstate = np.random.default_rng(916301)

__all__ = ["Artery"]


def _trim_input_func_measurement(ifm):
    assert isinstance(ifm, dict), f"{ifm} is a {type(ifm)}, but should be a dict."
    img = ifm['img']
    timesMid = ifm['timesMid']
    taus = ifm['taus']
    viable = ~np.isnan(timesMid)
    ifm.update({'img': img[viable], 'timesMid': timesMid[viable], 'taus': taus[viable]})
    return ifm


class Artery(object):

    def __init__(self,
                 input_func_measurement,
                 home=os.getcwd(),
                 nlive=1000):
        self.__input_func_measurement = input_func_measurement
        self.home = home
        self.nlive = nlive

        # Set numpy error handling for numerical issues such as underflow/overflow/invalid
        np.seterr(under='ignore')
        np.seterr(over='ignore')
        np.seterr(invalid='ignore')

    @property
    def fqfp(self):
        return self.input_func_measurement['fqfp']

    @staticmethod
    def data2t(data: dict):
        timesMid = data['timesMid']
        taus = data['taus']
        t0 = timesMid[0] - taus[0]/2
        tF = timesMid[-1] + taus[-1]/2
        t = np.arange(t0, tF)
        return t

    @property
    def input_func_measurement(self):
        if isinstance(self.__input_func_measurement, dict):
            return self.__input_func_measurement

        if os.path.isfile(self.__input_func_measurement):
            fqfn = self.__input_func_measurement

            # load img
            nii = nib.load(fqfn)
            img = nii.get_fdata()

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

    @staticmethod
    def quantile(res: dynesty.utils.Results):
        samples = res['samples'].T
        weights = res.importance_weights()
        ql = np.zeros(len(weights))
        qm = np.zeros(len(weights))
        qh = np.zeros(len(weights))
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
               f_2_ * Artery.solution_1bolus(t, t_0 + tau_2, a, b, max(0.5, p + dp_2)) +
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
               f_2_ * Artery.solution_1bolus(t, t_0 + tau_2, a, b, p + dp_2) +
               f_3_ * Artery.solution_1bolus(t, t_0 + tau_2 + tau_3, a, b, max(0.5, p + dp_2 + dp_3)) +
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




