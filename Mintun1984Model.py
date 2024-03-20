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

from TCModel import TCModel
from Boxcar import Boxcar

# general & system functions
import os

# basic numeric setup
import numpy as np


class Mintun1984Model(TCModel):
    """
    Mintun1984Model

    The Mintun1984Model class is a subclass of the TCModel class and represents the Mintun 1984 model for PET imaging data analysis.

    Attributes:
        labels (list of str): List of labels for each parameter of the model.
            These labels correspond to the order of the parameter values in the 'v' input.

    Methods:
        signalmodel(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Calculates the model signal given the input data.

            Args:
                data (dict): A dictionary containing the following keys and values:
                    - "halflife" (float): The radioactive isotope's half-life.
                    - "timesMid" (np.ndarray): The mid-point times of the PET measurements.
                    - "inputFuncInterp" (np.ndarray): The interpolated input function data.
                    - "raichleks" (np.ndarray): The Raichle-Kety parameters.
                    - "martinv1" (float): The Martin arterial volume parameter.
                    - "v" (np.ndarray): The model parameter values.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following arrays:
                    - rho (np.ndarray): The calculated model signal.
                    - timesMid (np.ndarray): The mid-point times of the PET measurements.
                    - rho_t (np.ndarray): The model signal at each time point.
                    - times (np.ndarray): The time values.

    """
    def __init__(self,
                 input_function,
                 pet_measurement,
                 truths=None,
                 home=os.getcwd(),
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag=""):
        super().__init__(input_function,
                         pet_measurement,
                         truths=truths,
                         home=home,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         tag=tag)

    @property
    def labels(self):
        return [
            r"OEF", r"frac. water of metab.", r"$v_{post} + 0.5 v_{cap}$", r"$t_0$", r"$\tau_a$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):

        hl = data["halflife"]
        timesMid = data["timesMid"]
        input_func_interp = data["inputFuncInterp"]
        raichleks = data["raichleks"]
        v1 = data["martinv1"]
        v = data["v"]

        ALPHA = np.log(2) / hl
        DENSITY_PLASMA = 1.03
        DENSITY_BLOOD = 1.06

        oef = v[0]
        metab_frac = v[1]
        v_post_cap = v[2]
        t_0 = v[3]
        tau_a = v[4]
        if raichleks.ndim == 1:
            f = raichleks[0]
            lamb = raichleks[1]
            PS = raichleks[2]
        elif raichleks.ndim == 2:
            f = raichleks[:, 0]
            lamb = raichleks[:, 1]
            PS = raichleks[:, 2]
        else:
            raise RuntimeError(Mintun1984Model.signalmodel.__name__+": raichleks.ndim->"+raichleks.ndim)
        m = 1 - np.exp(-PS / f)
        n = len(input_func_interp)
        times = np.arange(n)
        input_func_interp = Mintun1984Model.slide(input_func_interp, times, tau_a, hl)
        indices = np.where(input_func_interp > 0.05 * max(input_func_interp))
        try:
            idx0 = max([indices[0][0], 1])
        except IndexError:
            idx0 = 1
        idxU = min([idx0 + 90, n - 1])  # cf. Mintun1984

        # estimate shape of water of metabolism
        shape = np.zeros(n)
        n1 = n - idx0 + 1
        try:
            y = (n - idx0) / (idxU - idx0)
        except ZeroDivisionError:
            y = 1
        shape[-n1:] = np.linspace(0, y, n1)  # shape(idxU) == 1
        duc_times = np.zeros(n)
        duc_times[idx0:] = np.linspace(0, n1 - 2, n1 - 1)
        duc_shape = shape * np.power(2, -(duc_times - idxU + 1) / hl)  # decay-uncorrected

        # set scale of artery_h2o
        # activity of water of metab \approx activity of oxygen after 90 sec
        metab_scale = metab_frac * input_func_interp[idxU]
        metab_scale = metab_scale * DENSITY_PLASMA / DENSITY_BLOOD
        artery_h2o = metab_scale * duc_shape

        # compartment 2, using m, f, lamb
        artery_o2 = input_func_interp - artery_h2o
        artery_o2[artery_o2 < 0] = 0
        kernel = np.exp(-m * f * times / lamb - ALPHA * times)
        rho2 = (m * f * np.convolve(kernel, artery_h2o, mode="full") +
                oef * m * f * np.convolve(kernel, artery_o2, mode="full"))

        # compartment 1
        # v_post = 0.83*v1
        # v_cap = 0.01*v1
        # R = 0.85  # ratio of small-vessel to large-vessel Hct
        rho1 = v1 * (1 - oef * v_post_cap) * artery_o2

        rho_t = rho1[:n] + rho2[:n]
        rho_t = Mintun1984Model.slide(rho_t, times, t_0, hl)
        # rho = np.interp(timesMid, times, rho_t)
        rho = Boxcar.apply_boxcar(rho_t, data)
        return rho, timesMid, rho_t, times
