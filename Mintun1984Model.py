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

from __future__ import absolute_import
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
                    - "rhoInputFuncInterp" (np.ndarray): The interpolated input function data.
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
    def __init__(self, input_function, pet_measurement, **kwargs):
        super().__init__(input_function, pet_measurement, **kwargs)

    @property
    def labels(self):
        return [
            r"OEF", r"$f_{H_2O}$", r"$v_p + 0.5 v_c$", r"$t_0$", r"$\tau_a$", r"$\tau_d$", r"$\sigma$"]

    @staticmethod
    def signalmodel(data: dict):      

        hl = data["halflife"]
        ALPHA = np.log(2) / hl
        DENSITY_PLASMA = 1.03
        DENSITY_BLOOD = 1.06
        timesMid = data["timesMid"]
        input_func_interp = data["rhoInputFuncInterp"]
        raichleks = data["raichleks"]
        v1 = data["martinv1"]
        v = data["v"]

        oef = v[0]
        f_h2o = v[1] * DENSITY_PLASMA / DENSITY_BLOOD
        v_post_cap = v[2]
        t_0 = v[3]
        tau_a = v[4]
        tau_dispersion = v[5]

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

        n = max(input_func_interp.shape)
        timesIdeal = np.arange(0, n)

        # dispersion of input function

        dispersion = np.exp(-timesIdeal / tau_dispersion)
        z_dispersion = np.sum(dispersion)
        input_func_interp = np.convolve(input_func_interp, dispersion, mode="full")
        input_func_interp = input_func_interp[:n] / z_dispersion
        if not data["rho_experiences_boxcar"]:
            # slide input function to left since its measurements is delayed by catheters
            input_func_interp = Mintun1984Model.slide(input_func_interp, timesIdeal, tau_a, hl)

        # estimate shape of water of metabolism

        indices = np.where(input_func_interp > 0.05 * max(input_func_interp))
        try:
            idx0 = max([indices[0][0], 1])
        except IndexError:
            idx0 = 1
        idxU = min([idx0 + 90, n - 1])  # cf. Mintun1984
        shape = np.zeros(n)
        n1 = n - idx0 + 1
        try:
            y = (n - idx0) / (idxU - idx0)
        except ZeroDivisionError:
            y = 1
        shape[-n1:] = np.linspace(0, y, n1)  # shape(idxU) == 1
        timesDuc = np.zeros(n)
        timesDuc[idx0:] = np.linspace(0, n1 - 2, n1 - 1)
        shape_duc = shape * np.power(2, -(timesDuc - idxU + 1) / hl)  # decay-uncorrected

        # set scale of artery_h2o
        # activity of water of metab \approx activity of oxygen after 90 sec

        artery_h2o = f_h2o * input_func_interp[idxU] * shape_duc

        # compartment 2, using m, f, lamb

        artery_o2 = input_func_interp - artery_h2o
        artery_o2[artery_o2 < 0] = 0

        m = 1 - np.exp(-PS / f)
        propagator = np.exp(-m * f * timesIdeal / lamb - ALPHA * timesIdeal)
        rho2 = (m * f * np.convolve(propagator, artery_h2o, mode="full") +
                oef * m * f * np.convolve(propagator, artery_o2, mode="full"))

        # compartment 1
        # v_post = 0.83*v1
        # v_cap = 0.01*v1
        # R = 0.85  # ratio of small-vessel to large-vessel Hct needed when v1 := CBV * R

        rho1 = v1 * (1 - oef * v_post_cap) * artery_o2

        # package compartments

        rho_ideal = rho1[:n] + rho2[:n]
        if not data["rho_experiences_boxcar"]:
            rho_ideal = Mintun1984Model.slide(rho_ideal, timesIdeal, t_0, hl)
            rho = np.interp(timesMid, timesIdeal, rho_ideal)
        else:
            rho = Boxcar.apply_boxcar(rho_ideal, data)
        return rho, timesMid, rho_ideal, timesIdeal
