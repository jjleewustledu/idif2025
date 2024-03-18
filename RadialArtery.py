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

from Artery import Artery

# general & system functions
import os
from copy import deepcopy

# basic numeric setup
import numpy as np
import nibabel as nib


def kernel_fqfn(artery_fqfn: str):
    """
    :param artery_fqfn: The fully qualified file name for an artery.
    :return: The fully qualified file name for the corresponding kernel file.

    This method takes a fully qualified file name for an artery and returns the fully qualified file name for the corresponding kernel file. It uses the artery_fqfn parameter to determine
    * which kernel file to return based on the substring contained in the artery_fqfn.

    Example usage:

        artery_file = "/path/to/sub-108293_artery.nii.gz"
        kernel_file = kernel_fqfn(artery_file)
        print(kernel_file)  # /path/to/CCIR_01211/sourcedata/kernel_hct=46.8.nii.gz
    """
    sourcedata = os.path.join(os.getenv("SINGULARITY_HOME"), "CCIR_01211", "sourcedata",)
    if "sub-108293" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=46.8.nii.gz")
    if "sub-108237" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=43.9.nii.gz")
    if "sub-108254" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=37.9.nii.gz")
    if "sub-108250" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=42.8.nii.gz")
    if "sub-108284" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=39.7.nii.gz")
    if "sub-108306" in artery_fqfn:
        return os.path.join(sourcedata, "kernel_hct=41.1.nii.gz")

    # mean hct for females and males
    return os.path.join(sourcedata, "kernel_hct=44.5.nii.gz")


class RadialArtery(Artery):
    """
    Represents a radial artery.

    Args:
        input_func_measurement (dict): The measurement data for the input function.
        kernel_measurement (dict, optional): The measurement data for the kernel. Default is None.
        remove_baseline (bool, optional): Flag to indicate whether to remove baseline. Default is True.
        tracer (object, optional): The tracer object. Default is None.
        truths (object, optional): The truths object. Default is None.
        sample (str, optional): The sample type. Default is "rslice".
        nlive (int, optional): The number of live points. Default is 1000.
        rstate (RandomState, optional): The random state. Default is np.random.default_rng(916301).
        tag (str, optional): The tag. Default is "".

    Attributes:
        KERNEL (np.array): The kernel array.
        SIGMA (float): The sigma value.

    Properties:
        kernel_measurement (dict): The kernel measurement data.

    Methods:
        signalmodel(data: dict) -> tuple: Calculate the signal model of the radial artery.
        apply_dispersion(vec, data: dict) -> np.array: Apply dispersion to a vector.

    """
    def __init__(self, input_func_measurement,
                 kernel_measurement=None,
                 remove_baseline=True,
                 tracer=None,
                 truths=None,
                 sample="rslice",
                 nlive=1000,
                 rstate=np.random.default_rng(916301),
                 tag=""):
        super().__init__(input_func_measurement,
                         tracer=tracer,
                         truths=truths,
                         sample=sample,
                         nlive=nlive,
                         rstate=rstate,
                         tag=tag)

        self.__kernel_measurement = kernel_measurement  # set with fqfn
        self.KERNEL = self.kernel_measurement["img"].copy()  # get dict conforming to nibabel
        self.__remove_baseline = remove_baseline
        self.SIGMA = 0.1

    @property
    def kernel_measurement(self):
        if isinstance(self.__kernel_measurement, dict):
            return deepcopy(self.__kernel_measurement)

        if self.__kernel_measurement is None:
            self.__kernel_measurement = kernel_fqfn(self.input_func_measurement["fqfp"] + ".nii.gz")

        assert os.path.isfile(self.__kernel_measurement), f"{self.__kernel_measurement} was not found."
        fqfn = self.__kernel_measurement
        base, ext = os.path.splitext(fqfn)
        fqfp = os.path.splitext(base)[0]

        # load img
        nii = nib.load(fqfn)
        img = nii.get_fdata()

        # assemble dict
        self.__kernel_measurement = {
            "fqfp": fqfp,
            "img": np.array(img, dtype=float).reshape(-1)}
        return deepcopy(self.__kernel_measurement)

    @staticmethod
    def signalmodel(data: dict):
        t_ideal = np.arange(data["rho"].size)
        v = data["v"]
        t_0 = v[0]
        tau_2 = v[1]
        tau_3 = v[2]
        a = v[3]
        b = 1 / v[4]
        p = v[5]
        dp_2 = v[6]
        dp_3 = v[7]
        g = 1 / v[8]
        f_2 = v[9]
        f_3 = v[10]
        f_ss = v[11]
        A = v[12]

        # rho_ = A * RadialArtery.solution_1bolus(t_ideal, t_0, a, b, p)
        # rho_ = A * RadialArtery.solution_2bolus(t_ideal, t_0, a, b, p, g, f_ss)
        # rho_ = A * RadialArtery.solution_3bolus(t_ideal, t_0, tau_2, a, b, p, dp_2, g, f_2, f_ss)
        rho_ = A * RadialArtery.solution_4bolus(t_ideal, t_0, tau_2, tau_3, a, b, p, dp_2, dp_3, g, f_2, f_3, f_ss)
        rho = RadialArtery.apply_dispersion(rho_, data)
        A_qs = 1 / max(rho)
        signal = A_qs * rho
        ideal = A_qs * rho_
        return signal, ideal, t_ideal

    @staticmethod
    def apply_dispersion(vec, data: dict):
        k = data["kernel"].copy()
        k = k.T
        vec_sampled = np.convolve(vec, k, mode="full")
        return vec_sampled[:vec.size]
