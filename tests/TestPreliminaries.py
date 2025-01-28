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
import unittest
import os
from pprint import pprint


class TestPreliminaries(unittest.TestCase):
    
    def fqfn_ParcSchaeffer(self, tracer="oo1", singularity=False):
        if tracer == "co":
            fn = ("sub-108293_ses-20210421144815_trc-co_proc-delay0-BrainMoCo2-"
                 "createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-"
                 "to-schaeffer-schaeffer.nii.gz")
        elif tracer == "oo1":
            fn = ("sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-"
                 "createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-"
                 "to-schaeffer-schaeffer.nii.gz")
        elif tracer == "ho":
            fn = ("sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-"
                 "createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-"
                 "to-schaeffer-schaeffer.nii.gz")
        elif tracer == "oo2":
            fn = "sub-108293_ses-20210421154248_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz"
        elif tracer == "fdg":
            fn = "sub-108293_ses-20210421155709_trc-fdg_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz"
        return os.path.join(self.petdir(tracer, singularity), fn)
    
    def kernel_fqfn(self, hct=44.5):
        fqfn = os.path.join(
            os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "kernels", f"kernel_hct={hct}.nii.gz")
        if not os.path.exists(fqfn):
            raise FileNotFoundError(f"Kernel file not found: {fqfn}")
        return fqfn
    
    def petdir(self, tracer="oo1", singularity=False):

        if tracer == "co":
            ses = "ses-20210421144815"
        elif tracer == "oo1":
            ses = "ses-20210421150523"
        elif tracer == "ho":
            ses = "ses-20210421152358"
        elif tracer == "oo2":
            ses = "ses-20210421154248"
        elif tracer == "fdg":
            ses = "ses-20210421155709"

        if singularity: 
            return os.path.join(
                os.getenv("SINGULARITY_HOME"), "CCIR_01211", "derivatives", "sub-108293", ses, "pet")
        else:
            return os.path.join(
                os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", ses, "pet")

    def print_separator(self, text):
        print("\n" + "=" * 30 + " " + text + " " + "=" * 30)
