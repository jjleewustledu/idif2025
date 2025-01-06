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

import unittest
import os
from pprint import pprint


class TestPreliminaries(unittest.TestCase):
    
    def kernel_fqfn(self, hct=44.5):
        fqfn = os.path.join(
            os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "kernels", f"kernel_hct={hct}.nii.gz")
        if not os.path.exists(fqfn):
            raise FileNotFoundError(f"Kernel file not found: {fqfn}")
        return fqfn
    
    def petdir(self, tracer="oo1"):

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

        return os.path.join(
            os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", ses, "pet")
