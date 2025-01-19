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
from Raichle1983Model import Raichle1983Model
from testPreliminaries import TestPreliminaries 


class TestRaichle1983Model(TestPreliminaries): 

    _raichle_obj = []
    _raichle_artery_obj = []

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here 

    def test_ctor_Raichle(self):
        self.fail("Test not implemented")
        
        ifm = os.path.join(self.petdir("ho"), "sub-108293_ses-20210421152358_trc-ho_proc-MipIdif_idif.nii.gz")
        pet = os.path.join(self.petdir("ho"), "sub-108293_ses-20210421152358_trc-ho_proc-BrainMoCo2-createNiftiMovingAvgFrames-ParcWmparc-reshape-to-wmparc-select-all.nii.gz")
        self._raichle_obj = Raichle1983Model(ifm, pet, nlive=100)
        pprint(self._raichle_obj)

if __name__ == '__main__':
    unittest.main()
