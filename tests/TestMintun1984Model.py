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

from matplotlib import pyplot as plt
from Mintun1984Model import Mintun1984Model
from testPreliminaries import TestPreliminaries 


class TestMintun1984Model(TestPreliminaries):

    def setUp(self):
        self.idif = os.path.join(self.petdir("oo1", singularity=False), "sub-108293_ses-20210421150523_trc-oo_proc-MipIdif_idif_dynesty-Boxcar-ideal.nii.gz")
        self.twil = os.path.join(self.petdir("oo1", singularity=False), "sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
        self.kern = self.kernel_fqfn(hct=46.8)
        self.pet = self.fqfn_ParcSchaeffer("oo1", singularity=False)  # os.path.join(self.petdir("oo1", singularity=True), "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")
        self.truths_idif = [0.412, 0.978, 0.978, 14.834, -11, 0.025]
        self.truths_twil = [0.47, 0.27, 0.886, 6.742, -14, 0.024]
        self._parc_index = 25
        self._mintun_idif_obj = Mintun1984Model(
            self.idif, 
            self.pet, 
            truths=self.truths_idif, 
            nlive=100, 
            tag="main7-rc1p85-vrc1-3000") 
        self._mintun_twil_obj = Mintun1984Model(
            self.twil, 
            self.pet, 
            truths=self.truths_twil, 
            nlive=100, 
            tag="main7-rc1p85-vrc1-3000") 

        os.chdir(self.petdir("oo1"))

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_ctor_idif(self):
        self._mintun_idif_obj = Mintun1984Model(
            self.idif, 
            self.pet, 
            truths=self.truths_idif, 
            nlive=100, 
            tag="main7-rc1p85-vrc1-3000")
        pprint(self._mintun_idif_obj)    

    def test_ctor_twil(self):
        self._mintun_twil_obj = Mintun1984Model(
            self.twil, 
            self.pet, 
            truths=self.truths_twil, 
            nlive=100, 
            tag="main7-rc1p85-vrc1-3000")
        pprint(self._mintun_twil_obj)

    def test_data(self):
        if self._mintun_twil_obj:
            v = self._mintun_twil_obj.truths
            pprint(self._mintun_twil_obj.data(v))

    def test_plot_truths(self):
        if self._mintun_twil_obj:
            self._mintun_twil_obj.plot_truths(parc_index=self._parc_index)
            plt.show()

    def test_plot_variations(self): 
        if self._mintun_twil_obj:
            v = self._mintun_twil_obj.truths
            self._mintun_twil_obj.plot_variations(0, 0.05, 0.95, v)
            plt.show()

    def test_run_nested_for_indexed_tac(self):
        if self._mintun_twil_obj:
            self._mintun_twil_obj.run_nested_for_indexed_tac(self._parc_index, print_progress=True)
            plt.show()
if __name__ == '__main__':
    unittest.main()
