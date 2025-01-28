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
import numpy as np

from PETUtilities import PETUtilities
from Raichle1983Context import Raichle1983Context
from testPreliminaries import TestPreliminaries 


class TestPETUtilities(TestPreliminaries):

    def setUp(self):
        self.tac_decay_fast = {
            "img": np.power(2, -np.linspace(0.5, 9.5, 10)/2),
            "taus": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "times": np.linspace(0, 9, 10),
            "timesMid": np.linspace(0.5, 9.5, 10),
            "halflife": 2
        }
        self.tac_decay_slow = {
            "img": np.power(2, -np.linspace(0.5, 9.5, 10)/9),
            "taus": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "times": np.linspace(0, 9, 10),
            "timesMid": np.linspace(0.5, 9.5, 10),
            "halflife": 9
        }
        self.tac_const_fast = {
            "img": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "taus": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "times": np.linspace(0, 9, 10),
            "timesMid": np.linspace(0.5, 9.5, 10),
            "halflife": 2
        }
        self.tac_const_slow = {
            "img": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "taus": np.ones_like(np.linspace(0.5, 9.5, 10)),
            "times": np.linspace(0, 9, 10),
            "timesMid": np.linspace(0.5, 9.5, 10),
            "halflife": 9
        }

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_data2timesInterp_bug(self):
        hodir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421152358", "pet")
        idif = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-MipIdif_idif_dynesty-Boxcar-ideal.nii.gz")
        pet = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")        
        pickle_fqfn = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-Boxcar.pickle")
        data_dict_idif = {
            "input_func_fqfn": idif,
            "tissue_fqfn": pet,
            "tag": ""
        }
        context = Raichle1983Context(data_dict_idif)
        res = context.pickle_load(pickle_fqfn)
        context.solver.results_save(
            tag="test_data2timesInterp_bug",
            results=res,
            do_pickle=False)

    def test_data2timesInterp(self):
        d = {"times": [0.1, 1.1, 2.1, 4.1, 8.1, 16.1], "taus": [1, 1, 2, 4, 8, 16]}
        times = PETUtilities.data2timesInterp(d)
        print("\n")
        pprint(times)
        times_expected = np.linspace(0.1, 31.1, 32)
        np.testing.assert_allclose(times, times_expected)    

    def test_data2timesMidInterp(self):
        # simple, for Twilite AIF
        d = {"times": [0, 1, 2, 3, 4, 5], "taus": [1, 1, 1, 1, 1, 1]}
        times = PETUtilities.data2timesMidInterp(d)
        print("\n")
        pprint(times)
        times_expected = np.linspace(0.5, 5.5, 6)
        np.testing.assert_allclose(times, times_expected)

        # stretched, offset, for Boxcar IDIF
        d = {"times": [0.1, 1.1, 2.1, 4.1, 8.1, 16.1], "taus": [1, 1, 2, 4, 8, 16]}
        times = PETUtilities.data2timesMidInterp(d)
        print("\n")
        pprint(times)
        times_expected = np.linspace(0.6, 31.6, 32)
        np.testing.assert_allclose(times, times_expected)        

    def test_decay_correct_fast(self):
        print("\n")
        pprint(self.tac_decay_fast)
        tac = PETUtilities.decay_correct(self.tac_decay_fast)
        pprint(tac)

    def test_decay_uncorrect_fast(self):
        print("\n")
        pprint(self.tac_const_fast)
        tac = PETUtilities.decay_uncorrect(self.tac_const_fast)
        pprint(tac)

    def test_decay_correct_slow(self):
        print("\n")
        pprint(self.tac_decay_slow)
        tac = PETUtilities.decay_correct(self.tac_decay_slow)
        pprint(tac)

    def test_decay_uncorrect_slow(self):
        print("\n")
        pprint(self.tac_const_slow)
        tac = PETUtilities.decay_uncorrect(self.tac_const_slow)
        pprint(tac)


if __name__ == '__main__':
    unittest.main()
