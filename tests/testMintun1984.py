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

import nibabel as nib
import numpy as np
import dynesty.utils as dyutils

from testPreliminaries import TestPreliminaries 
from IOImplementations import TissueIO
from Mintun1984Context import Mintun1984Context
from Mintun1984Data import Mintun1984Data
from Mintun1984Solver import Mintun1984Solver
from TissuePlotting import TissuePlotting


class TestMintun1984(TestPreliminaries):

    def setUp(self):
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "kernels")
        kern = os.path.join(kerndir, "kernel_hct=46.8.nii.gz")

        oo1dir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421150523", "pet")
        idif = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-MipIdif_idif_dynesty-Boxcar-ideal.nii.gz")
        twil = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
        pet = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")
        
        hodir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421152358", "pet")
        ks_idif = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-Raichle1983Boxcar-main7-rc1p85-vrc1-3000-qm.nii.gz")
        ks_twil = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-Raichle1983Artery-main7-rc1p85-vrc1-3000-qm.nii.gz")

        codir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421144815", "pet")
        v1_idif = os.path.join(codir, "sub-108293_ses-20210421144815_trc-co_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-idif_martinv1.nii.gz")
        v1_twil = os.path.join(codir, "sub-108293_ses-20210421144815_trc-co_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-twilite_martinv1.nii.gz")

        self.data_dict_idif = {
            "input_func_fqfn": idif,
            "tissue_fqfn": pet,
            "v1_fqfn": v1_idif,
            "ks_fqfn": ks_idif,
        }
        self.data_dict_twil = {
            "kernel_fqfn": kern,
            "input_func_fqfn": twil,
            "tissue_fqfn": pet,
            "v1_fqfn": v1_twil,
            "ks_fqfn": ks_twil,
        }

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_ctor(self):
        context = Mintun1984Context(self.data_dict_idif)
        # print("\n")
        # pprint(context)
        self.assertIsInstance(context.io, TissueIO)
        self.assertIsInstance(context.data, Mintun1984Data)
        self.assertIsInstance(context.solver, Mintun1984Solver)
        self.assertIsInstance(context.plotting, TissuePlotting)

        context = Mintun1984Context(self.data_dict_twil)
        # print("\n")
        # pprint(context)
        self.assertIsInstance(context.io, TissueIO)
        self.assertIsInstance(context.data, Mintun1984Data)
        self.assertIsInstance(context.solver, Mintun1984Solver)
        self.assertIsInstance(context.plotting, TissuePlotting)

    def test_io_idif(self):
        context = Mintun1984Context(self.data_dict_idif)        
        print("\n")
        pprint(context.io.fqfp)
        pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["input_func_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (90,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (90,))
        self.assertEqual(niid["times"].shape, (90,))
        self.assertEqual(niid["timesMid"].shape, (90,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_twil(self):
        context = Mintun1984Context(self.data_dict_twil)        
        print("\n")
        pprint(context.io.fqfp)
        pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_twil["input_func_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_twil["input_func_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (179,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (179,))
        self.assertEqual(niid["times"].shape, (179,))
        self.assertEqual(niid["timesMid"].shape, (179,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_tissue(self):
        context = Mintun1984Context(self.data_dict_idif)        
        print("\n")
        pprint(context.io.fqfp)
        pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["tissue_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309,32))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (32,))
        self.assertEqual(niid["times"].shape, (32,))
        self.assertEqual(niid["timesMid"].shape, (32,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_v1(self):
        context = Mintun1984Context(self.data_dict_idif)        
        print("\n")
        pprint(context.io.fqfp)
        pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["v1_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["v1_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (290,))
        self.assertEqual(niid["times"].shape, (290,))
        self.assertEqual(niid["timesMid"].shape, (290,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_ks(self):
        context = Mintun1984Context(self.data_dict_idif)        
        print("\n")
        pprint(context.io.fqfp)
        pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["ks_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["ks_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309,6))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (110,))
        self.assertEqual(niid["times"].shape, (110,))
        self.assertEqual(niid["timesMid"].shape, (110,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_data(self):
        context = Mintun1984Context(self.data_dict_idif)
        self.assertEqual(context.data.data_dict["halflife"], 122.2416)
        self.assertEqual(context.data.data_dict["input_func_fqfn"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(context.data.data_dict["tissue_fqfn"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(context.data.data_dict["v1_fqfn"], self.data_dict_idif["v1_fqfn"])
        self.assertEqual(context.data.data_dict["ks_fqfn"], self.data_dict_idif["ks_fqfn"])
        # self.assertEqual(context.data.data_dict["v1"].shape, (309,))
        # self.assertEqual(context.data.data_dict["ks"].shape, (309,6))
        # self.assertEqual(context.data.data_dict["rho"].shape, (309,32))
        # self.assertEqual(context.data.data_dict["tag"], "")
        # self.assertEqual(context.data.data_dict["taus"].shape, (32,))
        # self.assertEqual(context.data.data_dict["times"].shape, (32,))
        # self.assertEqual(context.data.data_dict["timesMid"].shape, (32,))        
        context.data.print_concise(context.data.data_dict, "context.data_dict")
        # context.data.print_data()

        aifm = context.data.rho_input_func_measurement
        self.assertIn(aifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(aifm["img"].shape, (90,))
        self.assertAlmostEqual(np.max(aifm["img"]), 17.652565145242843, places=6)

        atm = context.data.rho_tissue_measurement
        self.assertIn(atm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(atm["img"].shape, (309,32))
        self.assertAlmostEqual(np.max(atm["img"]), 1.0, places=6)

        ifm = context.data.input_func_measurement
        self.assertIn(ifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(ifm["img"].shape, (90,))

        self.assertEqual(context.data.input_func_type, "Boxcar")
        self.assertTrue(context.data.isidif)
        self.assertAlmostEqual(context.data.max_tissue_measurement, 53984.688610472745, places=12)
        self.assertAlmostEqual(context.data.recovery_coefficient, 1.85, places=2)
        self.assertAlmostEqual(context.data.rho.max(), 1.0, places=6)
        
        v1 = context.data.v1_measurement
        self.assertIn(v1["fqfp"], self.data_dict_idif["v1_fqfn"])
        self.assertEqual(v1["img"].shape, (309,))
        
        tm = context.data.tissue_measurement
        self.assertIn(tm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(tm["img"].shape, (309,32))
        
        ks = context.data.ks_measurement
        self.assertIn(ks["fqfp"], self.data_dict_idif["ks_fqfn"])
        self.assertEqual(ks["img"].shape, (309,6))

    def test_solver(self):
        context = Mintun1984Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        qm_expected = [
            2.448700e-01,  1.273792e+00,  6.570719e-01,  1.008356e+01,
            -3.026665e+01,  1.734506e+01,  1.710380e-02]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()
