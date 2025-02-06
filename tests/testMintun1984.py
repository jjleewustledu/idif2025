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
import pickle
import unittest 
import os
from pprint import pprint

import nibabel as nib
import numpy as np
import dynesty.utils as dyutils
import pandas as pd

from testPreliminaries import TestPreliminaries 
from IOImplementations import TissueIO
from Mintun1984Context import Mintun1984Context
from Mintun1984Data import Mintun1984Data
from Mintun1984Solver import Mintun1984Solver
from TissuePlotting import TissuePlotting


class TestMintun1984(TestPreliminaries):

    @property
    def qm_expected_1d(self):
        return [
            0.337187,  1.060953,  0.728026, -1.686959, 11.187671,  0.013107
        ]
    
    @property
    def qm_expected_2d(self):
        return [
            [3.37187072e-01, 1.06095346e+00, 7.28026102e-01, -1.68695873e+00, 1.11876711e+01, 1.31074582e-02],
            [3.76220155e-01, 6.95255344e-01, 8.06672955e-01, -3.57661342e+00, 1.05960121e+01, 7.23063529e-03],
            [3.87279539e-01, 6.35304353e-01, 2.89642980e-01, -4.51961933e+00, 1.53542956e+01, 1.21079964e-02]
        ]

    def setUp(self):
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "kernels")
        kern = os.path.join(kerndir, "kernel_hct=46.8.nii.gz")
        oo1dir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", 
                              "ses-20210421150523", "pet")
        idif = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-MipIdif_idif_dynesty-"
                            "Boxcar-ideal.nii.gz")
        twil = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-"
                            "input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
        pet = os.path.join(oo1dir, "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-"
                           "createNiftiMovingAvgFrames_timeAppend-4-ParcSchaeffer-reshape-to-schaeffer-"
                           "schaeffer.nii.gz")
        
        hodir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", 
                             "ses-20210421152358", "pet")
        ks_idif = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-"
                               "createNiftiMovingAvgFrames-schaeffer-Raichle1983Boxcar-main7-rc1p85-vrc1-"
                               "3000-qm.nii.gz")
        ks_twil = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-"
                               "createNiftiMovingAvgFrames-schaeffer-Raichle1983Artery-main7-rc1p85-vrc1-"
                               "3000-qm.nii.gz")

        codir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", 
                             "ses-20210421144815", "pet")
        v1_idif = os.path.join(codir, "sub-108293_ses-20210421144815_trc-co_proc-delay0-BrainMoCo2-"
                               "createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-"
                               "idif_martinv1.nii.gz")
        v1_twil = os.path.join(codir, "sub-108293_ses-20210421144815_trc-co_proc-delay0-BrainMoCo2-"
                               "createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer-"
                               "twilite_martinv1.nii.gz")

        self.data_dict_idif = {
            "input_func_fqfn": idif,
            "tissue_fqfn": pet,
            "v1_fqfn": v1_idif,
            "ks_fqfn": ks_idif,
            "tag": "playground-Mintun1984-idif"
        }
        self.data_dict_twil = {
            "kernel_fqfn": kern,
            "input_func_fqfn": twil,
            "tissue_fqfn": pet,
            "v1_fqfn": v1_twil,
            "ks_fqfn": ks_twil,
            "tag": "playground-Mintun1984-twil"
        }

        self.pickle_fqfn = os.path.join(
            oo1dir, 
            "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-"
            "schaeffer-TissueIO-Boxcar-playground-Mintun1984-idif-test_pickle_dump_and_load.pickle")
        self.pickle_x3_fqfn = os.path.join(
            oo1dir,
            "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames_timeAppend-4-"
            "schaeffer-TissueIO-Boxcar-playground-Mintun1984-idif-test_pickle_dump_and_load_x3.pickle")
        
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
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
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
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
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
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["tissue_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309, 32))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (32,))
        self.assertEqual(niid["times"].shape, (32,))
        self.assertEqual(niid["timesMid"].shape, (32,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_v1(self):
        context = Mintun1984Context(self.data_dict_idif)        
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
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
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["ks_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["ks_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309, 6))
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
        self.assertEqual(aifm["img"].shape, (80,))
        self.assertAlmostEqual(np.max(aifm["img"]), 17.65022413078728, places=6)

        atm = context.data.rho_tissue_measurement
        self.assertIn(atm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(atm["img"].shape, (309, 32))
        self.assertAlmostEqual(np.max(atm["img"]), 1.0, places=6)

        ifm = context.data.input_func_measurement
        self.assertIn(ifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(ifm["img"].shape, (80,))

        self.assertEqual(context.data.input_func_type, "Boxcar")
        self.assertTrue(context.data.isidif)
        self.assertAlmostEqual(context.data.max_tissue_measurement, 53991.92113115365, places=12)
        self.assertAlmostEqual(context.data.recovery_coefficient, 1.85, places=2)
        self.assertAlmostEqual(context.data.rho.max(), 1.0, places=6)
        
        v1 = context.data.v1_measurement
        self.assertIn(v1["fqfp"], self.data_dict_idif["v1_fqfn"])
        self.assertEqual(v1["img"].shape, (309,))
        
        tm = context.data.tissue_measurement
        self.assertIn(tm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(tm["img"].shape, (309, 32))
        
        ks = context.data.ks_measurement
        self.assertIn(ks["fqfp"], self.data_dict_idif["ks_fqfn"])
        self.assertEqual(ks["img"].shape, (309, 6))

    def test_solver_run_nested(self):
        context = Mintun1984Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        # pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-4)

    def test_solver_run_nested_x3(self):
        context = Mintun1984Context(self.data_dict_idif)
        ress = context.solver.run_nested(parc_index=(25, 26, 27))
        self.assertEqual(len(ress), 3)
        for res in ress:
            self.assertIsInstance(res, dyutils.Results)
            # self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        # pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-4)

    def test_quantile(self):
        if not os.path.exists(self.pickle_fqfn):
            print("\nCall test_pickle_dump_and_load() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is single
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        # pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-4)

        # check cached qm 
        # qm1, _, _ = context.solver.quantile(verbose=False)
        # np.testing.assert_allclose(qm, qm1, rtol=1e-15)

    def test_quantile_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is 
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        # pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-4)

        # check cached qm 
        # qm1, _, _ = context.solver.quantile(verbose=False)
        # np.testing.assert_allclose(qm, qm1, rtol=1e-15)

    def test_package_results(self):
        if not os.path.exists(self.pickle_fqfn):
            print("\nCall test_pickle_dump_and_load() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")            
        
        # pickle of results is single
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check package from pickle
        pkg = context.solver.package_results(results=a_pickle, parc_index=25)
        pprint(pkg)
        np.testing.assert_allclose(pkg["information"], 13.911820134550084, rtol=1e-12)
        np.testing.assert_allclose(pkg["logz"], 79.61973508182612, rtol=1e-12)
        np.testing.assert_allclose(pkg["qm"], self.qm_expected_1d, rtol=1e-4)
        
    def test_package_results_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")            
        
        # pickle of results is list
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check package from pickle
        pkg = context.solver.package_results(results=a_pickle, parc_index=(25, 26, 27))
        pprint(pkg)
        np.testing.assert_allclose(
            pkg["information"], [13.91182013, 17.90491427, 14.34020011], rtol=1e-5)
        np.testing.assert_allclose(
            pkg["logz"], [79.61973508, 94.72808996, 81.79450775], rtol=1e-5)
        np.testing.assert_allclose(pkg["qm"], self.qm_expected_2d, rtol=1e-4)

    def test_pickle_dump_and_load(self):
        context = Mintun1984Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        fqfn = context.solver.pickle_dump(tag=self._testMethodName)
        self.assertTrue(os.path.exists(fqfn))
        self.assertTrue(os.path.getsize(fqfn) > 0)
        # pprint(f"pickled to {fqfn}")
        
        res1 = context.solver.pickle_load(fqfn)
        # pprint(f"loaded pickle from {fqfn}")

        # Compare the dictionaries containing the results data
        res_dict = res.asdict()
        res1_dict = res1.asdict()
        
        # Check all keys match
        res_keys = set(res_dict.keys())
        res1_keys = set(res1_dict.keys())
        if res_keys != res1_keys:
            print("Mismatched keys:", res_keys.symmetric_difference(res1_keys))
        self.assertEqual(res_keys, res1_keys)
        
        # Compare each dict object in the results
        for key in res_dict:
            if key == "bound":
                continue
            # Compare binary representations of objects using pickle
            # This handles both arrays and other objects consistently
            self.assertEqual(pickle.dumps(res_dict[key]), pickle.dumps(res1_dict[key]))

        # Also verify the importanceweights match exactly
        # Note: importance_weights may differ slightly due to floating point rounding
        self.assertEqual(pickle.dumps(res.importance_weights()), pickle.dumps(res1.importance_weights()))

    def test_pickle_dump_and_load_x3(self):
        context = Mintun1984Context(self.data_dict_idif)
        res_list = context.solver.run_nested(print_progress=True, parc_index=(25, 26, 27))
        fqfn = context.solver.pickle_dump(tag=self._testMethodName)
        self.assertTrue(os.path.exists(fqfn))
        self.assertTrue(os.path.getsize(fqfn) > 0)
        # pprint(f"pickled to {fqfn}")
        
        res1_list = context.solver.pickle_load(fqfn)
        # pprint(f"loaded pickle from {fqfn}")

        for res, res1 in zip(res_list, res1_list):

            # Compare the dictionaries containing the results data
            res_dict = res.asdict()
            res1_dict = res1.asdict()
            
            # Check all keys match
            res_keys = set(res_dict.keys())
            res1_keys = set(res1_dict.keys())
            if res_keys != res1_keys:
                print("Mismatched keys:", res_keys.symmetric_difference(res1_keys))
            self.assertEqual(res_keys, res1_keys)
            
            # Compare each dict object in the results
            for key in res_dict:
                if key == "bound":
                    continue
                # Compare binary representations of objects using pickle
                # This handles both arrays and other objects consistently
                self.assertEqual(pickle.dumps(res_dict[key]), pickle.dumps(res1_dict[key]))

            # Also verify the importanceweights match exactly
            # Note: importance_weights may differ slightly due to floating point rounding
            self.assertEqual(pickle.dumps(res.importance_weights()), pickle.dumps(res1.importance_weights()))

    def test_results_save(self):
        context = Mintun1984Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        fqfp = context.solver.results_save(tag=self._testMethodName, results=res, parc_index=25)
        metrics = ["resid", "qm", "ql", "qh", "information", "logz", "rho-ideal", "rho-pred", "ideal", "signal"]
        suffixes = [".nii.gz", ".json"]
        for metric in metrics:
            for suffix in suffixes:
                fqfn = f"{fqfp}-{metric}{suffix}"
                self.assertTrue(os.path.exists(fqfn), f"File does not exist: {fqfn}")
                self.assertTrue(os.path.getsize(fqfn) > 0, f"File is empty: {fqfn}")
        # pprint(f"saved results to {fqfp}-*")

    def test_truths(self):
        if not os.path.exists(self.pickle_fqfn):
            print("\nCall test_pickle_dump_and_load() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is single
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-6)

    def test_truths_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is single
        context = Mintun1984Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-6)

    def test_call(self):
        data_dict = self.data_dict_twil
        data_dict["tag"] = ""
        m = Mintun1984Context(data_dict)
        m()
        # self.assertTrue(os.path.exists(ra.data.results_fqfp))
        # self.assertTrue(os.path.getsize(ra.data.results_fqfp) > 0)

    def test_to_csv(self):
        context = Mintun1984Context(self.data_dict_idif)
        petpath = os.path.dirname(self.data_dict_idif["tissue_fqfn"])
        pkl = os.path.join(petpath, "sub-108293_ses-20210421150523_trc-oo_proc-delay0-BrainMoCo2-"
                           "createNiftiMovingAvgFrames_timeAppend-4-schaeffer-TissueIO-Boxcar-"
                           "playground-Mintun1984-idif-test_results_save.pickle")
        res = context.solver.pickle_load(pkl)
        qm, ql, qh = context.solver.quantile(results=res)

        fqfp1 = os.path.splitext(pkl)[0]
        
        # Handle both 1D and 2D cases
        if np.ndim(qm) == 1:            
            df = {"label": context.solver.labels}
            df.update({
                "qm": qm,
                "ql": ql, 
                "qh": qh
            })
            df = pd.DataFrame(df)
            df.to_csv(fqfp1 + "-quantiles.csv")
        else:
            # Create separate dataframes for qm, ql, qh
            df_qm = {"label": context.solver.labels}
            df_ql = {"label": context.solver.labels}
            df_qh = {"label": context.solver.labels}

            for i in range(qm.shape[0]):
                df_qm.update({f"qm_{i}": qm[i, :]})
                df_ql.update({f"ql_{i}": ql[i, :]})
                df_qh.update({f"qh_{i}": qh[i, :]})

            # Convert to dataframes and save to separate CSV files
            pd.DataFrame(df_qm).to_csv(fqfp1 + "-quantiles-qm.csv")
            pd.DataFrame(df_ql).to_csv(fqfp1 + "-quantiles-ql.csv") 
            pd.DataFrame(df_qh).to_csv(fqfp1 + "-quantiles-qh.csv")


if __name__ == '__main__':
    unittest.main()
