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

from testPreliminaries import TestPreliminaries 
from IOImplementations import TissueIO
from Raichle1983Context import Raichle1983Context
from Raichle1983Data import Raichle1983Data
from Raichle1983Solver import Raichle1983Solver
from TissuePlotting import TissuePlotting


class TestRaichle1983(TestPreliminaries):

    @property
    def qm_expected_1d(self):
        return [
            0.007253, 0.603286, 0.010033, 1.447088, 0.017043
        ]
    
    @property
    def qm_expected_2d(self):
        return [
            [0.007253, 0.603286, 0.010033, 1.447088, 0.017043],
            [0.007064, 0.846275, 0.017256, 0.803208, 0.011487],
            [0.007633, 0.867064, 0.007564, 0.637374, 0.01413]
        ]

    def setUp(self):
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "kernels")
        kern = os.path.join(kerndir, "kernel_hct=46.8.nii.gz")

        hodir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", 
                             "ses-20210421152358", "pet")
        idif = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-MipIdif_idif_dynesty-"
                            "Boxcar-ideal.nii.gz")
        twil = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-TwiliteKit-do-make-"
                            "input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
        pet = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-"
                           "createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")
        self.data_dict_idif = {
            "input_func_fqfn": idif,
            "tissue_fqfn": pet,
            "tag": "playground-Raichle1983-idif"
        }
        self.data_dict_twil = {
            "kernel_fqfn": kern,
            "input_func_fqfn": twil,
            "tissue_fqfn": pet,
            "tag": "playground-Raichle1983-twil"
        }

        self.pickle_fqfn = os.path.join(
            hodir, 
            "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-"
            "TissueIO-Boxcar-playground-Raichle1983-idif-test_pickle_dump_and_load.pickle")
        self.pickle_x3_fqfn = os.path.join(
            hodir,
            "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-"
            "TissueIO-Boxcar-playground-Raichle1983-idif-test_pickle_dump_and_load_x3.pickle")
        
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_ctor(self):
        context = Raichle1983Context(self.data_dict_idif)
        # print("\n")
        # pprint(context)
        self.assertIsInstance(context.io, TissueIO)
        self.assertIsInstance(context.data, Raichle1983Data)
        self.assertIsInstance(context.solver, Raichle1983Solver)
        self.assertIsInstance(context.plotting, TissuePlotting)

        context = Raichle1983Context(self.data_dict_twil)
        # print("\n")
        # pprint(context)
        self.assertIsInstance(context.io, TissueIO)
        self.assertIsInstance(context.data, Raichle1983Data)
        self.assertIsInstance(context.solver, Raichle1983Solver)
        self.assertIsInstance(context.plotting, TissuePlotting)

    def test_io_idif(self):
        context = Raichle1983Context(self.data_dict_idif)        
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["input_func_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (119,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (119,))
        self.assertEqual(niid["times"].shape, (119,))
        self.assertEqual(niid["timesMid"].shape, (119,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_twil(self):
        context = Raichle1983Context(self.data_dict_twil)        
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_twil["input_func_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_twil["input_func_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (181,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (181,))
        self.assertEqual(niid["times"].shape, (181,))
        self.assertEqual(niid["timesMid"].shape, (181,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_io_tissue(self):
        context = Raichle1983Context(self.data_dict_idif)        
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.nii_load(self.data_dict_idif["tissue_fqfn"])
        self.assertIn(niid["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (309, 110))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (110,))
        self.assertEqual(niid["times"].shape, (110,))
        self.assertEqual(niid["timesMid"].shape, (110,))
        # context.data.print_concise(niid, "input_func NIfTI dict")

    def test_data(self):
        context = Raichle1983Context(self.data_dict_idif)
        self.assertEqual(context.data.data_dict["halflife"], 122.2416)
        self.assertEqual(context.data.data_dict["input_func_fqfn"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(context.data.data_dict["tissue_fqfn"], self.data_dict_idif["tissue_fqfn"])
        # self.assertEqual(context.data.data_dict["rho"].shape, (309,32))
        # self.assertEqual(context.data.data_dict["tag"], "")
        # self.assertEqual(context.data.data_dict["taus"].shape, (32,))
        # self.assertEqual(context.data.data_dict["times"].shape, (32,))
        # self.assertEqual(context.data.data_dict["timesMid"].shape, (32,))        
        context.data.print_concise(context.data.data_dict, "context.data_dict")
        # context.data.print_data()

        aifm = context.data.rho_input_func_measurement
        self.assertIn(aifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(aifm["img"].shape, (114,))
        self.assertAlmostEqual(np.max(aifm["img"]), 16.492276368807648, places=6)

        atm = context.data.rho_tissue_measurement
        self.assertIn(atm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(atm["img"].shape, (309, 110))
        self.assertAlmostEqual(np.max(atm["img"]), 1.0, places=6)

        ifm = context.data.input_func_measurement
        self.assertIn(ifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(ifm["img"].shape, (114,))

        self.assertEqual(context.data.input_func_type, "Boxcar")
        self.assertTrue(context.data.isidif)
        self.assertAlmostEqual(context.data.max_tissue_measurement, 54770.959083060945, places=12)
        self.assertAlmostEqual(context.data.recovery_coefficient, 1.85, places=2)
        self.assertAlmostEqual(context.data.rho.max(), 1.0, places=6)
        
        tm = context.data.tissue_measurement
        self.assertIn(tm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(tm["img"].shape, (309, 110))

    def test_solver_run_nested(self):
        context = Raichle1983Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-4)

    def test_solver_run_nested_x3(self):
        context = Raichle1983Context(self.data_dict_idif)
        ress = context.solver.run_nested(parc_index=(25, 26, 27))
        self.assertEqual(len(ress), 3)
        for res in ress:
            self.assertIsInstance(res, dyutils.Results)
            # self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-4)

    def test_quantile(self):
        if not os.path.exists(self.pickle_fqfn):
            print("\nCall test_pickle_dump_and_load() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is single
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-4)

        # check cached qm 
        # qm1, _, _ = context.solver.quantile(verbose=False)
        # np.testing.assert_allclose(qm, qm1, rtol=1e-15)

    def test_quantile_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is 
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-4)

        # check cached qm 
        # qm1, _, _ = context.solver.quantile(verbose=False)
        # np.testing.assert_allclose(qm, qm1, rtol=1e-15)

    def test_package_results(self):
        if not os.path.exists(self.pickle_fqfn):
            print("\nCall test_pickle_dump_and_load() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")            
        
        # pickle of results is single
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check package from pickle
        pkg = context.solver.package_results(results=a_pickle, parc_index=25)
        pprint(pkg)
        np.testing.assert_allclose(pkg["information"], 15.671947264439893, rtol=1e-12)
        np.testing.assert_allclose(pkg["logz"], 277.60599628494566, rtol=1e-12)
        np.testing.assert_allclose(pkg["qm"], self.qm_expected_1d, rtol=1e-4)

    def test_package_results_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")            
        
        # pickle of results is list
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check package from pickle
        pkg = context.solver.package_results(results=a_pickle, parc_index=(25, 26, 27))
        pprint(pkg)
        np.testing.assert_allclose(
            pkg["information"], [15.67194726, 17.53458389, 16.07342179], rtol=1e-6)
        np.testing.assert_allclose(
            pkg["logz"], [277.60599628, 319.15701913, 297.06527284], rtol=1e-6)
        np.testing.assert_allclose(pkg["qm"], self.qm_expected_2d, rtol=1e-4)

    def test_pickle_dump_and_load(self):
        context = Raichle1983Context(self.data_dict_idif)
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
        context = Raichle1983Context(self.data_dict_idif)
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
        context = Raichle1983Context(self.data_dict_idif)
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
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_fqfn)
        self.assertIsInstance(a_pickle, dyutils.Results)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_1d, rtol=1e-4)

    def test_truths_x3(self):
        if not os.path.exists(self.pickle_x3_fqfn):
            print("\nCall test_pickle_dump_and_load_x3() first to generate a pickle for testing.")
            self.skipTest("Pickle not found")
        
        # pickle of results is single
        context = Raichle1983Context(self.data_dict_idif)
        a_pickle = context.solver.pickle_load(self.pickle_x3_fqfn)
        self.assertIsInstance(a_pickle, list)

        # check quantile from pickle
        qm, _, _ = context.solver.quantile(results=a_pickle, verbose=False)
        pprint(qm)
        np.testing.assert_allclose(qm, self.qm_expected_2d, rtol=1e-4)

    def test_call(self):
        data_dict = self.data_dict_idif
        data_dict["tag"] = ""
        r = Raichle1983Context(data_dict)
        r()
        # self.assertTrue(os.path.exists(ra.data.results_fqfp))
        # self.assertTrue(os.path.getsize(ra.data.results_fqfp) > 0)


if __name__ == '__main__':
    unittest.main()
