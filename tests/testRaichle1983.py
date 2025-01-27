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

    def setUp(self):
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "kernels")
        kern = os.path.join(kerndir, "kernel_hct=46.8.nii.gz")

        hodir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421152358", "pet")
        idif = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-MipIdif_idif_dynesty-Boxcar-ideal.nii.gz")
        twil = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc_dynesty-RadialArtery-ideal.nii.gz")
        pet = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-ParcSchaeffer-reshape-to-schaeffer-schaeffer.nii.gz")
        
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

        self.pickle_fqfn = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-Boxcar-playground-Raichle1983-idif-test_pickle_dump_and_load.pickle")
        self.pickle_x3_fqfn = os.path.join(hodir, "sub-108293_ses-20210421152358_trc-ho_proc-delay0-BrainMoCo2-createNiftiMovingAvgFrames-schaeffer-TissueIO-Boxcar-playground-Raichle1983-idif-test_pickle_dump_and_load_x3.pickle")
        
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
        self.assertEqual(niid["img"].shape, (309,110))
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
        self.assertEqual(aifm["img"].shape, (119,))
        self.assertAlmostEqual(np.max(aifm["img"]), 16.494463800372166, places=6)

        atm = context.data.rho_tissue_measurement
        self.assertIn(atm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(atm["img"].shape, (309,110))
        self.assertAlmostEqual(np.max(atm["img"]), 1.0, places=6)

        ifm = context.data.input_func_measurement
        self.assertIn(ifm["fqfp"], self.data_dict_idif["input_func_fqfn"])
        self.assertEqual(ifm["img"].shape, (119,))

        self.assertEqual(context.data.input_func_type, "Boxcar")
        self.assertTrue(context.data.isidif)
        self.assertAlmostEqual(context.data.max_tissue_measurement, 54763.62220587669, places=12)
        self.assertAlmostEqual(context.data.recovery_coefficient, 1.85, places=2)
        self.assertAlmostEqual(context.data.rho.max(), 1.0, places=6)
        
        tm = context.data.tissue_measurement
        self.assertIn(tm["fqfp"], self.data_dict_idif["tissue_fqfn"])
        self.assertEqual(tm["img"].shape, (309,110))

    def test_solver_run_nested(self):
        context = Raichle1983Context(self.data_dict_idif)
        res = context.solver.run_nested(print_progress=True, parc_index=25)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        qm_expected = [
            6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01,
            1.69328318e-02
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-4)

    def test_solver_run_nested_x3(self):
        context = Raichle1983Context(self.data_dict_idif)
        ress = context.solver.run_nested(parc_index=(25, 26, 27))
        self.assertEqual(len(ress), 3)
        for res in ress:
            self.assertIsInstance(res, dyutils.Results)
            # self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        qm_expected = [
            [6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01, 1.69328318e-02],
            [8.07996120e-03, 8.45594317e-01, 1.29251607e-02, 8.09855115e-01, 1.65465307e+01, 1.13907595e-02],
            [5.75432912e-03, 8.64451631e-01, 1.04023251e-02, 6.36434022e-01, 1.49533549e+01, 1.41159316e-02]
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-4)

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
        qm_expected = [
            6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01,
            1.69328318e-02
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-6)

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
        qm_expected = [
            [6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01, 1.69328318e-02],
            [8.07996120e-03, 8.45594317e-01, 1.29251607e-02, 8.09855115e-01, 1.65465307e+01, 1.13907595e-02],
            [5.75432912e-03, 8.64451631e-01, 1.04023251e-02, 6.36434022e-01, 1.49533549e+01, 1.41159316e-02]
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-6)

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
        np.testing.assert_allclose(pkg["information"], 16.020622070519494, rtol=1e-12)
        np.testing.assert_allclose(pkg["logz"], 277.29974714861606, rtol=1e-12)
        np.testing.assert_allclose(pkg["qm"], [
            6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01,
            1.69328318e-02
        ], rtol=1e-6)
        np.testing.assert_allclose(pkg["rho_pred"], [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00108577, 0.00575123, 0.01631941, 0.0342898,
            0.06053261, 0.09526701, 0.13805375, 0.18787114, 0.24329212, 0.30298809, 0.36469717,
            0.42487343, 0.48041834, 0.52932811, 0.57042904, 0.60331691, 0.62830203, 0.64629914,
            0.65863769, 0.66655662, 0.67115237, 0.67340949, 0.6740489, 0.67353298, 0.67213464,
            0.67001796, 0.66735513, 0.66431052, 0.66100656, 0.65754514, 0.65400927, 0.65046352,
            0.64695511, 0.64351585, 0.64016474, 0.63691071, 0.63369885, 0.63046094, 0.62716261,
            0.6237836, 0.62031395, 0.61675151, 0.61309992, 0.60936686, 0.60556263, 0.6016989,
            0.5977878, 0.59384123, 0.58987035, 0.58588532, 0.58189509, 0.57790739, 0.57392872,
            0.56996444, 0.56601889, 0.56209552, 0.55819699, 0.5543253, 0.55048193, 0.54666793,
            0.54288399, 0.53913053, 0.53540777, 0.53171575, 0.52805443, 0.52442364, 0.52082317,
            0.51725277, 0.51371214, 0.51020097, 0.50671896, 0.50326578, 0.49984111, 0.49644462,
            0.493076, 0.48973493, 0.4864211, 0.48313421, 0.47987396, 0.47664007, 0.47343223,
            0.47025019, 0.46709364, 0.46396234, 0.46085601, 0.4577744, 0.45471724, 0.45168428,
            0.44867528, 0.44569, 0.44272819, 0.43978963, 0.43687406, 0.43398128, 0.43111105,
            0.42826315, 0.42543736, 0.42263346, 0.41985125, 0.4170905, 0.41435102, 0.4116326,
            0.40893503, 0.40625812
        ], rtol=1e-4)
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
            pkg["information"], [16.02062207, 18.04949316, 15.90698274], rtol=1e-6)
        np.testing.assert_allclose(
            pkg["logz"], [277.29974715, 318.78296149, 297.37324705], rtol=1e-6)
        np.testing.assert_allclose(pkg["qm"], [
            [6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01, 1.69328318e-02],
            [8.07996120e-03, 8.45594317e-01, 1.29251607e-02, 8.09855115e-01, 1.65465307e+01, 1.13907595e-02],
            [5.75432912e-03, 8.64451631e-01, 1.04023251e-02, 6.36434022e-01, 1.49533549e+01, 1.41159316e-02]
        ], rtol=1e-6)
        # pprint(pkg["rho_pred"])
        np.testing.assert_allclose(pkg["rho_pred"], [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00108577, 0.00575123, 0.01631941, 0.0342898,
             0.06053261, 0.09526701, 0.13805375, 0.18787114, 0.24329212, 0.30298809, 0.36469717,
             0.42487343, 0.48041834, 0.52932811, 0.57042904, 0.60331691, 0.62830203, 0.64629914,
             0.65863769, 0.66655662, 0.67115237, 0.67340949, 0.6740489, 0.67353298, 0.67213464,
             0.67001796, 0.66735513, 0.66431052, 0.66100656, 0.65754514, 0.65400927, 0.65046352,
             0.64695511, 0.64351585, 0.64016474, 0.63691071, 0.63369885, 0.63046094, 0.62716261,
             0.6237836, 0.62031395, 0.61675151, 0.61309992, 0.60936686, 0.60556263, 0.6016989,
             0.5977878, 0.59384123, 0.58987035, 0.58588532, 0.58189509, 0.57790739, 0.57392872,
             0.56996444, 0.56601889, 0.56209552, 0.55819699, 0.5543253, 0.55048193, 0.54666793,
             0.54288399, 0.53913053, 0.53540777, 0.53171575, 0.52805443, 0.52442364, 0.52082317,
             0.51725277, 0.51371214, 0.51020097, 0.50671896, 0.50326578, 0.49984111, 0.49644462,
             0.493076, 0.48973493, 0.4864211, 0.48313421, 0.47987396, 0.47664007, 0.47343223,
             0.47025019, 0.46709364, 0.46396234, 0.46085601, 0.4577744, 0.45471724, 0.45168428,
             0.44867528, 0.44569, 0.44272819, 0.43978963, 0.43687406, 0.43398128, 0.43111105,
             0.42826315, 0.42543736, 0.42263346, 0.41985125, 0.4170905, 0.41435102, 0.4116326,
             0.40893503, 0.40625812],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.00043085, 0.00379303, 0.01334261, 0.03115729, 0.05856744,
             0.09614861, 0.14368563, 0.20021548, 0.26419188, 0.33385884, 0.40741032, 0.48098681,
             0.55027703, 0.61249119, 0.66586405, 0.70956127, 0.74362818, 0.76889395, 0.78678444,
             0.79894377, 0.80664591, 0.81116227, 0.8134756, 0.81424092, 0.81385009, 0.81253121,
             0.81045589, 0.8078494, 0.80486721, 0.80163884, 0.79827147, 0.79485022, 0.79143899,
             0.7880824, 0.78480862, 0.78163264, 0.77853709, 0.77541575, 0.77222022, 0.76892109,
             0.76550249, 0.76195881, 0.75829217, 0.75451023, 0.75062439, 0.74664818, 0.74259607,
             0.73848254, 0.73432141, 0.7301254, 0.72590588, 0.72167277, 0.71743451, 0.71319818,
             0.70896958, 0.70475336, 0.70055322, 0.69637204, 0.69221199, 0.68807469, 0.68396132,
             0.67987268, 0.67580931, 0.67177154, 0.66775954, 0.66377334, 0.65981291, 0.65587814,
             0.65196888, 0.64808495, 0.64422617, 0.64039233, 0.6365832, 0.63279858, 0.62903825,
             0.625302, 0.62158961, 0.6179009, 0.61423565, 0.61059366, 0.60697475, 0.60337873,
             0.59980542, 0.59625462, 0.59272618, 0.58921991, 0.58573564, 0.58227321, 0.57883246,
             0.57541322, 0.57201533, 0.56863864, 0.565283, 0.56194824, 0.55863423, 0.55534081,
             0.55206785, 0.54881518, 0.54558267, 0.54237019, 0.53917758, 0.53600472, 0.53285146,
             0.52971767, 0.52660322],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0006152, 0.00387384, 0.01204714, 0.02659970, 0.04844125,
             0.07791313, 0.11476927, 0.15822283, 0.20708338, 0.26011185, 0.31566704, 0.37074867,
             0.4223464, 0.46850372, 0.50801172, 0.54034859, 0.56563832, 0.58456892, 0.59824096,
             0.60781014, 0.61422727, 0.61839946, 0.62101222, 0.62251503, 0.62317491, 0.62314903,
             0.6225753, 0.62161069, 0.62036724, 0.61893838, 0.61740114, 0.61581627, 0.61422905,
             0.61267078, 0.611161, 0.60970984, 0.6082884, 0.60682626, 0.60528998, 0.60365951,
             0.60192436, 0.60008139, 0.59813291, 0.59608515, 0.59394692, 0.59172848, 0.58944068,
             0.58709432, 0.58469965, 0.58226608, 0.57980204, 0.57731486, 0.57481081, 0.57229516,
             0.56977223, 0.56724555, 0.56471795, 0.56219164, 0.55966835, 0.5571494, 0.55463582,
             0.55212833, 0.54962749, 0.5471337, 0.54464723, 0.54216829, 0.539697, 0.53723344,
             0.53477767, 0.53232972, 0.5298896, 0.52745732, 0.52503287, 0.52261625, 0.52020746,
             0.51780649, 0.51541332, 0.51302797, 0.51065042, 0.50828067, 0.50591873, 0.50356458,
             0.50121824, 0.4987897, 0.49654897, 0.49422606, 0.49191096, 0.48960369, 0.48730424,
             0.48501263, 0.48272886, 0.48045293, 0.47818486, 0.47592465, 0.47367231, 0.47142784,
             0.46919124, 0.46696253, 0.46474171, 0.46252879, 0.46032376, 0.45812664, 0.45593743,
             0.45375613, 0.45158275]
        ], rtol=1e-3)

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
        qm_expected = [
            6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01,
            1.69328318e-02
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-6)

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
        qm_expected = [
            [6.21782297e-03, 6.04999864e-01, 1.29103447e-02, 1.42984121e+00, 1.52333973e+01, 1.69328318e-02],
            [8.07996120e-03, 8.45594317e-01, 1.29251607e-02, 8.09855115e-01, 1.65465307e+01, 1.13907595e-02],
            [5.75432912e-03, 8.64451631e-01, 1.04023251e-02, 6.36434022e-01, 1.49533549e+01, 1.41159316e-02]
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-6)

    def test_call(self):
        data_dict = self.data_dict_idif
        data_dict["tag"] = ""
        r = Raichle1983Context(data_dict)
        r()
        # self.assertTrue(os.path.exists(ra.data.results_fqfp))
        # self.assertTrue(os.path.getsize(ra.data.results_fqfp) > 0)


if __name__ == '__main__':
    unittest.main()
