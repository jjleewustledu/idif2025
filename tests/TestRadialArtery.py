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
from copy import deepcopy
import pickle
import unittest
import os
from pprint import pprint

import matplotlib
import nibabel as nib
import numpy as np
import dynesty.utils as dyutils

from testPreliminaries import TestPreliminaries 
from RadialArteryContext import RadialArteryContext
from IOImplementations import RadialArteryIO
from RadialArteryData import RadialArteryData
from RadialArterySolver import RadialArterySolver
from DynestyPlotting import DynestyPlotting


class TestRadialArtery(TestPreliminaries):

    def setUp(self):
        petdir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "ses-20210421150523", "pet")
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "kernels")
        self.input_func_fqfn = os.path.join(petdir, "sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz")
        self.kernel_fqfn = os.path.join(kerndir, "kernel_hct=46.8.nii.gz")
        self.truths = [
            18.1405, 7.2114, 15.7225,
            3.4724, 3.8887, 2.9309, -1.4111, -1.9723, 732.1287,
            0.3583, 0.0512, 0.2570,
            2.5246,
            0.0373]
        self.data_dict = {
            "input_func_fqfn": self.input_func_fqfn,
            "kernel_fqfn": self.kernel_fqfn,
            "tag": "custom-tag-from-setUp"
        }

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_ctor(self):
        context = RadialArteryContext(self.data_dict)
        # print("\n")
        # pprint(context)
        self.assertIsInstance(context.io, RadialArteryIO)
        self.assertIsInstance(context.data, RadialArteryData)
        self.assertIsInstance(context.solver, RadialArterySolver)
        self.assertIsInstance(context.plotting, DynestyPlotting)

    def test_io(self):
        context = RadialArteryContext(self.data_dict)        
        # print("\n")
        # pprint(context.io.fqfp)
        # pprint(context.io.results_fqfp)
        niid = context.io.kernel_load(self.data_dict["kernel_fqfn"])
        self.assertEqual(niid["img"].shape, (121,))
        niid = context.io.nii_load(self.data_dict["input_func_fqfn"])
        self.assertIn(niid["fqfp"], self.input_func_fqfn)
        self.assertEqual(niid["halflife"], 122.2416)
        self.assertEqual(niid["img"].shape, (185,))
        self.assertIsInstance(niid["json"], dict)
        self.assertIsInstance(niid["nii"], nib.nifti1.Nifti1Image)
        self.assertEqual(niid["taus"].shape, (185,))
        self.assertEqual(niid["times"].shape, (185,))
        self.assertEqual(niid["timesMid"].shape, (185,))

        # context.data.print_concise(niid, "input_func NIfTI dict")
        """
        ======================= input_func NIfTI dict =======================
        {'fqfp': '/Users/jjlee/PycharmProjects/dynesty/idif2025/data/ses-20210421150523/pet/sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc',
        'halflife': 122.2416,
        'img': '<array shape=(185,)>',
        'json': "<dict keys=['Modality', 'ImagingFrequency', 'Manufacturer', "
                "'ManufacturersModelName', 'PatientPosition', 'SoftwareVersions', "
                "'SeriesDescription', 'ProtocolName', 'ImageType', 'SeriesNumber', "
                "'AcquisitionTime', 'AcquisitionNumber', 'ImageComments', "
                "'Radiopharmaceutical', 'RadionuclidePositronFraction', "
                "'RadionuclideTotalDose', 'RadionuclideHalfLife', "
                "'DoseCalibrationFactor', 'ConvolutionKernel', 'Units', "
                "'DecayCorrection', 'ReconstructionMethod', 'SliceThickness', "
                "'ImageOrientationPatientDICOM', 'ConversionSoftware', "
                "'ConversionSoftwareVersion', 'lm_start0_BMC_LM_00_ac_mc', 'starts', "
                "'taus', 'timesMid', 'times', 'timeUnit', 'datetime0', "
                "'datetimeForDecayCorrection', 'baselineActivityDensity', "
                "'TwiliteKit_do_make_input_func']>",
        'nii': '<Nifti1Image>',
        'taus': '<array shape=(185,)>',
        'times': '<array shape=(185,)>',
        'timesMid': '<array shape=(185,)>'}
        ======================================================================
        """

    def test_data(self):
        context = RadialArteryContext(self.data_dict)
        self.assertEqual(context.data.data_dict["halflife"], 122.2416)
        self.assertEqual(context.data.data_dict["input_func_fqfn"], self.input_func_fqfn)
        self.assertEqual(context.data.data_dict["kernel"].shape, (121,))
        self.assertEqual(context.data.data_dict["kernel_fqfn"], self.kernel_fqfn)
        self.assertEqual(context.data.data_dict["rho"].shape, (185,))
        self.assertEqual(context.data.data_dict["tag"], "custom-tag-from-setUp")
        self.assertEqual(context.data.data_dict["taus"].shape, (185,))
        self.assertEqual(context.data.data_dict["times"].shape, (185,))
        self.assertEqual(context.data.data_dict["timesMid"].shape, (185,))
        
        # context.data.print_concise(context.data.data_dict, "context.data_dict")
        # context.data.print_data()
        """
        ========================= context.data_dict =========================
        {'halflife': 122.2416,
        'input_func_fqfn': '/Users/jjlee/PycharmProjects/dynesty/idif2025/data/ses-20210421150523/pet/sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz',
        'kernel': '<array shape=(121,)>',
        'kernel_fqfn': '/Users/jjlee/PycharmProjects/dynesty/idif2025/data/kernels/kernel_hct=46.8.nii.gz',
        'nlive': 300,
        'rho': '<array shape=(185,)>',
        'rstate': Generator(PCG64) at 0x12F431E00,
        'sample': 'rslice',
        'sigma': 0.1,
        'tag': 'custom-tag-from-setUp',
        'taus': '<array shape=(185,)>',
        'times': '<array shape=(185,)>',
        'timesMid': '<array shape=(185,)>'}
        ======================================================================
        """

        ifm = context.data.input_func_measurement
        self.assertIn(ifm["fqfp"], self.input_func_fqfn)
        self.assertIsInstance(ifm["nii"], nib.nifti1.Nifti1Image)

    def test_solver(self):
        context = RadialArteryContext(self.data_dict)
        res = context.solver.run_nested(print_progress=True)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        qm_expected = [
            2.05423111e+01,  7.88519384e+00,  2.08209812e+00,  3.16500426e+00,
            2.88930632e+00, -1.53094174e+00, -1.94793258e+00,  3.29789352e+02,
            3.41842529e-01,  4.39920255e-02,  2.50152783e+00,  3.16138039e-02
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-2)
        
        # context.solver.results_save()
        # context.solver.results_plot()

    def test_solver_2(self):
        petdir2 = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "ses-20210421154248", "pet")
        input_func_fqfn2 = os.path.join(petdir2, "sub-108293_ses-20210421154248_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz")
        data_dict2 = {
            "input_func_fqfn": input_func_fqfn2,
            "kernel_fqfn": self.kernel_fqfn,
            "tag": "custom-tag-from-test-solver-2"
        }
        context = RadialArteryContext(data_dict2)
        res = context.solver.run_nested(print_progress=True)
        self.assertIsInstance(res, dyutils.Results)
        self.assertIs(res, context.solver.dynesty_results)
        
        qm, _, _ = context.solver.quantile(verbose=True)
        pprint(qm)
        qm_expected = [
            5.88134853e+00,  1.15479501e+01,  1.45791786e+00,  1.06962668e+01,
            2.74909526e+00, -2.09558268e-01, -2.50392975e+00,  4.01396989e+02,
            3.66289045e-01,  6.98781131e-02,  2.54331236e+00,  3.08361586e-02
        ]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-2)

    def test_pickle_dump_and_load(self):
        context = RadialArteryContext(self.data_dict)
        res = context.solver.run_nested(print_progress=True)
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

    def test_results_save(self):
        context = RadialArteryContext(self.data_dict)
        res = context.solver.run_nested(print_progress=True)
        fqfp = context.solver.results_save(tag=self._testMethodName, results=res)
        metrics = ["resid", "qm", "ql", "qh", "information", "logz", "rho-ideal", "rho-pred", "signal", "ideal"]
        suffixes = [".nii.gz", ".json"]
        for metric in metrics:
            for suffix in suffixes:
                fqfn = f"{fqfp}-{metric}{suffix}"
                self.assertTrue(os.path.exists(fqfn), f"File does not exist: {fqfn}")
                self.assertTrue(os.path.getsize(fqfn) > 0, f"File is empty: {fqfn}")
        # pprint(f"saved results to {fqfp}-*")

    def test_call(self):
        data_dict = {
            "input_func_fqfn": self.input_func_fqfn,
            "kernel_fqfn": self.kernel_fqfn,
            "nlive": 300
        }
        ra = RadialArteryContext(data_dict)
        ra()
        # self.assertTrue(os.path.exists(ra.data.results_fqfp))
        # self.assertTrue(os.path.getsize(ra.data.results_fqfp) > 0)

    def test_nii_hstack(self):
        fdgdir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", "data", "ses-20210421155709", "pet")

        twil_embed = os.path.join(fdgdir, "sub-108293_ses-20210421155709_trc-fdg_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc-embed.nii.gz")  # no deconv., decaying
        twil_deconv = os.path.join(fdgdir, "sub-108293_ses-20210421155709_trc-fdg_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc-RadialArteryIO-ideal.nii.gz")  # deconv., decaying, ~470 sec

        niid_hstack = RadialArteryData.nii_hstack(twil_deconv, twil_embed, t_crossover=300, output_format="niid")
        context = RadialArteryContext(self.data_dict)
        io = RadialArteryIO(context)
        # niid_deconv = RadialArteryIO.nii_load(twil_deconv)
        niid_embed = io.nii_load(twil_embed)

        # Plot the stacked input function
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(niid_embed["timesMid"], niid_embed["img"], 'k+', linestyle='none', markersize=6, label='Embedded Input Function')
        plt.plot(niid_hstack["timesMid"], niid_hstack["img"], 'm-', linewidth=2, label='Stacked Input Function', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Activity')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(fdgdir, "testRadialArtery_test_nii_hstack.png"))
        plt.close()
        # plt.show()


if __name__ == '__main__':
    matplotlib.use('Agg')  # disable interactive plotting
    unittest.main()
