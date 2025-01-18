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
import unittest
import os
from pprint import pprint

import nibabel as nib
from dynesty import dynesty, utils as dyutils
import numpy as np
from testPreliminaries import TestPreliminaries 
from RadialArteryContext import RadialArteryContext
from IOImplementations import RadialArteryIO
from RadialArteryData import RadialArteryData
from RadialArterySolver import RadialArterySolver
from DynestyPlotting import DynestyPlotting


class TestRadialArtery(TestPreliminaries):

    def setUp(self):
        petdir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "ses-20210421150523", "pet")
        kerndir = os.path.join(os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2024", "data", "kernels")
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
        niid = context.io.load_kernel(self.data_dict["kernel_fqfn"])
        self.assertEqual(niid["img"].shape, (121,))
        niid = context.io.load_nii(self.data_dict["input_func_fqfn"])
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
        {'fqfp': '/Users/jjlee/PycharmProjects/dynesty/idif2024/data/ses-20210421150523/pet/sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc',
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
        'input_func_fqfn': '/Users/jjlee/PycharmProjects/dynesty/idif2024/data/ses-20210421150523/pet/sub-108293_ses-20210421150523_trc-oo_proc-TwiliteKit-do-make-input-func-nomodel_inputfunc.nii.gz',
        'kernel': '<array shape=(121,)>',
        'kernel_fqfn': '/Users/jjlee/PycharmProjects/dynesty/idif2024/data/kernels/kernel_hct=46.8.nii.gz',
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
        qm_expected = [
            1.87825047e+01,  7.21877943e+00,  1.54331710e+01,  3.48161027e+00,
            3.89219304e+00,  2.93431483e+00, -1.40935949e+00, -1.94278285e+00,
            3.88042029e+02,  3.58691650e-01,  5.12421307e-02,  2.42558096e-01,
            2.47405036e+00,  3.72795087e-02]
        np.testing.assert_allclose(qm, qm_expected, rtol=1e-4)
        
        # context.solver.save_results()
        # context.solver.plot_results()

if __name__ == '__main__':
    unittest.main()
