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
import sys
from pprint import pprint

import numpy as np
from tests.testPreliminaries import TestPreliminaries
from IOImplementations import TrivialArteryIO


class TestIO(TestPreliminaries):

    def setUp(self):
        self.io = TrivialArteryIO()

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_IO_ctor(self):
        io = TrivialArteryIO()
        self.assertIsInstance(io, TrivialArteryIO)

    def test_nii_load(self):
        fqfn = self.fqfn_ParcSchaeffer("oo1", singularity=False)
        data = self.io.nii_load(fqfn=fqfn)
        # pprint(data)

        # Check JSON size
        json_size = sys.getsizeof(data["json"])
        self.assertLess(json_size, 10000)

        # Check that temporal dimensions match
        n_times = len(data['times'])
        n_taus = len(data['taus']) 
        n_timesMid = len(data['timesMid'])
        self.assertEqual(n_times, n_taus)
        self.assertEqual(n_times, n_timesMid)

        # Check image shape matches temporal dimension
        img_shape = data['img'].shape
        self.assertEqual(len(img_shape), 2)
        self.assertEqual(img_shape[1], n_times)

        # Verify shape matches NIfTI header
        nii_shape = data['nii'].shape
        self.assertEqual(img_shape[0], nii_shape[0])
        self.assertEqual(img_shape[1], nii_shape[1])

    def test_nii_save(self):
        fqfn1 = os.path.join(self.petdir("oo1", singularity=False), "test_save_trc-oo_nii.nii.gz")

        fqfn = self.fqfn_ParcSchaeffer("oo1", singularity=False)
        data = self.io.nii_load(fqfn=fqfn)
        self.io.nii_save(data=data, fqfn=fqfn1)

        # Load the saved data
        data1 = self.io.nii_load(fqfn=fqfn1)
        
        # Compare key components of data dictionaries
        for key in ['times', 'taus', 'timesMid']:
            self.assertTrue(np.array_equal(data[key], data1[key]))
            
        # Compare JSON contents recursively
        self.assertDictEqual(data['json'], data1['json'])
        
        # Compare numerical arrays and metadata
        self.assertTrue(np.array_equal(data['img'], data1['img']))
        self.assertEqual(data['nii'].shape, data1['nii'].shape)
        self.assertEqual(data['nii'].header, data1['nii'].header)
