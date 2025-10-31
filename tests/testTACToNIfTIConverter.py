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
import json
import numpy as np
import nibabel as nib
from pprint import pprint

from testPreliminaries import TestPreliminaries
from TACToNIfTIConverter import PTACToNIfTIConverter, TTACToNIfTIConverter


class TestTACToNIfTIConverter(TestPreliminaries):

    def setUp(self):
        """Set up test data paths."""
        self.data_dir = os.path.join(
            os.getenv("HOME"), "PycharmProjects", "dynesty", "idif2025", 
            "data", "sub-EVA118"
        )
        self.ptac_csv = os.path.join(
            self.data_dir, "sub-EVA118_ses-retest_metab_corr_ptac.csv"
        )
        self.ttac_csv = os.path.join(
            self.data_dir, "sub-EVA118_ses-retest_tacs.csv"
        )
        
        # Initialize converters
        self.ptac_converter = PTACToNIfTIConverter()
        self.ttac_converter = TTACToNIfTIConverter()
        
        # Output prefix for generated files
        self.test_output_prefix = os.path.join(
            self.data_dir, "test_output"
        )

    def test_something(self):
        self.assertEqual(True, True)

    # ==================== PTACToNIfTIConverter Tests ====================

    def test_ptac_read_csv(self):
        """Test reading pTAC CSV file with mc_pTAC column."""
        df = self.ptac_converter.read_csv(self.ptac_csv)
        
        # Check that DataFrame is not empty
        self.assertGreater(len(df), 0)
        
        # Check that pTAC column exists (should be renamed from mc_pTAC)
        self.assertIn('pTAC', df.columns)
        
        # Check required columns
        required_cols = ['DateTime', 'Min', 'Sec', 'pTAC']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(np.issubdtype(df['Sec'].dtype, np.number))
        self.assertTrue(np.issubdtype(df['pTAC'].dtype, np.number))
        
        print("\n")
        print(f"pTAC CSV shape: {df.shape}")
        print(f"pTAC columns: {list(df.columns)}")
        print(f"pTAC first 5 rows:\n{df.head()}")

    def test_ptac_extract_data(self):
        """Test extracting pTAC data and verifying shape (1, N_t)."""
        df = self.ptac_converter.read_csv(self.ptac_csv)
        tac_data, n_regions, n_t = self.ptac_converter.extract_tac_data(df)
        
        # Check shape is (1, N_t)
        self.assertEqual(n_regions, 1)
        self.assertEqual(tac_data.shape, (1, n_t))
        self.assertEqual(n_t, len(df))
        
        # Check data type
        self.assertEqual(tac_data.dtype, np.float32)
        
        # Check that data is non-negative (activity values)
        self.assertTrue(np.all(tac_data >= 0))
        
        print("\n")
        print(f"pTAC data shape: {tac_data.shape}")
        print(f"n_regions: {n_regions}, n_t: {n_t}")
        print(f"Data range: [{np.min(tac_data):.4f}, {np.max(tac_data):.4f}]")

    def test_ptac_timing_arrays(self):
        """Test calculation of timing arrays for pTAC."""
        df = self.ptac_converter.read_csv(self.ptac_csv)
        n_t = len(df)
        time_values = df['Sec'].values
        
        taus, times, timesMid = self.ptac_converter.calculate_timing_arrays(n_t, time_values)
        
        # Check array shapes
        self.assertEqual(len(taus), n_t)
        self.assertEqual(len(times), n_t)
        self.assertEqual(len(timesMid), n_t)
        
        # Check timesMid matches input Sec values
        np.testing.assert_array_almost_equal(timesMid, time_values)
        
        # Check first tau is 1.0
        self.assertAlmostEqual(taus[0], 1.0)
        
        # Check taus are either 1.0 or 10.0 (based on sampling intervals)
        unique_taus = np.unique(taus)
        for tau in unique_taus:
            self.assertIn(tau, [1.0, 10.0])
        
        # Check times = timesMid - taus/2
        np.testing.assert_array_almost_equal(times, timesMid - taus / 2.0)
        
        print("\n")
        print(f"Taus unique values: {np.unique(taus)}")
        print(f"Taus first 10: {taus[:10]}")
        print(f"Times first 10: {times[:10]}")
        print(f"TimesMid first 10: {timesMid[:10]}")

    def test_ptac_process_csv_to_nifti(self):
        """Test end-to-end pTAC CSV to NIfTI conversion."""
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv, 
            output_prefix=self.test_output_prefix + "_ptac"
        )
        
        # Check files were created
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertGreater(os.path.getsize(nifti_path), 0)
        self.assertGreater(os.path.getsize(json_path), 0)
        
        print("\n")
        print(f"Created NIfTI file: {nifti_path}")
        print(f"Created JSON file: {json_path}")

    def test_ptac_validate_nifti(self):
        """Test validation of generated pTAC NIfTI file."""
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=self.test_output_prefix + "_ptac"
        )
        
        # Validate NIfTI structure
        validation_info = self.ptac_converter.validate_nifti_file(nifti_path)
        
        # Check validation results
        self.assertEqual(validation_info['n_regions'], 1)
        self.assertGreater(validation_info['n_timepoints'], 0)
        self.assertEqual(validation_info['data_type'], np.float32)
        
        # Check dimensions
        self.assertEqual(validation_info['data_shape'][0], 1)
        
        # Load and check NIfTI header details
        img = nib.load(nifti_path)
        header = img.header
        
        # Check dim array
        self.assertEqual(header['dim'][0], 2)  # Number of dimensions
        self.assertEqual(header['dim'][1], 1)  # n_regions
        self.assertGreater(header['dim'][2], 0)  # n_timepoints
        
        # Check units
        xyz_units, t_units = header.get_xyzt_units()
        self.assertEqual(xyz_units, 'mm')
        self.assertEqual(t_units, 'sec')
        
        print("\n")
        print("pTAC NIfTI validation results:")
        pprint(validation_info)

    # ==================== TTACToNIfTIConverter Tests ====================

    def test_ttac_read_csv(self):
        """Test reading tTAC CSV file with FrameReferenceTime handling."""
        df = self.ttac_converter.read_csv(self.ttac_csv)
        
        # Check that DataFrame is not empty
        self.assertGreater(len(df), 0)
        
        # Check that MidFrameTime_sec_ column exists
        self.assertIn('MidFrameTime_sec_', df.columns)
        
        # Check that FrameReferenceTime was removed if it existed
        self.assertNotIn('FrameReferenceTime', df.columns)
        
        # Check no empty column names
        for col in df.columns:
            self.assertNotEqual(col, '')
        
        print("\n")
        print(f"tTAC CSV shape: {df.shape}")
        print(f"tTAC number of columns: {len(df.columns)}")
        print(f"tTAC first 3 columns: {list(df.columns[:3])}")

    def test_ttac_anatomical_columns(self):
        """Test extraction of anatomical region labels."""
        df = self.ttac_converter.read_csv(self.ttac_csv)
        anatomical_cols = self.ttac_converter.get_anatomical_columns(df)
        
        # Check that we have anatomical columns
        self.assertGreater(len(anatomical_cols), 0)
        
        # Check that MidFrameTime_sec_ is not in anatomical columns
        self.assertNotIn('MidFrameTime_sec_', anatomical_cols)
        
        # Check all anatomical columns exist in DataFrame
        for col in anatomical_cols:
            self.assertIn(col, df.columns)
        
        print("\n")
        print(f"Number of anatomical regions: {len(anatomical_cols)}")
        print(f"First 10 anatomical regions: {anatomical_cols[:10]}")
        print(f"Last 5 anatomical regions: {anatomical_cols[-5:]}")

    def test_ttac_extract_data(self):
        """Test extracting tTAC data and verifying shape (n_anatomical, N_t)."""
        df = self.ttac_converter.read_csv(self.ttac_csv)
        tac_data, n_anatomical, n_t = self.ttac_converter.extract_tac_data(df)
        
        # Check shape
        self.assertEqual(tac_data.shape, (n_anatomical, n_t))
        self.assertEqual(n_t, len(df))
        self.assertGreater(n_anatomical, 1)  # Should have multiple regions
        
        # Check data type
        self.assertEqual(tac_data.dtype, np.float32)
        
        # Check that data is non-negative (activity values)
        self.assertTrue(np.all(tac_data >= 0))
        
        print("\n")
        print(f"tTAC data shape: {tac_data.shape}")
        print(f"n_anatomical: {n_anatomical}, n_t: {n_t}")
        print(f"Data range: [{np.min(tac_data):.4f}, {np.max(tac_data):.4f}]")

    def test_ttac_timing_arrays(self):
        """Test calculation of timing arrays with specific tau patterns."""
        df = self.ttac_converter.read_csv(self.ttac_csv)
        n_t = len(df)
        time_values = df['MidFrameTime_sec_'].values if 'MidFrameTime_sec_' in df.columns else None
        
        taus, times, timesMid = self.ttac_converter.calculate_timing_arrays(n_t, time_values)
        
        # Check array shapes
        self.assertEqual(len(taus), n_t)
        self.assertEqual(len(times), n_t)
        self.assertEqual(len(timesMid), n_t)
        
        # Check tau pattern: first 36 should be 5.0, next 12 should be 10.0, rest 300.0
        if n_t >= 36:
            self.assertTrue(np.all(taus[:36] == 5.0))
        if n_t >= 48:
            self.assertTrue(np.all(taus[36:48] == 10.0))
        if n_t > 48:
            self.assertTrue(np.all(taus[48:] == 300.0))
        
        # Check times = timesMid - taus/2
        np.testing.assert_array_almost_equal(times, timesMid - taus / 2.0)
        
        # Check unique tau values
        unique_taus = np.unique(taus)
        for tau in unique_taus:
            self.assertIn(tau, [5.0, 10.0, 300.0])
        
        print("\n")
        print(f"Taus unique values: {np.unique(taus)}")
        print(f"Taus pattern: {taus[:50] if n_t >= 50 else taus}")
        print(f"First 5 timesMid: {timesMid[:5]}")
        print(f"First 5 times: {times[:5]}")

    def test_ttac_process_csv_to_nifti(self):
        """Test end-to-end tTAC CSV to NIfTI conversion."""
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=self.test_output_prefix + "_ttac",
            include_region_labels=True
        )
        
        # Check files were created
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertGreater(os.path.getsize(nifti_path), 0)
        self.assertGreater(os.path.getsize(json_path), 0)
        
        print("\n")
        print(f"Created NIfTI file: {nifti_path}")
        print(f"Created JSON file: {json_path}")

    def test_ttac_validate_nifti(self):
        """Test validation of generated tTAC NIfTI file with multiple regions."""
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=self.test_output_prefix + "_ttac",
            include_region_labels=True
        )
        
        # Validate NIfTI structure
        validation_info = self.ttac_converter.validate_nifti_file(nifti_path)
        
        # Check validation results
        self.assertGreater(validation_info['n_regions'], 1)  # Multiple regions
        self.assertGreater(validation_info['n_timepoints'], 0)
        self.assertEqual(validation_info['data_type'], np.float32)
        
        # Check dimensions
        self.assertGreater(validation_info['data_shape'][0], 1)
        
        # Load and check NIfTI header details
        img = nib.load(nifti_path)
        header = img.header
        
        # Check dim array
        self.assertEqual(header['dim'][0], 2)  # Number of dimensions
        self.assertGreater(header['dim'][1], 1)  # n_regions
        self.assertGreater(header['dim'][2], 0)  # n_timepoints
        
        # Check units
        xyz_units, t_units = header.get_xyzt_units()
        self.assertEqual(xyz_units, 'mm')
        self.assertEqual(t_units, 'sec')
        
        print("\n")
        print("tTAC NIfTI validation results:")
        pprint(validation_info)

    # ==================== JSON Sidecar Tests ====================

    def test_json_sidecar_structure_ptac(self):
        """Test JSON sidecar structure for pTAC."""
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=self.test_output_prefix + "_ptac"
        )
        
        # Load JSON file
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Check required fields
        required_fields = ['taus', 'times', 'timesMid', 'timeUnit']
        for field in required_fields:
            self.assertIn(field, json_data)
        
        # Check timeUnit
        self.assertEqual(json_data['timeUnit'], 'second')
        
        # Check array structures (should be N_t x 1 format)
        self.assertIsInstance(json_data['taus'], list)
        self.assertIsInstance(json_data['times'], list)
        self.assertIsInstance(json_data['timesMid'], list)
        
        # Check all arrays have same length
        n_t = len(json_data['taus'])
        self.assertEqual(len(json_data['times']), n_t)
        self.assertEqual(len(json_data['timesMid']), n_t)
        
        # Check array format (nested lists for column vector)
        for i in range(min(3, n_t)):
            self.assertIsInstance(json_data['taus'][i], list)
            self.assertEqual(len(json_data['taus'][i]), 1)
        
        print("\n")
        print(f"pTAC JSON fields: {list(json_data.keys())}")
        print(f"Number of time points: {n_t}")
        print(f"First 3 taus: {json_data['taus'][:3]}")

    def test_json_sidecar_structure_ttac(self):
        """Test JSON sidecar structure for tTAC with anatomical regions."""
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=self.test_output_prefix + "_ttac",
            include_region_labels=True
        )
        
        # Load JSON file
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Check required fields
        required_fields = ['taus', 'times', 'timesMid', 'timeUnit', 'anatomicalRegions']
        for field in required_fields:
            self.assertIn(field, json_data)
        
        # Check timeUnit
        self.assertEqual(json_data['timeUnit'], 'second')
        
        # Check anatomical regions
        self.assertIsInstance(json_data['anatomicalRegions'], list)
        self.assertGreater(len(json_data['anatomicalRegions']), 1)
        
        # Check array structures
        self.assertIsInstance(json_data['taus'], list)
        self.assertIsInstance(json_data['times'], list)
        self.assertIsInstance(json_data['timesMid'], list)
        
        # Check all timing arrays have same length
        n_t = len(json_data['taus'])
        self.assertEqual(len(json_data['times']), n_t)
        self.assertEqual(len(json_data['timesMid']), n_t)
        
        print("\n")
        print(f"tTAC JSON fields: {list(json_data.keys())}")
        print(f"Number of anatomical regions: {len(json_data['anatomicalRegions'])}")
        print(f"Number of time points: {n_t}")
        print(f"First 5 anatomical regions: {json_data['anatomicalRegions'][:5]}")

    def test_json_timing_arrays(self):
        """Test that JSON timing arrays match expected calculations."""
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=self.test_output_prefix + "_ptac"
        )
        
        # Load JSON file
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Extract timing arrays (flatten from column vector format)
        taus = np.array([val[0] for val in json_data['taus']])
        times = np.array([val[0] for val in json_data['times']])
        timesMid = np.array([val[0] for val in json_data['timesMid']])
        
        # Check relationship: times = timesMid - taus/2
        np.testing.assert_array_almost_equal(times, timesMid - taus / 2.0, decimal=5)
        
        # Check that taus and timesMid are positive
        self.assertTrue(np.all(taus > 0))
        self.assertTrue(np.all(timesMid > 0))
        
        # times can be slightly negative for first sample (timesMid - taus/2)
        # but should be reasonable values
        self.assertTrue(times[0] > -10.0)  # Allow reasonable negative start time
        
        # Check that timesMid is monotonically increasing
        self.assertTrue(np.all(np.diff(timesMid) > 0))
        
        print("\n")
        print("Timing arrays relationship verified: times = timesMid - taus/2")
        print(f"Taus range: [{np.min(taus):.2f}, {np.max(taus):.2f}]")
        print(f"Times range: [{np.min(times):.2f}, {np.max(times):.2f}]")
        print(f"TimesMid range: [{np.min(timesMid):.2f}, {np.max(timesMid):.2f}]")

    # ==================== File Extension Tests ====================

    def test_ptac_nii_format(self):
        """Test pTAC conversion with .nii format."""
        output_prefix = os.path.join(self.data_dir, "test_ptac_nii.nii")
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=output_prefix
        )
        
        # Verify correct file extensions
        self.assertTrue(nifti_path.endswith('.nii'))
        self.assertFalse(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        self.assertFalse(json_path.endswith('.nii.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Verify can load with nibabel
        img = nib.load(nifti_path)
        self.assertEqual(img.shape[0], 1)
        self.assertGreater(img.shape[1], 0)
        
        # Verify JSON is valid
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertIn('taus', json_data)
        
        print("\n")
        print(f"Created .nii format: {nifti_path}")
        print(f"JSON sidecar: {json_path}")

    def test_ptac_niigz_format(self):
        """Test pTAC conversion with .nii.gz format."""
        output_prefix = os.path.join(self.data_dir, "test_ptac_niigz.nii.gz")
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=output_prefix
        )
        
        # Verify correct file extensions
        self.assertTrue(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        self.assertFalse(json_path.endswith('.nii.json'))
        self.assertFalse(json_path.endswith('.nii.gz.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Verify can load with nibabel
        img = nib.load(nifti_path)
        self.assertEqual(img.shape[0], 1)
        self.assertGreater(img.shape[1], 0)
        
        # Verify JSON is valid
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertIn('taus', json_data)
        
        print("\n")
        print(f"Created .nii.gz format: {nifti_path}")
        print(f"JSON sidecar: {json_path}")

    def test_ptac_default_format(self):
        """Test pTAC conversion uses .nii.gz by default."""
        output_prefix = os.path.join(self.data_dir, "test_ptac_default")
        nifti_path, json_path = self.ptac_converter.process_csv_to_nifti(
            self.ptac_csv,
            output_prefix=output_prefix
        )
        
        # Verify default is .nii.gz (codebase standard)
        self.assertTrue(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        print("\n")
        print(f"Default format is .nii.gz: {nifti_path}")

    def test_ttac_nii_format(self):
        """Test tTAC conversion with .nii format."""
        output_prefix = os.path.join(self.data_dir, "test_ttac_nii.nii")
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=output_prefix,
            include_region_labels=True
        )
        
        # Verify correct file extensions
        self.assertTrue(nifti_path.endswith('.nii'))
        self.assertFalse(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        self.assertFalse(json_path.endswith('.nii.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Verify can load with nibabel
        img = nib.load(nifti_path)
        self.assertGreater(img.shape[0], 1)  # Multiple regions
        self.assertGreater(img.shape[1], 0)
        
        # Verify JSON is valid with anatomical regions
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertIn('taus', json_data)
        self.assertIn('anatomicalRegions', json_data)
        
        print("\n")
        print(f"Created .nii format: {nifti_path}")
        print(f"JSON sidecar: {json_path}")

    def test_ttac_niigz_format(self):
        """Test tTAC conversion with .nii.gz format."""
        output_prefix = os.path.join(self.data_dir, "test_ttac_niigz.nii.gz")
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=output_prefix,
            include_region_labels=True
        )
        
        # Verify correct file extensions
        self.assertTrue(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        self.assertFalse(json_path.endswith('.nii.json'))
        self.assertFalse(json_path.endswith('.nii.gz.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Verify can load with nibabel
        img = nib.load(nifti_path)
        self.assertGreater(img.shape[0], 1)  # Multiple regions
        self.assertGreater(img.shape[1], 0)
        
        # Verify JSON is valid with anatomical regions
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertIn('taus', json_data)
        self.assertIn('anatomicalRegions', json_data)
        
        print("\n")
        print(f"Created .nii.gz format: {nifti_path}")
        print(f"JSON sidecar: {json_path}")

    def test_ttac_default_format(self):
        """Test tTAC conversion uses .nii.gz by default."""
        output_prefix = os.path.join(self.data_dir, "test_ttac_default")
        nifti_path, json_path = self.ttac_converter.process_csv_to_nifti(
            self.ttac_csv,
            output_prefix=output_prefix,
            include_region_labels=True
        )
        
        # Verify default is .nii.gz (codebase standard)
        self.assertTrue(nifti_path.endswith('.nii.gz'))
        self.assertTrue(json_path.endswith('.json'))
        
        # Verify files exist
        self.assertTrue(os.path.exists(nifti_path))
        self.assertTrue(os.path.exists(json_path))
        
        print("\n")
        print(f"Default format is .nii.gz: {nifti_path}")


if __name__ == '__main__':
    unittest.main()

