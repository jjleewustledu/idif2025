import numpy as np
import pandas as pd
import nibabel as nib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod


class TACToNiftiConverter(ABC):
    """
    Abstract base class for converting time-activity curve (TAC) data from CSV to NIfTI format
    with accompanying JSON sidecar files.
    
    This base class provides common functionality for both pTAC and tTAC converters.
    dim1 always represents regions of interest, dim2 always represents time samples.
    """
    
    def __init__(self):
        """Initialize the converter with default NIfTI header parameters."""
        self.default_header_params = self._get_default_header_params()
    
    @abstractmethod
    def _get_default_header_params(self) -> Dict[str, Any]:
        """
        Get default header parameters for NIfTI files.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary containing default NIfTI header parameters
        """
        pass
    
    @abstractmethod
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Read TAC data from CSV file.
        Must be implemented by subclasses to handle specific CSV formats.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        pass
    
    @abstractmethod
    def extract_tac_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
        """
        Extract TAC data from DataFrame and return shaped array.
        
        Args:
            df: DataFrame containing TAC data
            
        Returns:
            Tuple of (tac_data array shaped as (n_regions, n_t), n_regions, n_t)
        """
        pass
    
    @abstractmethod
    def calculate_timing_arrays(self, n_t: int, time_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate timing arrays (taus, times, timesMid) based on specific requirements.
        Must be implemented by subclasses.
        
        Args:
            n_t: Number of time points
            time_values: Array of time values from CSV
            
        Returns:
            Tuple of (taus, times, timesMid) arrays
        """
        pass
    
    def create_nifti_header(self, n_regions: int, n_t: int) -> nib.Nifti1Header:
        """
        Create NIfTI header with appropriate dimensions and parameters.
        dim1 = number of regions, dim2 = number of time points
        
        Args:
            n_regions: Number of regions (dim1)
            n_t: Number of time points (dim2)
            
        Returns:
            Configured NIfTI header
        """
        header = nib.Nifti1Header()
        
        # Set dimensions: dim1 = regions, dim2 = time
        header['dim'][0] = 2  # Number of dimensions
        header['dim'][1] = n_regions  # Number of regions
        header['dim'][2] = n_t  # Number of time points
        header['dim'][3:8] = 1  # Other dimensions set to 1
        
        # Set data type
        header.set_data_dtype(np.float32)
        
        # Set pixel dimensions
        header['pixdim'] = self.default_header_params['pixdim']
        
        # Set scaling parameters
        header['cal_max'] = self.default_header_params.get('cal_max', 0.0)
        header['cal_min'] = self.default_header_params.get('cal_min', 0.0)
        header['scl_slope'] = self.default_header_params.get('scl_slope', 1.0)
        header['scl_inter'] = self.default_header_params.get('scl_inter', 0.0)
        
        # Set coordinate system parameters
        header['qform_code'] = self.default_header_params.get('qform_code', 0)
        header['sform_code'] = self.default_header_params.get('sform_code', 1)
        
        # Set offsets
        header['qoffset_x'] = self.default_header_params.get('qoffset_x', 179.710999)
        header['qoffset_y'] = self.default_header_params.get('qoffset_y', -12.151001)
        header['qoffset_z'] = self.default_header_params.get('qoffset_z', 1704.416138)
        
        # Set affine transformation
        header['srow_x'] = self.default_header_params.get('srow_x', [-0.825, 0.0, 0.0, 179.710999])
        header['srow_y'] = self.default_header_params.get('srow_y', [0.0, 0.825, 0.0, -12.151001])
        header['srow_z'] = self.default_header_params.get('srow_z', [0.0, 0.0, 1.650024, 1704.416138])
        
        # Set description
        descrip = self.default_header_params.get('descrip', 'Time-activity curve data')
        header['descrip'] = descrip.encode('ascii')[:80]
        
        # Set units
        header.set_xyzt_units(xyz='mm', t='sec')
        
        return header
    
    def create_json_sidecar(self, taus: np.ndarray, times: np.ndarray,
                          timesMid: np.ndarray, additional_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create JSON sidecar content with timing information.
        
        Args:
            taus: Array of tau values
            times: Array of time values
            timesMid: Array of mid-time values
            additional_fields: Optional additional fields to include in the sidecar
            
        Returns:
            Dictionary containing JSON sidecar content
        """
        # Convert arrays to column vectors (N_t x 1) format as nested lists
        sidecar_content = {
            'taus': [[float(val)] for val in taus],
            'times': [[float(val)] for val in times],
            'timesMid': [[float(val)] for val in timesMid],
            'timeUnit': 'second'
        }
        
        # Add any additional fields
        if additional_fields:
            sidecar_content.update(additional_fields)
        
        return sidecar_content
    
    def validate_nifti_file(self, nifti_path: str) -> Dict[str, Any]:
        """
        Validate the created NIfTI file and return key header information.
        
        Args:
            nifti_path: Path to the NIfTI file to validate
            
        Returns:
            Dictionary containing validation results and header information
        """
        img = nib.load(nifti_path)
        header = img.header
        data = img.get_fdata()
        
        validation_info = {
            'data_shape': data.shape,
            'data_type': header.get_data_dtype(),
            'dim': list(header['dim']),
            'pixdim': list(header['pixdim']),
            'qform_code': int(header['qform_code']),
            'sform_code': int(header['sform_code']),
            'xyzt_units': header.get_xyzt_units(),
            'data_min': float(np.min(data)),
            'data_max': float(np.max(data)),
            'scl_slope': float(header['scl_slope']),
            'scl_inter': float(header['scl_inter'])
        }
        
        # Add dimension interpretation
        if len(data.shape) >= 2:
            validation_info['n_regions'] = data.shape[0]
            validation_info['n_timepoints'] = data.shape[1]
        else:
            validation_info['n_regions'] = 1
            validation_info['n_timepoints'] = data.shape[0]
        
        return validation_info
    
    @abstractmethod
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None, **kwargs) -> Tuple[str, str]:
        """
        Process a CSV file and create corresponding NIfTI and JSON sidecar files.
        Must be implemented by subclasses.
        
        Args:
            csv_path: Path to the input CSV file
            output_prefix: Optional prefix for output files
            **kwargs: Additional keyword arguments for specific implementations
            
        Returns:
            Tuple of (NIfTI file path, JSON file path)
        """
        pass


class PTACToNIfTIConverter(TACToNiftiConverter):
    """
    Converter for plasma time-activity curve (pTAC) data from CSV to NIfTI format.
    Generates NIfTI with dim1=1 (single region) and dim2=N_t (time samples).
    """
    
    def _get_default_header_params(self) -> Dict[str, Any]:
        """
        Get default header parameters based on the provided fslhd log file for pTAC.
        
        Returns:
            Dictionary containing default NIfTI header parameters
        """
        return {
            'pixdim': [-1.0, 0.825, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'datatype': 16,  # FLOAT32
            'bitpix': 32,
            'cal_max': 0.0,
            'cal_min': 0.0,
            'scl_slope': 1.0,
            'scl_inter': 0.0,
            'qform_code': 0,
            'sform_code': 1,
            'qoffset_x': 179.710999,
            'qoffset_y': -12.151001,
            'qoffset_z': 1704.416138,
            'srow_x': [-0.825, 0.0, 0.0, 179.710999],
            'srow_y': [0.0, 0.825, 0.0, -12.151001],
            'srow_z': [0.0, 0.0, 1.650024, 1704.416138],
            'vox_units': 'mm',
            'time_units': 's',
            'descrip': 'Plasma time-activity curve data'
        }
    
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Read pTAC data from CSV file.
        
        Args:
            csv_path: Path to the CSV file containing pTAC data
            
        Returns:
            DataFrame containing the CSV data
        """
        df = pd.read_csv(csv_path)
        
        # Handle mc_pTAC column name (user typo) - rename to pTAC
        if 'mc_pTAC' in df.columns and 'pTAC' not in df.columns:
            df = df.rename(columns={'mc_pTAC': 'pTAC'})
        
        # Verify required columns exist
        required_columns = ['DateTime', 'Min', 'Sec', 'pTAC']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def extract_tac_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
        """
        Extract pTAC data from DataFrame and reshape to (1, N_t).
        
        Args:
            df: DataFrame containing pTAC data
            
        Returns:
            Tuple of (pTAC data array shaped as (1, N_t), n_regions=1, n_t)
        """
        ptac_data = df['pTAC'].values.astype(np.float32)
        n_t = len(ptac_data)
        n_regions = 1
        
        # Reshape to (1, N_t) - single region, multiple time points
        tac_data = ptac_data.reshape(n_regions, n_t)
        
        return tac_data, n_regions, n_t
    
    def calculate_timing_arrays(self, n_t: int, time_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate timing arrays for pTAC data.
        
        Args:
            n_t: Number of time points
            time_values: Array of seconds from the CSV file
            
        Returns:
            Tuple of (taus, times, timesMid) arrays
        """
        # timesMid is directly from the Sec column
        timesMid = time_values.astype(float)
        
        # Initialize taus array
        taus = np.zeros(n_t)
        taus[0] = 1.0
        
        # Calculate taus based on sampling intervals
        for i in range(1, n_t):
            time_diff = timesMid[i] - timesMid[i-1]
            if np.abs(time_diff - 1.0) < 1e-6:  # Using tolerance for floating point comparison
                taus[i] = 1.0
            else:
                taus[i] = 10.0
        
        # Calculate times
        times = timesMid - taus / 2.0
        
        return taus, times, timesMid
    
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None, **kwargs) -> Tuple[str, str]:
        """
        Process a pTAC CSV file and create corresponding NIfTI and JSON sidecar files.
        
        Args:
            csv_path: Path to the input CSV file
            output_prefix: Optional prefix for output files
            
        Returns:
            Tuple of (NIfTI file path, JSON file path)
        """
        # Read CSV data
        df = self.read_csv(csv_path)
        
        # Extract pTAC data with shape (1, N_t)
        tac_data, n_regions, n_t = self.extract_tac_data(df)
        
        # Calculate timing arrays
        taus, times, timesMid = self.calculate_timing_arrays(n_t, df['Sec'].values)
        
        # Create NIfTI header with dim1=1, dim2=N_t
        header = self.create_nifti_header(n_regions, n_t)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(tac_data, affine=None, header=header)
        
        # Determine output file paths
        if output_prefix is None:
            output_prefix = Path(csv_path).stem
        
        # Handle file extensions: support both .nii and .nii.gz
        # JSON must always end in .json, never .nii.json or .nii.gz.json
        if output_prefix.endswith('.nii.gz'):
            nifti_path = output_prefix
            json_base = output_prefix[:-7]  # Remove .nii.gz
            json_path = f"{json_base}.json"
        elif output_prefix.endswith('.nii'):
            nifti_path = output_prefix
            json_base = output_prefix[:-4]  # Remove .nii
            json_path = f"{json_base}.json"
        else:
            # Default to .nii.gz to match codebase standard
            nifti_path = f"{output_prefix}.nii.gz"
            json_path = f"{output_prefix}.json"
        
        # Save NIfTI file
        nib.save(nifti_img, nifti_path)
        
        # Create and save JSON sidecar
        sidecar_content = self.create_json_sidecar(taus, times, timesMid)
        with open(json_path, 'w') as f:
            json.dump(sidecar_content, f, indent=2)
        
        print(f"Successfully created:")
        print(f"  NIfTI file: {nifti_path}")
        print(f"  JSON sidecar: {json_path}")
        print(f"  Data dimensions: {tac_data.shape} (regions x time)")
        print(f"  Number of regions: {n_regions}")
        print(f"  Number of time points: {n_t}")
        
        return nifti_path, json_path


class TTACToNIfTIConverter(TACToNiftiConverter):
    """
    Converter for tissue time-activity curve (tTAC) data from CSV to NIfTI format.
    Generates NIfTI with dim1=n_anatomical (multiple regions) and dim2=N_t (time samples).
    """
    
    def __init__(self):
        """Initialize the converter with default parameters and prototype time values."""
        super().__init__()
        self.prototype_midframe_times = self._get_prototype_midframe_times()
    
    def _get_prototype_midframe_times(self) -> np.ndarray:
        """
        Get the prototype MidFrameTime_sec_ values from the reference file.
        
        Returns:
            Array of prototype midframe time values
        """
        prototype_times = []
        
        # First 36 frames: 5-second intervals (centered at 2.5, 7.5, 12.5, ...)
        for i in range(36):
            prototype_times.append(2.5 + i * 5)
        
        # Next 12 frames: 10-second intervals
        for i in range(12):
            prototype_times.append(185 + i * 10)
        
        # Remaining frames: 300-second intervals
        remaining_start = 305
        for i in range(71 - 48):  # Total 71 rows minus first 48
            prototype_times.append(remaining_start + i * 300)
        
        return np.array(prototype_times[:71])
    
    def _get_default_header_params(self) -> Dict[str, Any]:
        """
        Get default header parameters based on the fslhd_BrainMoCo2_ParcSchaeffer.log file.
        
        Returns:
            Dictionary containing default NIfTI header parameters
        """
        return {
            'pixdim': [-1.0, 0.825, 0.825, 1.650024, 0.0, 0.0, 0.0, 0.0],
            'datatype': 16,  # FLOAT32
            'bitpix': 32,
            'cal_max': 0.0,
            'cal_min': 0.0,
            'scl_slope': 1.0,  # Corrected from 0.0 to 1.0 for proper scaling
            'scl_inter': 0.0,
            'qform_code': 0,
            'sform_code': 1,
            'qoffset_x': 179.710999,
            'qoffset_y': -12.151001,
            'qoffset_z': 1704.416138,
            'srow_x': [-0.825, 0.0, 0.0, 179.710999],
            'srow_y': [0.0, 0.825, 0.0, -12.151001],
            'srow_z': [0.0, 0.0, 1.650024, 1704.416138],
            'vox_units': 'mm',
            'time_units': 's',
            'descrip': 'Tissue time-activity curve data'
        }
    
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Read tTAC data from CSV file and handle irregularities.
        
        Args:
            csv_path: Path to the CSV file containing tTAC data
            
        Returns:
            DataFrame containing the processed CSV data
        """
        df = pd.read_csv(csv_path)
        
        # Handle FrameReferenceTime column replacement
        if 'FrameReferenceTime' in df.columns:
            print("Replacing FrameReferenceTime with prototype MidFrameTime_sec_ values")
            df['MidFrameTime_sec_'] = self.prototype_midframe_times[:len(df)]
            df = df.drop(columns=['FrameReferenceTime'])
        
        # Remove columns with no label (empty column names)
        df = df.loc[:, df.columns != '']
        df = df.dropna(axis=1, how='all')  # Also remove completely empty columns
        
        # Verify time column exists
        if 'MidFrameTime_sec_' not in df.columns:
            raise ValueError("Missing required column: MidFrameTime_sec_")
        
        return df
    
    def get_anatomical_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of anatomical column names (all columns except MidFrameTime_sec_).
        
        Args:
            df: DataFrame containing tTAC data
            
        Returns:
            List of anatomical column names
        """
        anatomical_cols = [col for col in df.columns if col != 'MidFrameTime_sec_']
        return anatomical_cols
    
    def extract_tac_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
        """
        Extract tTAC data from DataFrame and shape as (n_anatomical, N_t).
        
        Args:
            df: DataFrame containing tTAC data
            
        Returns:
            Tuple of (tTAC data array shaped as (n_anatomical, N_t), n_anatomical, n_t)
        """
        anatomical_cols = self.get_anatomical_columns(df)
        n_anatomical = len(anatomical_cols)
        n_t = len(df)
        
        # Extract tTAC data and shape as (n_anatomical, N_t)
        ttac_data = np.zeros((n_anatomical, n_t), dtype=np.float32)
        for i, col in enumerate(anatomical_cols):
            ttac_data[i, :] = df[col].values.astype(np.float32)
        
        return ttac_data, n_anatomical, n_t
    
    def calculate_timing_arrays(self, n_t: int, time_values: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate timing arrays for tTAC data with specific tau patterns.
        
        Args:
            n_t: Number of time points
            time_values: Optional array of midframe times from CSV
            
        Returns:
            Tuple of (taus, times, timesMid) arrays
        """
        # Use provided time_values or prototype
        if time_values is not None:
            timesMid = time_values.astype(float)
        else:
            timesMid = self.prototype_midframe_times[:n_t]
        
        # Initialize taus array with specific values based on index ranges
        taus = np.zeros(n_t)
        
        # Set taus values according to specifications
        if n_t > 0:
            taus[0:min(36, n_t)] = 5.0
        if n_t > 36:
            taus[36:min(48, n_t)] = 10.0
        if n_t > 48:
            taus[48:] = 300.0
        
        # Calculate times
        times = timesMid - taus / 2.0
        
        return taus, times, timesMid
    
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None, 
                            include_region_labels: bool = True, **kwargs) -> Tuple[str, str]:
        """
        Process a tTAC CSV file and create corresponding NIfTI and JSON sidecar files.
        
        Args:
            csv_path: Path to the input CSV file
            output_prefix: Optional prefix for output files
            include_region_labels: Whether to include anatomical region labels in JSON
            
        Returns:
            Tuple of (NIfTI file path, JSON file path)
        """
        # Read and process CSV data
        df = self.read_csv(csv_path)
        
        # Extract tTAC data with shape (n_anatomical, N_t)
        tac_data, n_anatomical, n_t = self.extract_tac_data(df)
        anatomical_cols = self.get_anatomical_columns(df)
        
        print(f"Processing {n_anatomical} anatomical regions with {n_t} time points")
        
        # Calculate timing arrays
        midframe_times = df['MidFrameTime_sec_'].values if 'MidFrameTime_sec_' in df.columns else None
        taus, times, timesMid = self.calculate_timing_arrays(n_t, midframe_times)
        
        # Create NIfTI header with dim1=n_anatomical, dim2=N_t
        header = self.create_nifti_header(n_anatomical, n_t)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(tac_data, affine=None, header=header)
        
        # Determine output file paths
        if output_prefix is None:
            output_prefix = Path(csv_path).stem
        
        # Handle file extensions: support both .nii and .nii.gz
        # JSON must always end in .json, never .nii.json or .nii.gz.json
        if output_prefix.endswith('.nii.gz'):
            nifti_path = output_prefix
            json_base = output_prefix[:-7]  # Remove .nii.gz
            json_path = f"{json_base}.json"
        elif output_prefix.endswith('.nii'):
            nifti_path = output_prefix
            json_base = output_prefix[:-4]  # Remove .nii
            json_path = f"{json_base}.json"
        else:
            # Default to .nii.gz to match codebase standard, add _ttac suffix
            nifti_path = f"{output_prefix}_ttac.nii.gz"
            json_path = f"{output_prefix}_ttac.json"
        
        # Save NIfTI file
        nib.save(nifti_img, nifti_path)
        
        # Prepare additional fields for JSON sidecar
        additional_fields = {}
        if include_region_labels:
            additional_fields['anatomicalRegions'] = anatomical_cols
        
        # Create and save JSON sidecar
        sidecar_content = self.create_json_sidecar(taus, times, timesMid, additional_fields)
        with open(json_path, 'w') as f:
            json.dump(sidecar_content, f, indent=2)
        
        print(f"Successfully created:")
        print(f"  NIfTI file: {nifti_path}")
        print(f"  JSON sidecar: {json_path}")
        print(f"  Data dimensions: {tac_data.shape} (regions x time)")
        print(f"  Anatomical regions: {n_anatomical}")
        print(f"  Time points: {n_t}")
        
        return nifti_path, json_path


# Example usage
if __name__ == "__main__":
    # Example for pTAC conversion
    print("="*50)
    print("Testing PTACToNIfTIConverter")
    print("="*50)
    
    ptac_converter = PTACToNIfTIConverter()
    ptac_csv_file = "subEVA118_sesretest_metab_corr_ptac.csv"
    
    try:
        # Convert pTAC CSV to NIfTI with JSON sidecar
        ptac_nifti, ptac_json = ptac_converter.process_csv_to_nifti(ptac_csv_file)
        
        # Validate the created NIfTI file
        ptac_validation = ptac_converter.validate_nifti_file(ptac_nifti)
        print("\nValidation Results for pTAC:")
        for key, value in ptac_validation.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError:
        print(f"Error: CSV file '{ptac_csv_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")
    
    # Example for tTAC conversion
    print("\n" + "="*50)
    print("Testing TTACToNIfTIConverter")
    print("="*50)
    
    ttac_converter = TTACToNIfTIConverter()
    ttac_csv_file = "subEVA118_sesretest_tacs.csv"
    
    try:
        # Convert tTAC CSV to NIfTI with JSON sidecar
        ttac_nifti, ttac_json = ttac_converter.process_csv_to_nifti(
            ttac_csv_file,
            include_region_labels=True
        )
        
        # Validate the created NIfTI file
        ttac_validation = ttac_converter.validate_nifti_file(ttac_nifti)
        print("\nValidation Results for tTAC:")
        for key, value in ttac_validation.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError:
        print(f"Error: CSV file '{ttac_csv_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")
