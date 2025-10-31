"""
Refactored TAC data processing architecture following OOP best practices.
This module provides a clean separation of concerns with proper code reuse.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import interpolate
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Protocol, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings


# ============================================================================
# Data Models (Single Responsibility: Data representation)
# ============================================================================

@dataclass
class TACData:
    """
    Data model for time-activity curve data.
    Single responsibility: Encapsulate TAC data and metadata.
    """
    data: np.ndarray  # Shape: (n_regions, n_timepoints)
    times: np.ndarray  # Shape: (n_timepoints,)
    taus: Optional[np.ndarray] = None  # Shape: (n_timepoints,)
    timesMid: Optional[np.ndarray] = None  # Shape: (n_timepoints,)
    region_labels: Optional[List[str]] = None
    time_unit: str = "second"
    
    @property
    def n_regions(self) -> int:
        """Number of regions/ROIs."""
        return self.data.shape[0] if self.data.ndim >= 2 else 1
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return self.data.shape[1] if self.data.ndim >= 2 else self.data.shape[0]
    
    @property
    def is_plasma(self) -> bool:
        """Check if this is plasma (single region) data."""
        return self.n_regions == 1
    
    def to_nifti_shape(self) -> np.ndarray:
        """Ensure data is in proper NIfTI shape (n_regions, n_timepoints)."""
        if self.data.ndim == 1:
            return self.data.reshape(1, -1)
        return self.data


# ============================================================================
# File I/O Interfaces (Dependency Inversion Principle)
# ============================================================================

class TACReader(ABC):
    """
    Abstract interface for reading TAC data.
    Dependency Inversion: High-level modules depend on this abstraction.
    """
    
    @abstractmethod
    def can_read(self, filepath: Path) -> bool:
        """Check if this reader can handle the given file."""
        pass
    
    @abstractmethod
    def read(self, filepath: Path) -> TACData:
        """Read TAC data from file."""
        pass


class TACWriter(ABC):
    """
    Abstract interface for writing TAC data.
    Dependency Inversion: High-level modules depend on this abstraction.
    """
    
    @abstractmethod
    def can_write(self, filepath: Path) -> bool:
        """Check if this writer can handle the given file."""
        pass
    
    @abstractmethod
    def write(self, data: TACData, filepath: Path, **kwargs) -> None:
        """Write TAC data to file."""
        pass


# ============================================================================
# Utility Functions (Single Responsibility: Common operations)
# ============================================================================

class TACFileUtils:
    """Utility functions for TAC file operations."""
    
    @staticmethod
    def get_json_sidecar_path(nifti_path: Path) -> Path:
        """
        Get JSON sidecar path for a NIfTI file.
        Handles both .nii and .nii.gz extensions.
        """
        if nifti_path.suffix == '.gz':
            stem = nifti_path.stem  # removes .gz
            json_path = nifti_path.parent / f"{Path(stem).stem}.json"
        else:
            json_path = nifti_path.with_suffix('.json')
        return json_path
    
    @staticmethod
    def get_file_extension(filepath: Path) -> str:
        """Get normalized file extension."""
        if filepath.suffix == '.gz' and filepath.stem.endswith('.nii'):
            return '.nii.gz'
        return filepath.suffix.lower()
    
    @staticmethod
    def get_prototype_midframe_times(n_frames: int = 71) -> np.ndarray:
        """Get prototype midframe times for standard PET acquisition."""
        times = []
        # First 36 frames: 5-second intervals
        for i in range(min(36, n_frames)):
            times.append(2.5 + i * 5)
        # Next 12 frames: 10-second intervals
        if n_frames > 36:
            for i in range(min(12, n_frames - 36)):
                times.append(185 + i * 10)
        # Remaining frames: 300-second intervals
        if n_frames > 48:
            for i in range(n_frames - 48):
                times.append(305 + i * 300)
        return np.array(times[:n_frames])


# ============================================================================
# Concrete Readers (Open-Closed Principle: Extend via new implementations)
# ============================================================================

class PTACCSVReader(TACReader):
    """Reader for pTAC CSV files."""
    
    def can_read(self, filepath: Path) -> bool:
        """Check if file is a CSV with pTAC data."""
        if filepath.suffix.lower() != '.csv':
            return False
        try:
            df = pd.read_csv(filepath, nrows=1)
            return 'pTAC' in df.columns and 'Sec' in df.columns
        except:
            return False
    
    def read(self, filepath: Path) -> TACData:
        """Read pTAC data from CSV."""
        df = pd.read_csv(filepath)
        
        if 'pTAC' not in df.columns or 'Sec' not in df.columns:
            raise ValueError(f"CSV must contain 'pTAC' and 'Sec' columns")
        
        data = df['pTAC'].values.reshape(1, -1)
        times = df['Sec'].values
        
        # Calculate taus based on sampling intervals
        n_t = len(times)
        taus = np.zeros(n_t)
        taus[0] = 1.0
        
        for i in range(1, n_t):
            time_diff = times[i] - times[i-1]
            taus[i] = 1.0 if np.abs(time_diff - 1.0) < 1e-6 else 10.0
        
        timesMid = times.copy()
        times_adjusted = timesMid - taus / 2.0
        
        return TACData(
            data=data,
            times=times_adjusted,
            taus=taus,
            timesMid=timesMid,
            region_labels=None
        )


class TTACCSVReader(TACReader):
    """Reader for tTAC CSV files."""
    
    def __init__(self):
        self.utils = TACFileUtils()
    
    def can_read(self, filepath: Path) -> bool:
        """Check if file is a CSV with tTAC data."""
        if filepath.suffix.lower() != '.csv':
            return False
        try:
            df = pd.read_csv(filepath, nrows=1)
            return ('MidFrameTime_sec_' in df.columns or 
                    'FrameReferenceTime' in df.columns)
        except:
            return False
    
    def read(self, filepath: Path) -> TACData:
        """Read tTAC data from CSV."""
        df = pd.read_csv(filepath)
        
        # Handle time column variations
        if 'MidFrameTime_sec_' in df.columns:
            times_mid = df['MidFrameTime_sec_'].values
        elif 'FrameReferenceTime' in df.columns:
            times_mid = self.utils.get_prototype_midframe_times(len(df))
        else:
            raise ValueError("No recognized time column found")
        
        # Get anatomical columns
        time_cols = ['MidFrameTime_sec_', 'FrameReferenceTime', '']
        anatomical_cols = [col for col in df.columns 
                         if col not in time_cols and not df[col].isna().all()]
        
        # Extract data
        n_regions = len(anatomical_cols)
        n_t = len(df)
        data = np.zeros((n_regions, n_t))
        
        for i, col in enumerate(anatomical_cols):
            data[i, :] = df[col].values
        
        # Calculate taus for tTAC
        taus = np.zeros(n_t)
        if n_t > 0:
            taus[0:min(36, n_t)] = 5.0
        if n_t > 36:
            taus[36:min(48, n_t)] = 10.0
        if n_t > 48:
            taus[48:] = 300.0
        
        times = times_mid - taus / 2.0
        
        return TACData(
            data=data,
            times=times,
            taus=taus,
            timesMid=times_mid,
            region_labels=anatomical_cols
        )


class NIfTIReader(TACReader):
    """Reader for NIfTI files with JSON sidecars."""
    
    def __init__(self):
        self.utils = TACFileUtils()
    
    def can_read(self, filepath: Path) -> bool:
        """Check if file is a NIfTI with JSON sidecar."""
        ext = self.utils.get_file_extension(filepath)
        if ext not in ['.nii', '.nii.gz']:
            return False
        json_path = self.utils.get_json_sidecar_path(filepath)
        return json_path.exists()
    
    def read(self, filepath: Path) -> TACData:
        """Read TAC data from NIfTI with JSON sidecar."""
        # Load NIfTI data
        img = nib.load(str(filepath))
        data = img.get_fdata()
        
        # Ensure 2D shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Load JSON sidecar
        json_path = self.utils.get_json_sidecar_path(filepath)
        with open(json_path, 'r') as f:
            sidecar = json.load(f)
        
        # Extract timing arrays
        taus = np.array([t[0] for t in sidecar.get('taus', [])])
        times = np.array([t[0] for t in sidecar.get('times', [])])
        timesMid = np.array([t[0] for t in sidecar.get('timesMid', [])])
        
        # Use timesMid as primary time array if times not available
        if len(times) == 0 and len(timesMid) > 0:
            times = timesMid
        
        return TACData(
            data=data,
            times=times,
            taus=taus if len(taus) > 0 else None,
            timesMid=timesMid if len(timesMid) > 0 else None,
            region_labels=sidecar.get('anatomicalRegions', None),
            time_unit=sidecar.get('timeUnit', 'second')
        )


# ============================================================================
# Concrete Writers (Open-Closed Principle: Extend via new implementations)
# ============================================================================

class NIfTIWriter(TACWriter):
    """Writer for NIfTI files with JSON sidecars."""
    
    def __init__(self, header_params: Optional[Dict[str, Any]] = None):
        self.utils = TACFileUtils()
        self.header_params = header_params or self._get_default_header_params()
    
    def _get_default_header_params(self) -> Dict[str, Any]:
        """Get default NIfTI header parameters."""
        return {
            'pixdim': [-1.0, 0.825, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        }
    
    def can_write(self, filepath: Path) -> bool:
        """Check if can write to NIfTI format."""
        ext = self.utils.get_file_extension(filepath)
        return ext in ['.nii', '.nii.gz']
    
    def write(self, data: TACData, filepath: Path, **kwargs) -> None:
        """Write TAC data to NIfTI with JSON sidecar."""
        # Create NIfTI header
        header = nib.Nifti1Header()
        nifti_data = data.to_nifti_shape()
        
        # Set dimensions
        header['dim'][0] = 2
        header['dim'][1] = nifti_data.shape[0]  # n_regions
        header['dim'][2] = nifti_data.shape[1]  # n_timepoints
        header['dim'][3:8] = 1
        
        # Apply header parameters
        for key, value in self.header_params.items():
            if hasattr(header, key):
                header[key] = value
        
        header.set_data_dtype(np.float32)
        header.set_xyzt_units(xyz='mm', t='sec')
        
        # Create and save NIfTI
        nifti_img = nib.Nifti1Image(nifti_data.astype(np.float32), 
                                    affine=None, header=header)
        nib.save(nifti_img, str(filepath))
        
        # Create JSON sidecar
        json_path = self.utils.get_json_sidecar_path(filepath)
        sidecar = {
            'timeUnit': data.time_unit
        }
        
        if data.taus is not None:
            sidecar['taus'] = [[float(t)] for t in data.taus]
        if data.times is not None:
            sidecar['times'] = [[float(t)] for t in data.times]
        if data.timesMid is not None:
            sidecar['timesMid'] = [[float(t)] for t in data.timesMid]
        if data.region_labels is not None:
            sidecar['anatomicalRegions'] = data.region_labels
        
        with open(json_path, 'w') as f:
            json.dump(sidecar, f, indent=2)


# ============================================================================
# Repository Pattern (Single Responsibility: Data access coordination)
# ============================================================================

class TACDataRepository:
    """
    Repository for TAC data access.
    Single Responsibility: Coordinate data access through registered readers/writers.
    Open-Closed: Add new formats by registering new readers/writers.
    """
    
    def __init__(self):
        self.readers: List[TACReader] = []
        self.writers: List[TACWriter] = []
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default readers and writers."""
        # Register readers
        self.register_reader(PTACCSVReader())
        self.register_reader(TTACCSVReader())
        self.register_reader(NIfTIReader())
        
        # Register writers
        self.register_writer(NIfTIWriter())
    
    def register_reader(self, reader: TACReader):
        """Register a new reader."""
        self.readers.append(reader)
    
    def register_writer(self, writer: TACWriter):
        """Register a new writer."""
        self.writers.append(writer)
    
    def load(self, filepath: str) -> TACData:
        """
        Load TAC data from file using appropriate reader.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            TACData object
            
        Raises:
            ValueError: If no suitable reader found
        """
        path = Path(filepath)
        for reader in self.readers:
            if reader.can_read(path):
                return reader.read(path)
        
        raise ValueError(f"No reader available for file: {filepath}")
    
    def save(self, data: TACData, filepath: str, **kwargs) -> None:
        """
        Save TAC data to file using appropriate writer.
        
        Args:
            data: TACData object to save
            filepath: Path where to save the data
            **kwargs: Additional writer-specific parameters
            
        Raises:
            ValueError: If no suitable writer found
        """
        path = Path(filepath)
        for writer in self.writers:
            if writer.can_write(path):
                writer.write(data, path, **kwargs)
                return
        
        raise ValueError(f"No writer available for file: {filepath}")


# ============================================================================
# Visualization Strategies (Single Responsibility: Visualization only)
# ============================================================================

class TACVisualizer(ABC):
    """Abstract base for TAC visualizers."""
    
    @abstractmethod
    def can_visualize(self, data: TACData) -> bool:
        """Check if this visualizer can handle the data."""
        pass
    
    @abstractmethod
    def visualize(self, data: TACData, **kwargs) -> plt.Figure:
        """Create visualization."""
        pass


class PlasmaVisualizer(TACVisualizer):
    """Visualizer for plasma TAC data."""
    
    def can_visualize(self, data: TACData) -> bool:
        """Can visualize single-region data."""
        return data.is_plasma
    
    def visualize(self, data: TACData, **kwargs) -> plt.Figure:
        """Create line plot for plasma TAC."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        ptac_values = data.data[0, :]
        times = data.timesMid if data.timesMid is not None else data.times
        
        if kwargs.get('interpolate', True) and len(times) > 3:
            f_interp = interpolate.interp1d(times, ptac_values, kind='cubic',
                                          fill_value='extrapolate')
            times_fine = np.linspace(times.min(), times.max(), 500)
            ptac_fine = f_interp(times_fine)
            
            ax.plot(times_fine, ptac_fine, 'b-', alpha=0.7, linewidth=2,
                   label='pTAC (interpolated)')
            ax.scatter(times, ptac_values, color='red', s=30, zorder=5,
                      label='pTAC (measured)', alpha=0.8)
        else:
            ax.plot(times, ptac_values, 'b-', linewidth=2, marker='o',
                   markersize=5, label='pTAC')
        
        ax.set_xlabel(f'Time ({data.time_unit}s)', fontsize=12)
        ax.set_ylabel('Activity Concentration', fontsize=12)
        ax.set_title(kwargs.get('title', 'Plasma Time-Activity Curve'), fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class TissueHeatmapVisualizer(TACVisualizer):
    """Visualizer for tissue TAC heatmaps."""
    
    def can_visualize(self, data: TACData) -> bool:
        """Can visualize multi-region data."""
        return not data.is_plasma
    
    def visualize(self, data: TACData, **kwargs) -> plt.Figure:
        """Create heatmap for tissue TAC."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (14, 8)))
        
        plot_data = data.data.copy()
        if kwargs.get('normalize', False):
            for i in range(data.n_regions):
                min_val, max_val = plot_data[i, :].min(), plot_data[i, :].max()
                if max_val > min_val:
                    plot_data[i, :] = (plot_data[i, :] - min_val) / (max_val - min_val)
        
        im = ax.imshow(plot_data, aspect='auto', cmap=kwargs.get('cmap', 'viridis'),
                      interpolation=kwargs.get('interpolation', 'bilinear'))
        
        # Time axis
        times = data.timesMid if data.timesMid is not None else data.times
        n_time_ticks = min(10, len(times))
        time_indices = np.linspace(0, len(times)-1, n_time_ticks, dtype=int)
        ax.set_xticks(time_indices)
        ax.set_xticklabels([f'{times[i]:.0f}' for i in time_indices], rotation=45)
        ax.set_xlabel(f'Time ({data.time_unit}s)', fontsize=12)
        
        # Region axis
        if data.region_labels:
            n_regions = data.n_regions
            if n_regions > 30:
                step = n_regions // 30
                y_indices = list(range(0, n_regions, step))
                ax.set_yticks(y_indices)
                ax.set_yticklabels([data.region_labels[i] for i in y_indices], fontsize=8)
            else:
                ax.set_yticks(range(n_regions))
                ax.set_yticklabels(data.region_labels, fontsize=9)
        else:
            ax.set_ylabel('Region Index', fontsize=12)
        
        ax.set_title(kwargs.get('title', 'Tissue Time-Activity Curves Heatmap'), fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity Concentration' + 
                      (' (normalized)' if kwargs.get('normalize', False) else ''),
                      rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig


class OverlayVisualizer(TACVisualizer):
    """Visualizer for overlay plots."""
    
    def can_visualize(self, data: TACData) -> bool:
        """This is a special visualizer that needs two datasets."""
        return False  # Handled differently
    
    def visualize(self, data: TACData, **kwargs) -> plt.Figure:
        """Not used directly - see visualize_overlay."""
        raise NotImplementedError("Use visualize_overlay for overlay plots")
    
    def visualize_overlay(self, plasma_data: TACData, tissue_data: TACData, 
                         **kwargs) -> plt.Figure:
        """Create overlay plot with plasma over tissue ribbon."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 7)))
        
        # Get time arrays
        ptac_times = plasma_data.timesMid if plasma_data.timesMid is not None else plasma_data.times
        ttac_times = tissue_data.timesMid if tissue_data.timesMid is not None else tissue_data.times
        
        # Calculate tissue envelope
        ttac_min = tissue_data.data.min(axis=0)
        ttac_max = tissue_data.data.max(axis=0)
        ttac_mean = tissue_data.data.mean(axis=0)
        
        if kwargs.get('interpolate', True):
            # Create interpolation functions
            f_min = interpolate.interp1d(ttac_times, ttac_min, kind='cubic',
                                        fill_value='extrapolate')
            f_max = interpolate.interp1d(ttac_times, ttac_max, kind='cubic',
                                        fill_value='extrapolate')
            f_mean = interpolate.interp1d(ttac_times, ttac_mean, kind='cubic',
                                         fill_value='extrapolate')
            
            ptac_values = plasma_data.data[0, :]
            f_ptac = interpolate.interp1d(ptac_times, ptac_values, kind='cubic',
                                         fill_value='extrapolate')
            
            # Common time grid
            time_min = max(ttac_times.min(), ptac_times.min())
            time_max = min(ttac_times.max(), ptac_times.max())
            times_fine = np.linspace(time_min, time_max, 500)
            
            # Plot interpolated
            ax.fill_between(times_fine, f_min(times_fine), f_max(times_fine),
                           alpha=0.3, color='gray', label='tTAC range')
            ax.plot(times_fine, f_mean(times_fine), 'g--', alpha=0.5,
                   linewidth=1.5, label='tTAC mean')
            ax.plot(times_fine, f_ptac(times_fine), 'r-', linewidth=2.5,
                   label='pTAC (interpolated)')
            ax.scatter(ptac_times, plasma_data.data[0, :], color='darkred',
                      s=40, zorder=5, alpha=0.7, label='pTAC (measured)')
        else:
            ax.fill_between(ttac_times, ttac_min, ttac_max,
                           alpha=0.3, color='gray', label='tTAC range')
            ax.plot(ttac_times, ttac_mean, 'g--', alpha=0.5,
                   linewidth=1.5, label='tTAC mean')
            ax.plot(ptac_times, plasma_data.data[0, :], 'r-', linewidth=2.5,
                   marker='o', markersize=5, label='pTAC')
        
        ax.set_xlabel(f'Time ({plasma_data.time_unit}s)', fontsize=12)
        ax.set_ylabel('Activity Concentration', fontsize=12)
        ax.set_title(kwargs.get('title', 'Plasma and Tissue Time-Activity Curves'),
                    fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if kwargs.get('show_stats', True):
            stats_text = f'Regions: {tissue_data.n_regions}\n'
            stats_text += f'Time points: {tissue_data.n_timepoints}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ============================================================================
# Refactored High-Level Classes (Using shared infrastructure)
# ============================================================================

class TACToNiftiConverter(ABC):
    """
    Refactored abstract base class for TAC to NIfTI conversion.
    Now uses the shared repository infrastructure.
    """
    
    def __init__(self):
        """Initialize with repository for data access."""
        self.repository = TACDataRepository()
        self.file_utils = TACFileUtils()
    
    @abstractmethod
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None, 
                            **kwargs) -> Tuple[str, str]:
        """Process CSV to NIfTI - must be implemented by subclasses."""
        pass
    
    def validate_nifti_file(self, nifti_path: str) -> Dict[str, Any]:
        """Validate NIfTI file using shared repository."""
        data = self.repository.load(nifti_path)
        return {
            'data_shape': data.data.shape,
            'n_regions': data.n_regions,
            'n_timepoints': data.n_timepoints,
            'has_labels': data.region_labels is not None,
            'time_unit': data.time_unit,
            'data_min': float(np.min(data.data)),
            'data_max': float(np.max(data.data))
        }


class PTACToNIfTIConverter(TACToNiftiConverter):
    """Refactored pTAC to NIfTI converter using shared infrastructure."""
    
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None,
                            **kwargs) -> Tuple[str, str]:
        """Process pTAC CSV to NIfTI."""
        # Load data using repository
        data = self.repository.load(csv_path)
        
        # Determine output paths
        if output_prefix is None:
            output_prefix = Path(csv_path).stem
        
        nifti_path = f"{output_prefix}.nii"
        
        # Save using repository
        self.repository.save(data, nifti_path)
        
        json_path = self.file_utils.get_json_sidecar_path(Path(nifti_path))
        
        print(f"Successfully created:")
        print(f"  NIfTI file: {nifti_path}")
        print(f"  JSON sidecar: {json_path}")
        print(f"  Data shape: {data.data.shape} (regions x time)")
        
        return nifti_path, str(json_path)


class TTACToNIfTIConverter(TACToNiftiConverter):
    """Refactored tTAC to NIfTI converter using shared infrastructure."""
    
    def process_csv_to_nifti(self, csv_path: str, output_prefix: str = None,
                            include_region_labels: bool = True, **kwargs) -> Tuple[str, str]:
        """Process tTAC CSV to NIfTI."""
        # Load data using repository
        data = self.repository.load(csv_path)
        
        # Remove labels if not wanted
        if not include_region_labels:
            data.region_labels = None
        
        # Determine output paths
        if output_prefix is None:
            output_prefix = Path(csv_path).stem
        
        nifti_path = f"{output_prefix}_ttac.nii"
        
        # Save using repository
        self.repository.save(data, nifti_path)
        
        json_path = self.file_utils.get_json_sidecar_path(Path(nifti_path))
        
        print(f"Successfully created:")
        print(f"  NIfTI file: {nifti_path}")
        print(f"  JSON sidecar: {json_path}")
        print(f"  Data shape: {data.data.shape} (regions x time)")
        print(f"  Regions: {data.n_regions}")
        
        return nifti_path, str(json_path)


class TACViewer:
    """
    Refactored TAC viewer using shared infrastructure.
    Now properly reuses code from converters via the repository pattern.
    """
    
    def __init__(self):
        """Initialize viewer with shared components."""
        self.repository = TACDataRepository()
        self.visualizers = [
            PlasmaVisualizer(),
            TissueHeatmapVisualizer(),
        ]
        self.overlay_visualizer = OverlayVisualizer()
    
    def _get_visualizer(self, data: TACData) -> TACVisualizer:
        """Get appropriate visualizer for data."""
        for viz in self.visualizers:
            if viz.can_visualize(data):
                return viz
        raise ValueError("No suitable visualizer found for data")
    
    def view(self, *args, **kwargs) -> plt.Figure:
        """
        Smart viewer with automatic visualization selection.
        
        Examples:
            viewer.view("ptac.csv")  # Plasma plot
            viewer.view("ttac.nii")  # Tissue heatmap
            viewer.view("ptac.csv", "ttac.nii")  # Overlay
        """
        if len(args) == 1:
            # Single file
            data = self.repository.load(args[0])
            visualizer = self._get_visualizer(data)
            return visualizer.visualize(data, **kwargs)
        
        elif len(args) == 2:
            # Two files - try overlay
            data1 = self.repository.load(args[0])
            data2 = self.repository.load(args[1])
            
            # Determine which is plasma and which is tissue
            if data1.is_plasma and not data2.is_plasma:
                return self.overlay_visualizer.visualize_overlay(data1, data2, **kwargs)
            elif data2.is_plasma and not data1.is_plasma:
                return self.overlay_visualizer.visualize_overlay(data2, data1, **kwargs)
            else:
                raise ValueError("Overlay requires one plasma and one tissue dataset")
        
        else:
            raise ValueError(f"Expected 1 or 2 files, got {len(args)}")
    
    def plot_ptac(self, filepath: str, **kwargs) -> plt.Figure:
        """Plot plasma TAC."""
        data = self.repository.load(filepath)
        if not data.is_plasma:
            raise ValueError("File does not contain plasma data")
        return PlasmaVisualizer().visualize(data, **kwargs)
    
    def plot_ttac_heatmap(self, filepath: str, **kwargs) -> plt.Figure:
        """Plot tissue TAC heatmap."""
        data = self.repository.load(filepath)
        if data.is_plasma:
            raise ValueError("File does not contain tissue data")
        return TissueHeatmapVisualizer().visualize(data, **kwargs)
    
    def plot_overlay(self, ptac_file: str, ttac_file: str, **kwargs) -> plt.Figure:
        """Create overlay plot."""
        plasma_data = self.repository.load(ptac_file)
        tissue_data = self.repository.load(ttac_file)
        
        if not plasma_data.is_plasma:
            plasma_data, tissue_data = tissue_data, plasma_data
        
        if not plasma_data.is_plasma or tissue_data.is_plasma:
            raise ValueError("Need one plasma and one tissue dataset for overlay")
        
        return self.overlay_visualizer.visualize_overlay(plasma_data, tissue_data, **kwargs)


# Example usage demonstrating the refactored architecture
if __name__ == "__main__":
    print("Refactored TAC Data Architecture")
    print("=" * 50)
    print("\nKey improvements:")
    print("1. Single Responsibility: Each class has one clear purpose")
    print("2. Open-Closed: Easy to add new formats without modifying existing code")
    print("3. Liskov Substitution: All readers/writers/visualizers are interchangeable")
    print("4. Dependency Inversion: High-level classes depend on abstractions")
    print("5. Code Reuse: Converters and viewers share the same data infrastructure")
    print("\nShared components:")
    print("- TACData: Common data model")
    print("- TACDataRepository: Centralized data access")
    print("- TACFileUtils: Shared utilities")
    print("- Readers/Writers: Reusable I/O components")
    print("\nUsage remains simple:")
    print("  viewer = TACViewer()")
    print("  viewer.view('data.csv')  # Automatic visualization")
    print("  converter = PTACToNIfTIConverter()")
    print("  converter.process_csv_to_nifti('data.csv')  # Uses same infrastructure")
