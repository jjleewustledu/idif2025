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

from abc import abstractmethod
from copy import deepcopy
from pprint import pprint
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


class DynestyData:
    def __init__(self, context, data_dict: dict = {}):
        self.context = context
        self._data_dict = data_dict
        
    def __deepcopy__(self, memo):
        return DynestyData(deepcopy(self._data_dict, memo))
    
    def __getstate__(self):
        return {'data_dict': self._data_dict}
    
    def __setstate__(self, state):
        self._data_dict = state['data_dict']
    
    @property
    def data_dict(self) -> dict:
        return deepcopy(self._data_dict)
    
    @property
    def fqfp(self) -> str:
        return self.context.io.fqfp

    @property
    def results_fqfp(self) -> str:
        return self.context.io.results_fqfp
    
    @property
    @abstractmethod
    def rho(self) -> NDArray:
        pass
    
    def fileparts(self, fqfn: str) -> tuple:
        return self.context.io.fileparts(fqfn)
    
    def fqfileprefix(self, fqfn: str) -> str:
        return self.context.io.fqfileprefix(fqfn)
    
    def nii_load(self, fqfn: str, time_last: float | None = None) -> dict:
        return self.context.io.nii_load(fqfn, time_last=time_last)
    
    def nii_save(self, data: dict, fqfn: str | None = None) -> None:
        return self.context.io.nii_save(data, fqfn)
    
    def pickle_dump(self, data: Any, fqfn: str | None = None) -> None:
        return self.context.io.pickle_dump(data, fqfn)
    
    def pickle_load(self, fqfn: str) -> Any:
        return self.context.io.pickle_load(fqfn)
        
    def print_concise(self, data: dict, title: str = "") -> None:
        """ Print a concise representation of a data dictionary useful for debugging. """
        self.print_separator(title)
        data_display = deepcopy(data)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data_display[key] = f"<array shape={value.shape}>"
            elif isinstance(value, (list, tuple)):
                data_display[key] = f"<{type(value).__name__} size={len(value)}>"
            elif isinstance(value, dict):
                data_display[key] = f"<dict keys={list(value.keys())}>"
            elif isinstance(value, nib.Nifti1Image):
                data_display[key] = f"<{type(value).__name__}>"
        pprint(data_display)
        self.print_separator(title, closing=True)

    def print_data(self) -> None:
        """ Print the data in a table format. """
        self.print_separator("Data")
        for key, value in self.data_dict.items():
            if isinstance(value, np.ndarray):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Values: {np.array2string(value, threshold=6, edgeitems=3)}")
            else:
                print(f"{key}: {value}")
        self.print_separator("Data", closing=True)

    def print_separator(self, text: str, closing: bool=False) -> None:
        if not closing:
            if len(text) > 62:
                print("\n\n" + "=" * 3 + " " + text + " " + "=" * 3)
            else:
                line_len = 70 - len(text) - 2  # -2 for spaces around text
                pad = line_len // 2
                print("\n\n" + "=" * pad + " " + text + " " + "=" * pad)
        else:
            print("=" * 70)
        
    def print_truths(self) -> None:
        """ Print the truths in a table format. """
        labels = self.context.solver.labels
        truths = self.context.solver.truths

        self.print_separator("Truths")
        print(f"{'Parameter':<25} {'Value':>12}")
        print("-" * 40)
        if truths.ndim == 2:
            for i, truth_row in enumerate(truths):
                print(f"\nRow {i}:")
                for label, value in zip(labels, truth_row):
                    print(f"{label:<25} {value:>12.5f}")
        else:
            for label, value in zip(labels, truths):
                print(f"{label:<25} {value:>12.5f}")
        self.print_separator("Truths", closing=True)
    
    def to_csv(self, data: dict, fqfn: str | None = None) -> None:
        return self.context.io.to_csv(data, fqfn)
    
