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


from numpy.typing import NDArray

from InputFuncData import InputFuncData


class RadialArteryData(InputFuncData):
    """Class for handling radial artery input function data from PET imaging.

    This class extends InputFuncData to provide specialized handling of radial artery input functions,
    including kernel convolution for dispersion correction. It requires all incoming PET data to be
    decay corrected.

    Args:
        context: The context object containing solver, data, and IO information.
        data_dict (dict, optional): Dictionary containing configuration and data. Defaults to {}.
            Required keys:
                kernel_fqfn (str): Fully qualified filename for dispersion kernel data.

    Attributes:
        kernel (NDArray): Dispersion kernel array used for convolution.
        kernel_fqfn (str): Fully qualified filename for dispersion kernel data.
    """
    
    """ input function data assumed to be decay-corrected, consistent with tomography """
    
    def __init__(self, context, data_dict: dict = {}):
        super().__init__(context, data_dict)

        assert "kernel_fqfn" in self.data_dict, "data_dict missing required key 'kernel_fqfn'"
     
        self._data_dict["kernel"] = self.context.io.kernel_load(self.data_dict["kernel_fqfn"])["img"]   

    @property
    def kernel(self) -> NDArray:
        return self.data_dict["kernel"].copy()

    @property
    def kernel_fqfn(self) -> str:
        return self.data_dict["kernel_fqfn"]
