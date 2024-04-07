# PyTorch C++/CUDA Kernel Playground
Simple vector add C++/CUDA kernel. Based off of a similar tutorial from PyTorch from https://pytorch.org/tutorials/advanced/cpp_extension.html

## Installation
Execute `python setup.py develop` in the `cpp` directory

## Usage
Run `python main_cuda.py -l <some number> -d <cuda|cpu>`
Currently, a simple vector add C++/CUDA kernel is implemented for 1D, floating-point PyTorch Tensors.
*NOTE:* Still need to add a backward(...) function!