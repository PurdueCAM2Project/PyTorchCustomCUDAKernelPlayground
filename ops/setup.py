import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, COMMON_MSVC_FLAGS

if os.name == "nt":
    COMMON_MSVC_FLAGS += ['/Zc:__cplusplus']

setup(
    ### Minimal Example Args
    name="gorby",
    install_requires=["torch >= 2.1", "pybind11"],
    ### PyTorch C++/CUDA Examples
    ### NOTE: /Zc:__cplusplus is related to MSVC incorrectly setting __cplusplus macro. See CUTLASS CMAKE and 
    ### https://github.com/NVIDIA/cutlass/issues/1474
    ext_modules=[
        CUDAExtension(
           name="gorby_vector_add", sources=["gorby_vector_add.cu"],
        ),
        CUDAExtension(
           name="gorby_nvtx", sources=["gorby_nvtx.cpp"],
        ),
        CUDAExtension(
            name="gorby_sdpa", sources=["gorby_sdpa.cu"],
            include_dirs=['C:\\Users\\neliopou\\Documents\\PhD\\Github\\cutlass-dev\\include'],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)