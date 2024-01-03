from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(
    ### Minimal Example Args
    name="gorby_vector_add",
    version="1.0.0",
    description="Dummy script for a C++ or C++/CUDA version of vector add invoked via Python",
    packages=find_packages(),
    install_requires=["torch >= 2.0", "pybind11"],
    ### PyTorch C++/CUDA Example
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="gorby_vector_add_cuda", sources=["cpp/gorby_vector_add_cuda.cu"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
