from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    ### Minimal Example Args
    name="gorby",
    install_requires=["torch >= 2.1", "pybind11"],
    ### PyTorch C++/CUDA Examples
    ext_modules=[
        #CUDAExtension(
        #    name="gorby_vector_add", sources=["gorby_vector_add.cu"]
        #),
        #CUDAExtension(
        #    name="gorby_nvtx", sources=["gorby_nvtx.cpp"]
        #),
        CUDAExtension(
            name="gorby_sdpa", sources=["gorby_sdpa.cu"],
            include_dirs=['C:\\Users\\neliopou\\Documents\\PhD\\Github\\cutlass\\include'],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': ['-std=c++17'],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)