"""
CUDA Extension Build Script

Builds custom CUDA kernels as a Python extension module.
Requires: CUDA Toolkit, PyTorch (for compilation), pybind11
"""

import os
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake-based extension for CUDA kernels."""
    
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command using CMake."""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={os.sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        
        build_args = ["--config", "Release"]
        
        # Set build parallelism
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["-j4"]
        
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


# Alternative: Direct setuptools build without CMake
def get_cuda_extension():
    """Build CUDA extension using setuptools directly."""
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    
    include_dirs = [
        os.path.join(cuda_home, "include"),
        "kernels",
    ]
    
    library_dirs = [
        os.path.join(cuda_home, "lib64"),
    ]
    
    sources = [
        "bindings.cpp",
        "kernels/resize_kernel.cu",
        "kernels/normalize_kernel.cu",
        "kernels/preprocess_kernel.cu",
        "kernels/nms_kernel.cu",
    ]
    
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # T4
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
            "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
            "--expt-relaxed-constexpr",
        ],
    }
    
    return CUDAExtension(
        name="cuda_kernels",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["cudart"],
        extra_compile_args=extra_compile_args,
    )


# Check if we have PyTorch available for building
try:
    from torch.utils.cpp_extension import BuildExtension
    
    ext_modules = [get_cuda_extension()]
    cmdclass = {"build_ext": BuildExtension}
    
except ImportError:
    print("PyTorch not found, using CMake build")
    ext_modules = [CMakeExtension("cuda_kernels")]
    cmdclass = {"build_ext": CMakeBuild}


setup(
    name="gpu_pipeline_cuda",
    version="1.0.0",
    author="Saurabh Rai",
    author_email="rai.saurabh9491@gmail.com",
    description="GPU-accelerated preprocessing kernels for ML inference",
    long_description=open("../../README.md").read() if os.path.exists("../../README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
)
