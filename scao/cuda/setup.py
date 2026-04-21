"""
Optional CUDA extension build script for SCAO.
Run:  python setup.py build_ext --inplace
"""

from setuptools import setup
try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    ext_modules = [
        CUDAExtension(
            name="scao.cuda._scao_cuda",
            sources=["low_rank_ops.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_70,code=sm_70",   # V100
                    "-gencode=arch=compute_75,code=sm_75",   # T4 / RTX 20xx
                    "-gencode=arch=compute_80,code=sm_80",   # A100
                    "-gencode=arch=compute_86,code=sm_86",   # RTX 30xx / A40
                    "-gencode=arch=compute_89,code=sm_89",   # RTX 40xx / H100
                    "-gencode=arch=compute_90,code=sm_90",   # H100 SXM
                ],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    name="scao_cuda",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
