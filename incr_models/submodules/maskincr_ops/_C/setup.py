from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup
import glob, os

_cu_srcdir = './'
_cpp_srcdir = './'
_cudafiles = glob.glob("{}/*.cu".format(_cu_srcdir))
_cppfiles = glob.glob("{}/*.cpp".format(_cpp_srcdir))

_srcfiles = _cppfiles + _cudafiles

_header_dir = os.path.join(os.getcwd(), './include/')  # HACK: only works if setup.py is in the current working directory

print(_header_dir)
setup(
    name='incr_modules_C_pointops',
    ext_modules=[
        CUDAExtension(
            name='c_ops',
            sources=_srcfiles,
            include_dirs=[_header_dir],
            extra_compile_args={
                'cxx': ['-O3', '-ffast-math', '-DNDEBUG'],
                'nvcc': ['-O3' ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
