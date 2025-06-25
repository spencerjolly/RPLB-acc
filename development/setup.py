from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

#python setup.py build_ext --inplace
setup(
    ext_modules = cythonize(
                  "cython_wrappers.pyx", 
                  compiler_directives={'language_level' : "3"}
                  ),
    include_dirs = [np.get_include()],
)