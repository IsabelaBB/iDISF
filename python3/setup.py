from distutils.core import setup, Extension
from glob import glob
import numpy


idisf_module = Extension("idisf",
                sources = ["iDISF_py3.c"],
                include_dirs = ["../include", numpy.get_include()],
                library_dirs = ["../lib"],
                libraries = ["idisf"],
                extra_compile_args = ["-O3", "-fopenmp"],
                extra_link_args=['-lgomp', '-lstdc++', "-fPIC", "-ffast-math", "-march=skylake", "-mfma"]
              );    

setup(name = "iDISF", 
      version = "1.0",
      author = "IB Barcelos",
      description = "Interative Segmentation based on Dynamic and Iterative Spanning Forest",
      ext_modules = [idisf_module]);
