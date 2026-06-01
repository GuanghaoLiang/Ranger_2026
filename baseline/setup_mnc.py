from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mnc",
        ["mnc.cpp"],
        extra_compile_args=["-std=c++11", "-fopenmp"],  
        extra_link_args=["-fopenmp"],
         
    ),
]


setup(
    name="mnc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
    