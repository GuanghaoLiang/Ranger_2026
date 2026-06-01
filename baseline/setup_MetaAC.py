from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "MetaAC",
        ["MetaAC.cpp"],
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="MetaAC",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
