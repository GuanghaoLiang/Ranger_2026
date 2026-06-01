from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "LG_EXP1",
        ["bfs_lg_exp1.cpp"],
        extra_compile_args=["-std=c++11", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="LG_EXP1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
