from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "Ranger13",
        ["Ranger13.cpp"],
        extra_compile_args=[
            "-std=c++17",
            "-fopenmp",
            "-O3",
            "-march=icelake-server",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512vl",
            "-mavx512dq",
            "-mavx512cd",
            "-mavx512vpopcntdq",
            "-mavx512bitalg",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="Ranger13",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
