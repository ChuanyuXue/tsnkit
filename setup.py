from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Create __init__.py for the cython package if it doesn't exist
cython_dir = "tsnkit/simulation/cython"
init_file = os.path.join(cython_dir, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# Cython simulation extensions\n")

# Define Cython extensions
extensions = [
    Extension(
        "tsnkit.simulation.cython.simulation_core",
        ["tsnkit/simulation/cython/simulation_core.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/O2"] if os.name == "nt" else ["-O3", "-ffast-math"]
    )
]

setup(
    name="tsnkit-cython-extensions",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'cdivision': True,
    }),
    zip_safe=False,
)
