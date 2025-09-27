from setuptools import setup, Extension
import os

try:
    # Cython is optional when building from an sdist that includes generated C
    from Cython.Build import cythonize  # type: ignore
    HAVE_CYTHON = True
except Exception:
    HAVE_CYTHON = False

import numpy

# Create __init__.py for the cython package if it doesn't exist
cython_dir = "tsnkit/simulation/cython"
init_file = os.path.join(cython_dir, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# Cython simulation extensions\n")

src_pyx = "tsnkit/simulation/cython/simulation_core.pyx"
src_c = "tsnkit/simulation/cython/simulation_core.c"
source = src_pyx if os.path.exists(src_pyx) else src_c

# Define Cython/C extensions
extensions = [
    Extension(
        "tsnkit.simulation.cython.simulation_core",
        [source],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/O2"] if os.name == "nt" else ["-O3", "-ffast-math"],
    )
]

setup(
    name="tsnkit-cython-extensions",
    ext_modules=(
        cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "initializedcheck": False,
                "cdivision": True,
            },
        )
        if HAVE_CYTHON and source.endswith(".pyx")
        else extensions
    ),
    zip_safe=False,
)
