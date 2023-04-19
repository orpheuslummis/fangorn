from setuptools import find_packages, setup

package_dir = {"": "."}

package_data = {"": ["*"]}

install_requires = ["jax", "numpyro", "optax", "seaborn", "matplotlib", "arviz", "geojson_pydantic"]

setup_kwargs = {
    "name": "fangorn",
    "version": "0.1.0",
    "description": "The core engine of Digital Gaia.",
    "long_description": None,
    "author": "Digital Gaia, Inc",
    "author_email": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "package_dir": package_dir,
    "packages": find_packages(),
    "package_data": package_data,
    "python_requires": ">=3.10",
    "install_requires": install_requires,
}

setup(**setup_kwargs)
