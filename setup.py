from setuptools import find_packages, setup

package_dir = {"": "."}

package_data = {"": ["*"]}

install_requires = [
    "meteostat~=1.6.5",
    "numpyro~=0.10.1",
    "jaxlib~=0.4.4",
    "jax~=0.4.4",
    "geopandas~=0.12.1",
    "shapely~=1.8.5.post1",
    "geojson-pydantic~=0.4.3",
    "rtree~=1.0.1",
    "pandas~=1.5.2",
    "funsor~=0.4.3",
    "mpu~=0.23.1",
    "numpy~=1.23.5",
    "matplotlib~=3.6.2",
    "joblib~=1.2.0",
    "devtools~=0.10.0",
    "indexed~=1.3.0",
    "scipy~=1.9.3",
    "coolname~=2.0.0",
    "optax~=0.1.4",
    "pydantic~=1.10.2",
    "ipykernel~=6.20.1",
    "jupyterlab~=3.6.1",
    "jaxns~=2.0.0",
]
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
