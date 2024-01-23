import os
import re
import subprocess
from setuptools import setup, find_packages


_mydir = os.path.dirname(__file__)

# parse version number from airglowlut/__init__.py
version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
with open(os.path.join(_mydir, "airglowlut", "__init__.py")) as f:
    content = f.read()
re_search = re.search(version_re, content, re.M)
if re_search:
    version_str = re_search.group(1)
else:
    raise RuntimeError("Could not parse version string from __init__.py")

# add the git hash to version_str
if os.path.exists(os.path.join(_mydir, ".git")):
    git_hash = subprocess.check_output(
        "git rev-parse --verify --short HEAD", cwd=_mydir, text=True, shell=True
    ).strip()
    git_version_str = f"v{version_str}"
    tags = subprocess.check_output("git tag", cwd=_mydir, text=True, shell=True)
    if git_version_str not in tags:
        subprocess.check_output(
            f"git tag -a {git_version_str} {git_hash} -m 'tagged by setup.py to {version_str}'",
            cwd=_mydir,
            text=True,
            shell=True,
        )
    version_str = f"{version_str}+git.{git_hash}"

setup(
    name="airglowlut",
    description="Generate a lookup table and parametrization for oxygen airglow density profiles using data from https://doi.org/10.7910/DVN/T1WRWQ",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    version=version_str,
    url="https://github.com/rocheseb/oxygen_airglow_lut",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "dask",
        "netcdf4",
    ],
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires="~=3.10",
    entry_points={
        "console_scripts": [
            "airglowlut=airglowlut.airglow_lut:main",
        ],
    },
)
