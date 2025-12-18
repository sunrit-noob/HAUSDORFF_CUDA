import os
from setuptools import setup, find_packages
from hausdorffCuda import __version__

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="hausdorff_cuda",
    version=__version__,
    author="Sunrit Sarkar",
    author_email="sunrit2022pers@gmail.com",
    description="A PyTorch extension for computing Hausdorff Distance using CUDA.",
    packages=find_packages(),
    package_data={
        "hausdorffCuda": ["src/hausdorff.cpp", "src/hausdorff.cu"]
    },
    install_requires=requirements
)