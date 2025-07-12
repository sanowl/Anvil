#!/usr/bin/env python3
"""
Setup script for Anvil Framework Python bindings
"""

from setuptools import setup, find_packages
from pyo3_build import build

setup(
    name="anvil-ml",
    version="0.1.0",
    description="Advanced Rust-based Machine Learning Framework",
    author="Anvil Team",
    author_email="team@anvil-ml.com",
    url="https://github.com/anvil-ml/anvil",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.15",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    cmdclass={"build_ext": build},
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
) 