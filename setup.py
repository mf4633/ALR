"""
Setup script for quantum_hydraulics package.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = "1.0.0"

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Quantum-inspired hydraulic simulation with vortex particle methods."

setup(
    name="quantum_hydraulics",
    version=version,
    author="Michael Flynn",
    author_email="",
    description="Physics-based hydraulic simulation with vortex particle methods and observation-dependent resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.5",
    ],
    extras_require={
        "pcswmm": ["pyswmm>=1.0", "pandas>=1.3"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-hydraulics=quantum_hydraulics.demos.engineering_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
