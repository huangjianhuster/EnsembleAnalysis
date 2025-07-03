from setuptools import setup, find_packages

setup(
    name="EnsembleAnalysis",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Necessary to include files from MANIFEST.in
    package_data={
        "EnsembleAnalysis": ["database/*"],  # Include files in the database/ folder
    },
    author="Jian Huang",
    author_email="huangjianhuster@gmail.com",
    description="A simple Python package for analyzing conformational ensembles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huangjianhuster/EnsembleAnalysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.2",
        "MDAnalysis>=2.8.0",
        "MDtraj>=1.11.0",
        "biopython>=1.85",
        "scipy>=1.11.4",
        "matplotlib>=3.10.0"
        ],
)
