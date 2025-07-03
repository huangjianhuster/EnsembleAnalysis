from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Necessary to include files from MANIFEST.in
    package_data={
        "your_package_name": ["database/*"],  # Include files in the database/ folder
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple Python package with bundled data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_package_name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
