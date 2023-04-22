import subprocess
from setuptools import setup, find_packages, Command

setup(
    name="fastllama",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "dataclasses",
        "cpuinfo",
        # Add any other dependencies your project needs here
    ],
    entry_points={
        "console_scripts": [
            "build_fastllama=build_python:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)