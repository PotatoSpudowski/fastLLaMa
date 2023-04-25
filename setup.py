import subprocess
import sys
import os
from setuptools import setup, find_packages, Command

def run_build_python():
    if not os.path.exists("build.py"):
        print("Error: build.py not found in the current directory", file=sys.stderr)
        sys.exit(1)

    return_code = os.system("python build.py -l python > build_logs.txt")

    if return_code != 0:
        print("Error: build.py execution failed", file=sys.stderr)
        sys.exit(1)

setup(
    name="fastllama",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.2",
        "sentencepiece==0.1.97",
        "torch==2.0.0",
        "py-cpuinfo==9.0.0",
        "inquirer==3.1.3"
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

# Run build_python.py after the package is installed
run_build_python()