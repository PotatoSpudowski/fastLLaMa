import subprocess
import sys
import os
import shutil
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
import site

import importlib


class CustomInstallCommand(install):

    def run(self):
        # Install required packages before running compile.py
        self.distribution.install_requires = [
            "numpy>=1.24.2",
            "py-cpuinfo>=9.0.0",
            "inquirer>=3.1.3",
            "cmake>=3.20.2"
        ]

        # Explicitly install the required packages using subprocess
        for package in self.distribution.install_requires:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # install.run(self)

        # Run compile.py after the package is installed
        if not os.path.exists("compile.py"):
            print("Error: compile.py not found in the current directory", file=sys.stderr)
            sys.exit(1)

        print("Current working directory:", os.getcwd())

        # Dynamically import the 'compile' module
        compile_module = importlib.import_module("compile")
        compile_main = getattr(compile_module, "main")

        compile_main(["-l", "python"])

        # Copy the .so file to the fastLLaMa folder in site-packages
        site_packages_dir = site.getsitepackages()[0]
        fastllama_dir = os.path.join(site_packages_dir, "fastLLaMa")
        shutil.copy("build/interfaces/python/pyfastllama.so", fastllama_dir)

setup(
    name="fastllama",
    version="0.5",
    package_dir={"fastLLaMa": "fastLLaMa"},
    packages=["fastLLaMa"],
    install_requires=[
        "numpy>=1.24.2",
        "py-cpuinfo>=9.0.0",
        "inquirer>=3.1.3",
        "cmake>=3.20.2"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)