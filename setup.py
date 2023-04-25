import subprocess
import sys
import os
from setuptools import setup, find_packages, Command
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        # Install required packages before running compile.py
        self.distribution.install_requires = [
            "numpy==1.24.2",
            "sentencepiece==0.1.97",
            "torch==2.0.0",
            "py-cpuinfo==9.0.0",
            "inquirer==3.1.3"
        ]
        install.run(self)

        # Run compile.py after the package is installed
        if not os.path.exists("compile.py"):
            print("Error: compile.py not found in the current directory", file=sys.stderr)
            sys.exit(1)

        return_code = os.system("python compile.py -l python > build_logs.txt")

        if return_code != 0:
            print("Error: compile.py execution failed", file=sys.stderr)
            sys.exit(1)

setup(
    name="fastllama",
    version="0.5",
    package_dir={"fastLLaMa": "fastLLaMa"},
    packages=["fastLLaMa"],
    install_requires=[
        "numpy==1.24.2",
        "sentencepiece==0.1.97",
        "torch==2.0.0",
        "py-cpuinfo==9.0.0",
        "inquirer==3.1.3"
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