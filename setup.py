import subprocess
import sys
import os
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

sys.path.insert(0, os.path.abspath("."))

from compile import main as compile_main


class CustomBuildExtCommand(build_ext):
    def run(self):
        # Run compile.py after the package is installed
        # if not os.path.exists("compile.py"):
        #     print("Error: compile.py not found in the current directory", file=sys.stderr)
        #     sys.exit(1)

        # print("Current working directory:", os.getcwd())

        compile_main(["-l", "python"])

        super().run()


class CustomInstallCommand(install):
    def run(self):
        # Install required packages before running compile.py
        # self.distribution.install_requires = [
        #     "numpy>=1.24.2",
        #     "py-cpuinfo>=9.0.0",
        #     "inquirer>=3.1.3",
        #     "cmake>=3.20.2"
        # ]

        # Explicitly install the required packages using subprocess
        # for package in self.distribution.install_requires:
        #     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Call the CustomBuildExtCommand to build the extension
        build_ext_cmd = self.get_finalized_command("build_ext")
        build_ext_cmd.run()

        super().run()

description = """
fastLLaMa is an experimental high-performance framework for running Decoder-only LLMs with 4-bit quantization in Python using a C/C++ backend.
"""  
setup(
    name="fastllama",
    version="1.0.0",
    package_dir={"fastllama": "examples/python/fastllama"},
    packages=["fastllama"],
    package_data={"fastllama": ["pyfastllama.so"]},
    author="PotatoSpudowski, Amit Singh",
    author_email="bahushruth.cs@gmail.com, amitsingh19975@gmail.com",
    description=description,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    cmdclass={
        'build_ext': CustomBuildExtCommand,
        'install': CustomInstallCommand,
    },
    requires=["setuptools", "wheel", "py-cpuinfo>=9.0.0", "cmake>=3.20.2"]
)