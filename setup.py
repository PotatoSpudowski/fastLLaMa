from pathlib import Path
import sys
import os
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import shutil
from site import getsitepackages

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

        source_path = Path(os.path.dirname(os.path.abspath(__file__)))/ "examples" / "python" / "fastllama"
        dest_path = Path(getsitepackages()[0])/ "fastllama"
        dest_path.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            raise Exception("Project did not compile correctly.")
        
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

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

setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
        'install': CustomInstallCommand,
    },
)