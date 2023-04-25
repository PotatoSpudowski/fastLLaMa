import argparse
import subprocess
import sys
import os

def main():
    # Check if build.py exists in the current directory
    if not os.path.exists("build.py"):
        print("Error: build.py not found in the current directory")
        sys.exit(1)

    # Run build.py using subprocess
    result = subprocess.run(["python", "build.py", "-l", "python"], capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error: build.py execution failed")
        print("Output:")
        print(result.stdout)
        print("Error message:")
        print(result.stderr)
        sys.exit(1)

    # Print the output of build.py
    print("build.py output:")
    print(result.stdout)

if __name__ == "__main__":
    main()