import argparse
import subprocess

def main():
    subprocess.run(["python", "setup.py", "-l", "python"], check=True)

if __name__ == "__main__":
    main()