import sys

from predictmaint.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
