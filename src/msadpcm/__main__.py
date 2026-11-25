import sys

from .cli import main

if __name__ == "__main__":  # the guard matters: worker processes import this module
    sys.exit(main())
