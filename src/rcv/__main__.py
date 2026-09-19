"""`python -m rcv` -- the same entry point as the `rcv` command."""
from __future__ import annotations

import sys

from rcv.cli import main

if __name__ == "__main__":
    sys.exit(main())
