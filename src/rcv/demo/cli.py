"""Print a UTF-8 replay of the five published chain records."""
from __future__ import annotations

import argparse
import os
import sys

from rcv.demo.records import load_chains
from rcv.demo.transcript import render


def _force_utf8(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        raise RuntimeError("stdout does not support UTF-8 reconfiguration.")
    reconfigure(encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> int:
    _force_utf8(sys.stdout)
    try:
        for line in render(load_chains()):
            print(line)
        # Catch buffered-write failures here, before interpreter shutdown.
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rcv demo", description=__doc__.splitlines()[0])
    return run(parser.parse_args(argv))
