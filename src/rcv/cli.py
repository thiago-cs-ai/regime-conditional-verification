"""Replay the five published Regime-Conditional Verification chains."""
from __future__ import annotations

import argparse

from rcv.demo import cli as demo_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rcv", description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    demo = subcommands.add_parser("demo", help="replay the five published RCV chains")
    demo.set_defaults(handler=demo_cli.run)

    args = parser.parse_args(argv)
    return args.handler(args)
