"""The console-script entry point: parse, configure logging, dispatch.

Named `entry` and not `main` so that `from autogeoref.cli import main` fails at
import instead of quietly binding a module nobody can call.

autogeoref run <volume> --city <city-config> --work work
autogeoref status                                  # what is REALLY placed
autogeoref report <volume> --work work
autogeoref review --city <city-config> --work work [--apply]
"""

from __future__ import annotations

import logging
import sys

from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
