"""Command-line entrypoint for the machine-vision application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.utils.image_processing import ImageProcessingError, process_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process an image with the bachelor machine-vision pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source image, for example data/images/input.jpg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the processed image, for example data/output/edges.png.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = process_image(args.input, args.output)
    except ImageProcessingError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Processed image written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
