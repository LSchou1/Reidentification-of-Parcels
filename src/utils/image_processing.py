"""Reusable image-processing operations."""

from __future__ import annotations

from pathlib import Path

import cv2


class ImageProcessingError(RuntimeError):
    """Raised when an input image cannot be processed or written."""


def process_image(input_path: Path, output_path: Path) -> Path:
    """Read an image, detect its edges, and write a three-channel result."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError(f"Could not read input image: {input_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(output_path), result)
    except (OSError, cv2.error) as error:
        raise ImageProcessingError(
            f"Could not write output image: {output_path}"
        ) from error

    if not written:
        raise ImageProcessingError(f"Could not write output image: {output_path}")

    return output_path
