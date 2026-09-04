from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.image_processing import ImageProcessingError, process_image


def write_test_image(path: Path) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (100, 60), (255, 255, 255), thickness=-1)
    assert cv2.imwrite(str(path), image)


def test_process_image_writes_edge_image(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output" / "edges.png"
    write_test_image(input_path)

    returned_path = process_image(input_path, output_path)

    result = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert returned_path == output_path
    assert result is not None
    assert result.shape == (80, 120, 3)
    assert np.any(result > 0)


def test_process_image_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ImageProcessingError, match="Could not read input image"):
        process_image(tmp_path / "missing.png", tmp_path / "output.png")


def test_process_image_rejects_invalid_image(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.png"
    input_path.write_text("not an image", encoding="ascii")

    with pytest.raises(ImageProcessingError, match="Could not read input image"):
        process_image(input_path, tmp_path / "output.png")


def test_process_image_rejects_unsupported_output_format(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    write_test_image(input_path)

    with pytest.raises(ImageProcessingError, match="Could not write output image"):
        process_image(input_path, tmp_path / "output.unsupported")
