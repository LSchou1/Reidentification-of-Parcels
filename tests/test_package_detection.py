"""Check output and input safety without downloading or running neural weights."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.detect_packages import find_images, main, save_detection


def test_only_direct_clean_images_are_selected(tmp_path):
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "movie.mov").touch()
    annotated = tmp_path / "annotely"
    annotated.mkdir()
    (annotated / "page_1.png").touch()
    assert find_images(tmp_path) == [tmp_path / "photo.jpg"]


def test_duplicate_stems_cannot_overwrite_outputs(tmp_path):
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "photo.png").touch()
    with pytest.raises(ValueError, match="different names"):
        find_images(tmp_path)


def test_drawing_preserves_source_and_prediction_coordinates(tmp_path):
    original = np.zeros((100, 120, 3), dtype=np.uint8)
    box = SimpleNamespace(xyxy=np.array([[10.2, 30.1, 90.9, 80.8]]), conf=[0.42])
    output = tmp_path / "detected.jpg"
    records = save_detection(original, SimpleNamespace(boxes=[box]), output)
    assert not original.any()
    assert cv2.imread(str(output)).any()
    assert records == [{"xyxy": [10.2, 30.1, 90.9, 80.8], "confidence": 0.42, "class": "package"}]


def test_no_detections_is_a_valid_result(tmp_path):
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    output = tmp_path / "empty.jpg"
    assert save_detection(image, SimpleNamespace(boxes=[]), output) == []
    assert output.is_file()


def test_output_cannot_overwrite_an_input(tmp_path):
    (tmp_path / "photo.jpg").touch()
    (tmp_path / "photo_detected.jpg").touch()
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path), "--output", str(tmp_path)])
    assert error.value.code == 1
