"""Exercise real decoding and coordinate handling without downloading ML weights."""

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import zxingcpp

from src.detect_barcodes import (crop_barcode, decode_crop, decoding_variants, main,
                                 package_candidates, process_image, write_gallery)
from src import setup_barcode_model


def barcode_image(text="001234567890001234"):
    code = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.Code128)
    gray = np.asarray(zxingcpp.write_barcode_to_image(code, scale=3))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_real_code128_decode_retains_leading_zeros_and_source_coordinates():
    code = barcode_image()
    height, width = code.shape[:2]
    image = np.full((height + 200, width + 300, 3), 255, np.uint8)
    image[80:80 + height, 120:120 + width] = code
    original = image.copy()
    box = [120, 80, 120 + width, 80 + height]
    crop, coordinates = crop_barcode(image, box)
    decoded = decode_crop(crop, coordinates, box)
    assert len(decoded) == 1
    assert decoded[0]["text"] == "001234567890001234"
    assert decoded[0]["format"] == "Code128"
    assert all(120 <= p[0] <= 120 + width and 80 <= p[1] <= 80 + height
               for p in decoded[0]["polygon"])
    assert np.array_equal(image, original)


def test_decode_does_not_attach_neighbouring_barcode():
    first, second = barcode_image("FIRST"), barcode_image("SECOND")
    height, width = first.shape[:2]
    canvas = np.full((max(height, second.shape[0]) + 40, width + second.shape[1] + 100, 3), 255, np.uint8)
    canvas[20:20 + height, 20:20 + width] = first
    canvas[20:20 + second.shape[0], width + 70:width + 70 + second.shape[1]] = second
    codes = decode_crop(canvas, [0, 0, canvas.shape[1], canvas.shape[0]], [20, 20, 20 + width, 20 + height])
    assert [c["text"] for c in codes] == ["FIRST"]


def test_rotated_code128_can_be_decoded():
    image = cv2.rotate(barcode_image(), cv2.ROTATE_90_CLOCKWISE)
    box = [0, 0, image.shape[1], image.shape[0]]
    assert decode_crop(image, box, box)[0]["text"] == "001234567890001234"


def test_diagonal_decode_maps_coordinates_back_into_original_crop():
    original = barcode_image()
    _, rotated, _ = next(v for v in decoding_variants(original) if v[0] == 'rotation_+30')
    diagonal = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    height, width = diagonal.shape[:2]
    box = [40, 60, 40 + width, 60 + height]
    codes = decode_crop(diagonal, box, box)
    assert codes[0]["text"] == "001234567890001234"
    assert codes[0]["preprocessing"].startswith("rotation_")
    assert all(40 <= x <= 40 + width and 60 <= y <= 60 + height for x, y in codes[0]["polygon"])


def test_ml_detection_without_readable_code_is_preserved(tmp_path):
    image = np.full((100, 120, 3), 255, np.uint8)
    box = SimpleNamespace(xyxy=np.array([[0.0, 0.0, 70.0, 80.0]]), conf=[0.8], cls=[0])
    result = SimpleNamespace(boxes=[box], names={0: "barcode"})
    detections, preview = process_image(image, result, tmp_path, "blank")
    assert detections[0]["status"] == "detected_not_decoded"
    assert detections[0]["codes"] == []
    assert detections[0]["crop_xyxy"] == [0, 0, 86, 96]
    assert detections[0]["package_index"] is None
    assert (tmp_path / preview).is_file()
    assert np.all(image == 255)


def test_empty_ml_result_is_valid(tmp_path):
    result = SimpleNamespace(boxes=[], names={0: "barcode"})
    detections, preview = process_image(np.zeros((100, 100, 3), np.uint8), result, tmp_path, "empty")
    assert detections == []
    assert (tmp_path / preview).is_file()


def test_overlapping_packages_remain_ambiguous():
    packages = [{"xyxy": [0, 0, 100, 100]}, {"xyxy": [50, 50, 150, 150]}]
    assert package_candidates([65, 65, 85, 85], packages) == [0, 1]


def test_cli_rejects_input_inside_output_before_loading_weights(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"original source")
    with pytest.raises(SystemExit) as error:
        main(["--input", str(photo), "--output", str(tmp_path)])
    assert error.value.code == 1
    assert photo.read_bytes() == b"original source"


def test_existing_model_checksum_is_verified(tmp_path, monkeypatch):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"sha256": hashlib.sha256(b"expected").hexdigest()}))
    model = tmp_path / "model.pt"
    model.write_bytes(b"changed")
    monkeypatch.setattr(setup_barcode_model, "CONFIG", config)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        setup_barcode_model.download_weights(model)
    assert model.read_bytes() == b"changed"


def test_gallery_escapes_decoded_content(tmp_path):
    report = {"summary": {"images": 1, "ml_detections": 1, "decoded_detections": 1},
              "images": [{"image": "photo.jpg", "preview": "photo_barcodes.jpg", "detections": [
                  {"confidence": 0.9, "crop_file": "crops/1.png", "codes": [
                      {"format": "Code128", "text": "<script>alert(1)</script>"}]}]}]}
    write_gallery(report, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
