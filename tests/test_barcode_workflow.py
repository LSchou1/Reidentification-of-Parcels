"""Known-answer checks for annotation safety and object-detection evaluation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest
import yaml

from src.barcode_dataset import check_dataset, initialize, label_path, read_annotations, import_coco
from src.detect_barcodes import label_records, process_image
from src.evaluate_barcodes import evaluate, iou, join_predictions, predict_samples, main as evaluate_main
from src.train_barcodes import run_training


@pytest.fixture
def dataset(tmp_path):
    data = initialize(tmp_path / "dataset", with_labels=True)
    for number, split in enumerate(("train", "val", "test")):
        image = data.parent / "images" / split / "photo.png"
        assert cv2.imwrite(str(image), np.full((100, 120, 3), number * 80, np.uint8))
        label_path(image).write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.8 0.8\n")
    return data


def box(name="barcode", score=None, xyxy=None):
    record = {"class": name, "xyxy": xyxy or [0, 0, 20, 20]}
    if score is not None:
        record["confidence"] = score
    return record


def test_template_contains_no_automatic_negative_annotations(tmp_path):
    data = initialize(tmp_path / "new")
    assert yaml.safe_load(data.read_text())["names"] == {0: "barcode"}
    assert list(data.parent.rglob("*.txt")) == [data.parent / "README.txt"]
    with pytest.raises(ValueError, match="train split is empty"):
        check_dataset(data)


def test_missing_annotation_is_not_a_negative(dataset):
    annotation = dataset.parent / "labels/val/photo.txt"
    annotation.unlink()
    with pytest.raises(ValueError, match="Missing annotation"):
        check_dataset(dataset)
    annotation.write_text("")
    checked = check_dataset(dataset)
    assert checked["splits"]["val"][0]["annotations"] == []


@pytest.mark.parametrize("line", ["0.5 0.5 0.5 0.2 0.2", "0 0.05 0.5 0.2 0.2",
                                  "0 nan 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2",
                                  "0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 0.2 0.9"])
def test_invalid_or_unreviewed_prediction_format_is_rejected(tmp_path, line):
    path = tmp_path / "label.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="Invalid YOLO annotation"):
        read_annotations(path, 120, 100, ["barcode", "label"])


def test_duplicate_pixels_across_splits_are_rejected(dataset):
    target = dataset.parent / "images/val/photo.png"
    target.write_bytes((dataset.parent / "images/train/photo.png").read_bytes())
    with pytest.raises(ValueError, match="identical image pixels"):
        check_dataset(dataset)


def test_groups_cannot_leak_across_splits(dataset):
    groups = dataset.parent / "groups.csv"
    groups.write_text("image,group\nimages/train/photo.png,same_parcel\nimages/val/photo.png,same_parcel\nimages/test/photo.png,other\n")
    with pytest.raises(ValueError, match="occurs in multiple splits"):
        check_dataset(dataset)
    groups.write_text(groups.read_text().replace("images/val/photo.png,same_parcel", "images/val/photo.png,independent"))
    assert check_dataset(dataset)["groups_checked"]


def test_coco_import_maps_category_names_and_keeps_clean_images(tmp_path):
    data = initialize(tmp_path / "dataset", with_labels=True)
    images = tmp_path / "originals"
    images.mkdir()
    pixels = np.full((100, 120, 3), 170, np.uint8)
    cv2.imwrite(str(images / "photo.png"), pixels)
    source = {"images": [{"id": 7, "file_name": "photo.png", "width": 120, "height": 100}],
              "categories": [{"id": 5, "name": "Stregkode"}, {"id": 9, "name": "Fragtlabel"}],
              "annotations": [{"image_id": 7, "category_id": 5, "bbox": [48, 40, 24, 20]},
                              {"image_id": 7, "category_id": 9, "bbox": [12, 10, 96, 80]}]}
    annotations = tmp_path / "coco.json"
    annotations.write_text(json.dumps(source))
    result = import_coco(annotations, images, data, "val", "Stregkode", "Fragtlabel")
    assert result == {"split": "val", "images": 1, "objects": 2}
    copied = data.parent / "images/val/photo.png"
    assert copied.read_bytes() == (images / "photo.png").read_bytes()
    boxes = read_annotations(label_path(copied), 120, 100, ["barcode", "label"])
    assert boxes[0]["xyxy"] == pytest.approx([48, 40, 72, 60])
    with pytest.raises(ValueError, match="overwrite"):
        import_coco(annotations, images, data, "val", "Stregkode", "Fragtlabel")


def test_training_requires_positive_examples_of_every_configured_class(dataset, tmp_path):
    (dataset.parent / "labels/train/photo.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="no reviewed examples of class label"):
        run_training(training_args(dataset, tmp_path))


def test_duplicate_predictions_and_misses_have_known_metrics():
    samples = [{"image": "photo", "annotations": [box(), box(xyxy=[40, 40, 60, 60])],
                "predictions": [box(score=0.9), box(score=0.8)]}]
    result = evaluate(samples, ["barcode"])
    assert result["overall"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert result["confusion_matrix"] == [[1, 1], [1, 0]]
    assert result["map50"] == pytest.approx(51 / 101)


def test_iou_uses_intersection_over_union():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 0, 30, 10]) == 0


def test_iou_summaries_include_only_correct_matches_at_selected_thresholds():
    sample = {"image": "photo", "annotations": [box(), box("label", xyxy=[40, 40, 60, 60])],
              "predictions": [box(score=0.9, xyxy=[0, 0, 10, 20]),
                              box("label", 0.8, [40, 40, 55, 60]),
                              box(score=0.99, xyxy=[80, 80, 100, 100])]}
    result = evaluate([sample], ["barcode", "label"])
    assert result["iou_summary"]["overall"] == {"matches": 2, "mean": 0.625, "median": 0.625, "min": 0.5, "max": 0.75}
    assert result["iou_summary"]["per_class"]["barcode"]["mean"] == 0.5
    assert result["iou_summary"]["per_class"]["label"]["mean"] == 0.75
    assert result["images"][0]["match_details"][0]["iou"] == 0.5
    assert result["overall"]["fp"] == 1
    strict = evaluate([sample], ["barcode", "label"], overlap=0.8)
    assert strict["iou_summary"]["overall"]["mean"] is None
    assert strict["iou_summary"]["overall"]["matches"] == 0
    assert strict["overall"]["fn"] == 2
    assert "accuracy" not in result["overall"]


def test_wrong_class_is_visible_in_matrix_and_counts_as_fp_and_fn():
    samples = [{"image": "photo", "annotations": [box()], "predictions": [box("label", 0.9)]}]
    result = evaluate(samples, ["barcode", "label"])
    assert result["confusion_matrix"] == [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    assert result["overall"]["fp"] == result["overall"]["fn"] == 1
    assert result["average_precision"]["label"]["ap50"] is None
    assert result["map50"] == 0
    assert result["iou_summary"]["overall"]["mean"] is None


def test_higher_confidence_can_remove_correct_detections():
    samples = [{"image": "photo", "annotations": [box()],
                "predictions": [box(score=0.3), box(score=0.9, xyxy=[50, 50, 70, 70])]}]
    low = evaluate(samples, ["barcode"], confidence=0.25)
    high = evaluate(samples, ["barcode"], confidence=0.5)
    assert low["overall"]["recall"] == 1
    assert high["overall"]["recall"] == 0
    assert low["map50"] == high["map50"]  # AP uses all stored detections.
    with pytest.raises(ValueError, match="inference floor"):
        evaluate(samples, ["barcode"], confidence=0.1, floor=0.25)


def test_empty_images_have_no_invented_true_negative_count():
    result = evaluate([{"image": "empty", "annotations": [], "predictions": []}], ["barcode"])
    assert result["confusion_matrix"] == [[0, 0], [0, 0]]
    assert result["map50"] is None
    assert result["iou_summary"]["overall"] == {"matches": 0, "mean": None, "median": None, "min": None, "max": None}


def test_predictions_must_match_source_hash_and_cover_every_image(dataset):
    sample = check_dataset(dataset)["splits"]["val"][0]
    record = {k: sample[k] for k in ("image", "width", "height", "sha256")}
    report = {"images": [record | {"detections": []}]}
    assert join_predictions([sample], report, ["barcode", "label"])[0]["predictions"] == []
    record["sha256"] = "stale"
    with pytest.raises(ValueError, match="checksum"):
        join_predictions([sample], {"images": [record]}, ["barcode", "label"])
    with pytest.raises(ValueError, match="Missing predictions"):
        join_predictions([sample], {"images": []}, ["barcode", "label"])


def test_low_confidence_inference_handles_zero_area_boundary_proposals(dataset, tmp_path, monkeypatch):
    class Model:
        names = {0: "barcode"}
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            boxes = [SimpleNamespace(xyxy=np.array([[0, 0, 0, 20]]), conf=[0.002], cls=[0]),
                     SimpleNamespace(xyxy=np.array([[48, 40, 72, 60]]), conf=[0.9], cls=[0])]
            return [SimpleNamespace(boxes=boxes)]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"model")
    samples = check_dataset(dataset)["splits"]["val"]
    report = predict_samples(samples, weights, ["barcode", "label"], 0.001, 128, "cpu")
    assert report["settings"]["discarded_degenerate_predictions"] == 1
    assert len(report["images"][0]["detections"]) == 1
    assert len(join_predictions(samples, report, ["barcode", "label"])[0]["predictions"]) == 1


def training_args(dataset, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test checkpoint")
    return argparse.Namespace(data=dataset, weights=weights, output=tmp_path / "run", epochs=1,
                              imgsz=128, batch=1, device="cpu", patience=2, seed=42, dry_run=True)


def test_dry_run_checks_data_without_loading_model_or_writing_outputs(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    args = training_args(dataset, tmp_path)
    manifest = run_training(args)
    assert manifest["settings"]["epochs"] == 1
    assert not args.output.exists()


def test_training_uses_checked_data_and_keeps_test_out_of_model_selection(dataset, tmp_path, monkeypatch):
    calls = []
    class Model:
        def __init__(self, *args, **kwargs):
            pass
        def train(self, **kwargs):
            calls.append(kwargs)
            checkpoint = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trained checkpoint")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    args = training_args(dataset, tmp_path)
    args.dry_run = False
    run_training(args)
    resolved = yaml.safe_load((args.output / "resolved_data.yaml").read_text())
    assert "test" not in resolved
    assert Path(resolved["train"]).read_text().strip().endswith("images/train/photo.png")
    assert calls[0]["workers"] == 0 and calls[0]["seed"] == 42
    assert calls[0]["name"] != "model"  # Ultralytics treats this as a reserved automatic name.
    assert (args.output / "experiment.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        run_training(args)


def test_label_and_nested_barcode_are_preserved_and_linked(tmp_path):
    image = np.full((100, 120, 3), 255, np.uint8)
    label_box = SimpleNamespace(xyxy=np.array([[5, 5, 110, 95]]), conf=[0.7], cls=[1])
    labels = label_records(image, SimpleNamespace(boxes=[label_box]), tmp_path, "photo", 0.1)
    barcode_box = SimpleNamespace(xyxy=np.array([[30, 30, 80, 60]]), conf=[0.8], cls=[0])
    detections, _ = process_image(image, SimpleNamespace(boxes=[barcode_box], names={0: "barcode"}),
                                  tmp_path, "photo", labels=labels)
    assert detections[0]["label_index"] == 0
    assert detections[0]["label_candidates"] == [0]
    assert detections[0]["codes"] == []
    assert labels[0]["class"] == "label"
    assert (tmp_path / labels[0]["crop_file"]).is_file()


def test_evaluation_cli_writes_a_report_from_reviewed_data(dataset, tmp_path):
    sample = check_dataset(dataset)["splits"]["val"][0]
    record = {k: sample[k] for k in ("image", "width", "height", "sha256")}
    record["detections"] = [box(score=0.9, xyxy=[48, 40, 72, 60])]
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"settings": {"confidence": 0.001}, "images": [record]}))
    output = tmp_path / "evaluation"
    assert evaluate_main(["--data", str(dataset), "--predictions", str(predictions), "--output", str(output)]) == 0
    result = json.loads((output / "evaluation.json").read_text())
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fn"] == 1  # The whole label was not detected.
    assert (output / "confusion_matrix.png").is_file()
    assert (output / "iou_distribution.png").is_file()
    assert result["iou_summary"]["overall"]["mean"] == 1.0
    assert "Gns. IoU" in (output / "index.html").read_text()
    assert "Background" in (output / "index.html").read_text()
