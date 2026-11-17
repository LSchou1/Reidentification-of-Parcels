"""Exercise real data boundaries and the complete lifecycle without expensive ML."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import workflow
from src.barcode_dataset import label_path, read_annotations
from src.workflow_data import dataset_directory, frozen_dataset, prepare_photos, related_groups, save_json, split_photos
import src.workflow_reports as reports


CLASSES = ["barcode", "label", "parcel"]
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
BOXES = "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.5 0.5\n2 0.5 0.5 0.9 0.9\n"


@pytest.fixture
def batch(tmp_path):
    source, project = tmp_path / "raw/batch", tmp_path / "annotated/batch"
    for group in range(10):
        for frame in range(2):
            image = source / f"physical_{group}" / f"frame_{frame}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            # Similar-looking, independently generated photos must stay independent.
            pixels = np.full((80, 100, 3), 90 + group * 2 + frame, np.uint8)
            cv2.imwrite(str(image), pixels)
    prepare_photos(source, project, CLASSES)
    for image in (project / "images").rglob("*.png"):
        label_path(image).write_text(BOXES)
    return source, project, tmp_path / "output/batch"


def test_prepare_keeps_independent_clean_copies_and_no_fake_annotations(tmp_path):
    source, project = tmp_path / "input", tmp_path / "project"
    source.mkdir()
    image = source / "photo.png"
    cv2.imwrite(str(image), np.full((10, 12, 3), 80, np.uint8))
    prepare_photos(source, project, CLASSES)
    output = tmp_path / "output"
    copied = project / "images/photo.png"
    assert copied.read_bytes() == image.read_bytes()
    assert copied.stat().st_ino != image.stat().st_ino
    assert not label_path(copied).exists()
    with pytest.raises(ValueError, match="Missing group"):
        split_photos(project, output, RATIOS, 42, CLASSES)


def test_transitive_parcel_ids_and_duplicate_pixels_are_kept_together(tmp_path):
    records = [{"image": name, "pixel_sha256": fingerprint} for name, fingerprint in
               [("a", "a"), ("both", "b"), ("b", "c"), ("copy", "a"), ("lookalike", "d")]]
    groups = tmp_path / "groups.csv"
    groups.write_text("image,group\na,parcel_A\nboth,parcel_A|parcel_B\nb,parcel_B\ncopy,other\nlookalike,different_parcel\n")
    components = related_groups(records, groups)
    assert sorted(map(sorted, components)) == [["a", "b", "both", "copy"], ["lookalike"]]


def test_group_split_and_class_toggle_preserve_assignments_and_master_boxes(batch):
    source, project, output = batch
    originals = {p: p.read_bytes() for p in source.rglob("*.png")}
    split_photos(project, output, RATIOS, 42, CLASSES)
    all_data, _ = frozen_dataset(output, CLASSES)
    assigned = {sample["path"].parts[-2]: split for split, samples in all_data["splits"].items() for sample in samples}
    assert len(assigned) == 10
    assert all(len({split for split, samples in all_data["splits"].items() for s in samples if s["path"].parts[-2] == group}) == 1 for group in assigned)
    plan = (output / "split_plan.json").read_bytes()
    # Remove label and reorder the remaining classes: output IDs must be remapped.
    active = ["parcel", "barcode"]
    split_photos(project, output, RATIOS, 42, active)
    view, _ = frozen_dataset(output, active)
    assert view["names"] == active
    for split, samples in view["splits"].items():
        for sample in samples:
            assert assigned[sample["path"].parts[-2]] == split
            annotations = read_annotations(label_path(sample["path"]), 100, 80, active)
            assert [a["class"] for a in annotations] == ["barcode", "parcel"]
            assert label_path(sample["path"]).read_text().splitlines()[0].startswith("1 ")
    assert (output / "split_plan.json").read_bytes() == plan
    assert all(label_path(p).read_text() == BOXES for p in (project / "images").rglob("*.png"))
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_frozen_split_rejects_annotation_edits_and_seed_changes(batch):
    _, project, output = batch
    split_photos(project, output, RATIOS, 42, CLASSES)
    with pytest.raises(ValueError, match="frozen"):
        split_photos(project, output, RATIOS, 10, CLASSES)
    data, _ = frozen_dataset(output, CLASSES)
    label_path(data["splits"]["train"][0]["path"]).write_text("")
    with pytest.raises(ValueError, match="Frozen image or annotation changed"):
        frozen_dataset(output, CLASSES)


def test_single_connected_collection_cannot_be_randomly_split(batch):
    _, project, output = batch
    groups = project / "groups.csv"
    rows = list(csv.DictReader(groups.open()))
    with groups.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "group"])
        writer.writerows((row["image"], "one_physical_parcel") for row in rows)
    with pytest.raises(ValueError, match="at least 3 independent"):
        split_photos(project, output, RATIOS, 42, CLASSES)
    assert not dataset_directory(output, CLASSES).exists()


def test_missing_annotation_cannot_be_treated_as_a_negative(batch):
    _, project, output = batch
    next((project / "labels").rglob("*.txt")).unlink()
    with pytest.raises(ValueError, match="Missing annotation"):
        split_photos(project, output, RATIOS, 42, CLASSES)


def configuration(tmp_path, project, output):
    config = yaml.safe_load(Path("workflow.yaml").read_text())
    config["annotations"] = str(project)
    config["output"] = str(output)
    for key in ("barcode_weights", "world_weights"):
        weights = tmp_path / f"{key}.pt"
        weights.write_bytes(key.encode())
        config[key] = str(weights)
    return config


def test_choose_confidence_uses_all_scores_and_can_keep_a_simple_tied_value():
    samples = [{"image": "a", "annotations": [{"class": "barcode", "xyxy": [0, 0, 10, 10]}],
                "predictions": [{"class": "barcode", "xyxy": [0, 0, 10, 10], "confidence": 0.5321},
                                {"class": "barcode", "xyxy": [30, 30, 40, 40], "confidence": 0.3142}]}]
    chosen, rows = reports.choose_confidence(samples, ["barcode"], 0.001, 0.5)
    assert chosen == 0.35
    assert 0.3142 in [row["confidence"] for row in rows]
    assert max(row["f1"] for row in rows) == 1


def test_full_lifecycle_freezes_val_confidence_and_rejects_stale_comparisons(batch, tmp_path, monkeypatch):
    _, project, output = batch
    split_photos(project, output, RATIOS, 42, CLASSES)
    config = configuration(tmp_path, project, output)
    root = workflow.run_directory(config)
    seen = []

    def predict(samples, names, config, kind, trained_weights):
        seen.append([s["image"] for s in samples])
        return {"model": {"kind": kind, "weights": reports.model_files(config, kind, trained_weights)},
                "settings": {"confidence": config["prediction_floor"]},
                "images": [{**{key: s[key] for key in ("image", "sha256", "width", "height")},
                            # Test scores differ: using test to retune would change the threshold.
                            "detections": [{**a, "confidence": 0.72 if '/val/' in s['image'] else 0.11} for a in s["annotations"]]}
                           for s in samples]}

    def write(result, samples, output):
        save_json(output / "evaluation.json", result)
        (output / "index.html").write_text("test report")

    monkeypatch.setattr(reports, "predict_boxes", predict)
    monkeypatch.setattr(reports, "write_report", write)
    with pytest.raises(ValueError, match="Run train"):
        workflow.test_models(config)
    baseline = reports.run_analysis(config, root, "baseline", "val")
    assert all('/val/' in image for image in seen[-1])
    # Mock only the expensive optimizer; use the real workflow/trainer boundary.
    def training(args):
        checkpoint = args.output / "detector/weights/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"trained")
        return {"trained_weights": {"sha256": workflow.sha256(checkpoint)}}
    monkeypatch.setattr(workflow, "run_training", training)
    workflow.train(config)
    weights = workflow.require_trained_model(config)
    reports.run_analysis(config, root, "finetuned", "val", weights)
    comparison = workflow.test_models(config)
    assert comparison.is_file()
    assert (root / "baseline_val/metrics.csv").is_file()
    links = json.loads((root / "baseline_val/associations.json").read_text())["objects"]
    barcode_link = next(link for link in links if link["class"] == "barcode")
    assert barcode_link["label_index"] == 1 and barcode_link["parcel_index"] == 2
    test = json.loads((root / "baseline_test/evaluation.json").read_text())
    assert test["settings"]["confidence"] == baseline["settings"]["confidence"] == 0.35
    assert test["overall"]["tp"] == 0  # No test-driven threshold rescue.
    assert len(seen) == 4
    workflow.test_models(config)
    assert len(seen) == 4  # Completed tests are reused, not silently rerun/tuned.
    altered = deepcopy(test)
    altered["image_hashes"][next(iter(altered["image_hashes"]))] = "different"
    with pytest.raises(ValueError, match="different image_hashes"):
        reports.compatible_results(test, altered)
    changed = deepcopy(config)
    changed["iou"] = 0.6
    with pytest.raises(ValueError, match="validation model/setting changed"):
        workflow.test_models(changed)
    exported = json.loads((comparison.parent / "comparison.json").read_text())
    assert all("accuracy" not in row for row in exported["rows"])
