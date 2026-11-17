"""Exercise real data boundaries and the complete lifecycle without expensive ML."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil

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
    prepare_photos(source, project, tmp_path / "tmp/batch", CLASSES)
    coordinates = json.loads((project / "annotations.json").read_text())
    for record in coordinates.values():
        record.update(reviewed=True, barcode=[[40, 32, 60, 48]],
                      label=[[25, 20, 75, 60]], parcel=[[5, 4, 95, 76]])
    save_json(project / "annotations.json", coordinates)
    return source, project, tmp_path / "tmp/batch", tmp_path / "output/batch"


def test_prepare_keeps_independent_clean_copies_and_no_fake_annotations(tmp_path):
    source, project = tmp_path / "input", tmp_path / "project"
    source.mkdir()
    image = source / "photo.png"
    cv2.imwrite(str(image), np.full((10, 12, 3), 80, np.uint8))
    prepare_photos(source, project, tmp_path / "tmp/batch", CLASSES)
    output = tmp_path / "output/batch"
    temporary = tmp_path / "tmp/batch"
    assert sorted(p.name for p in project.iterdir()) == ["annotations.json"]
    assert image.exists()
    assert json.loads((project / "annotations.json").read_text())["photo.png"]["reviewed"] is False
    with pytest.raises(ValueError, match="Missing group"):
        split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)


def test_transitive_parcel_ids_and_duplicate_pixels_are_kept_together(tmp_path):
    records = [{"image": name, "pixel_sha256": fingerprint} for name, fingerprint in
               [("a", "a"), ("both", "b"), ("b", "c"), ("copy", "a"), ("lookalike", "d")]]
    groups = tmp_path / "groups.csv"
    groups.write_text("image,group\na,parcel_A\nboth,parcel_A|parcel_B\nb,parcel_B\ncopy,other\nlookalike,different_parcel\n")
    components = related_groups(records, list(csv.DictReader(groups.open())))
    assert sorted(map(sorted, components)) == [["a", "b", "both", "copy"], ["lookalike"]]


def test_group_split_and_class_toggle_preserve_assignments_and_master_boxes(batch):
    source, project, temporary, output = batch
    originals = {p: p.read_bytes() for p in source.rglob("*.png")}
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    all_data, _ = frozen_dataset(temporary, CLASSES)
    assigned = {sample["path"].parts[-2]: split for split, samples in all_data["splits"].items() for sample in samples}
    assert len(assigned) == 10
    assert all(len({split for split, samples in all_data["splits"].items() for s in samples if s["path"].parts[-2] == group}) == 1 for group in assigned)
    plan = (output / "dataset_split.json").read_bytes()
    # Remove label and reorder the remaining classes: output IDs must be remapped.
    active = ["parcel", "barcode"]
    split_photos(source, project, temporary, output, RATIOS, 42, active)
    view, _ = frozen_dataset(temporary, active)
    assert view["names"] == active
    for split, samples in view["splits"].items():
        for sample in samples:
            assert assigned[sample["path"].parts[-2]] == split
            annotations = read_annotations(label_path(sample["path"]), 100, 80, active)
            assert [a["class"] for a in annotations] == ["barcode", "parcel"]
            assert label_path(sample["path"]).read_text().splitlines()[0].startswith("1 ")
    assert (output / "dataset_split.json").read_bytes() == plan
    assert all(r["parcel"] == [[5, 4, 95, 76]] for r in json.loads((project / "annotations.json").read_text()).values())
    assert all(p.is_file() and (p.suffix == ".png" or p.name == "annotations.json") for p in project.iterdir())
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_frozen_split_rejects_annotation_edits_and_seed_changes(batch):
    source, project, temporary, output = batch
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    with pytest.raises(ValueError, match="frozen"):
        split_photos(source, project, temporary, output, RATIOS, 10, CLASSES)
    data, _ = frozen_dataset(temporary, CLASSES)
    label_path(data["splits"]["train"][0]["path"]).write_text("")
    with pytest.raises(ValueError, match="Frozen image or annotation changed"):
        frozen_dataset(temporary, CLASSES)


def test_single_connected_collection_cannot_be_randomly_split(batch):
    source, project, temporary, output = batch
    path = project / "annotations.json"
    coordinates = json.loads(path.read_text())
    for record in coordinates.values():
        record["group"] = "one_physical_parcel"
    save_json(path, coordinates)
    with pytest.raises(ValueError, match="at least 3 independent"):
        split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    assert not dataset_directory(temporary, CLASSES).exists()


def test_missing_annotation_cannot_be_treated_as_a_negative(batch):
    source, project, temporary, output = batch
    path = project / "annotations.json"
    coordinates = json.loads(path.read_text())
    next(iter(coordinates.values()))["reviewed"] = False
    save_json(path, coordinates)
    with pytest.raises(ValueError, match="Missing annotation"):
        split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)


def configuration(tmp_path, project, temporary, output):
    config = yaml.safe_load(Path("workflow.yaml").read_text())
    config["annotations"] = str(project)
    config["output"] = str(output)
    config["temporary"] = str(temporary)
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
    source, project, temporary, output = batch
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    config = configuration(tmp_path, project, temporary, output)
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


def test_deleting_tmp_rebuilds_identical_split_without_changing_json(batch):
    source, project, temporary, output = batch
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    _, before_hash = frozen_dataset(temporary, CLASSES)
    source_json = (project / "annotations.json").read_bytes()
    permanent_split = (output / "dataset_split.json").read_bytes()
    shutil.rmtree(temporary)
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    assert frozen_dataset(temporary, CLASSES)[1] == before_hash
    assert (project / "annotations.json").read_bytes() == source_json
    assert (output / "dataset_split.json").read_bytes() == permanent_split


def test_json_edits_cannot_silently_replace_frozen_benchmark(batch):
    source, project, temporary, output = batch
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    path = project / "annotations.json"
    coordinates = json.loads(path.read_text())
    next(iter(coordinates.values()))["barcode"] = [[10, 10, 30, 30]]
    save_json(path, coordinates)
    with pytest.raises(ValueError, match="frozen"):
        split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)


@pytest.mark.parametrize("change", ["invalid_box", "missing_class"])
def test_json_coordinates_need_valid_complete_boxes(batch, change):
    source, project, temporary, output = batch
    path = project / "annotations.json"
    coordinates = json.loads(path.read_text())
    record = next(iter(coordinates.values()))
    if change == "invalid_box":
        record["parcel"] = [[-1, 0, 200, 200]]
    else:
        del record["parcel"]
    save_json(path, coordinates)
    with pytest.raises(ValueError, match="Invalid|Missing annotation"):
        split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    assert not dataset_directory(temporary, CLASSES).exists()


def test_training_uses_clean_raw_pixels_instead_of_boxed_previews(batch):
    source, project, temporary, output = batch
    preview = project / "physical_0__frame_0.jpg"
    cv2.imwrite(str(preview), np.zeros((80, 100, 3), np.uint8))
    split_photos(source, project, temporary, output, RATIOS, 42, CLASSES)
    dataset, _ = frozen_dataset(temporary, CLASSES)
    for samples in dataset["splits"].values():
        for sample in samples:
            relative = Path(*Path(sample["image"]).parts[2:])
            assert sample["path"].read_bytes() == (source / relative).read_bytes()
    assert preview.is_file()


def test_gallery_rebuilds_from_json_and_respects_phone_orientation(tmp_path):
    from PIL import Image
    from src.export_annotation_gallery import main as gallery

    raw, annotated, temporary = (tmp_path / name for name in ('raw', 'annotated', 'tmp'))
    raw.mkdir()
    # Stored pixels are portrait; EXIF rotates them to 100x80 for viewers/OpenCV.
    photo = raw / 'phone.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (80, 100), 'white').save(photo, exif=exif)
    record = {'width': 100, 'height': 80, 'group': 'parcel_A', 'reviewed': True,
              'barcode': [[10, 10, 40, 30]], 'label': [], 'parcel': [[0, 0, 100, 80]]}
    save_json(annotated / 'annotations.json', {'phone.jpg': record})
    config = yaml.safe_load(Path('workflow.yaml').read_text())
    config.update(input=str(raw), annotations=str(annotated), temporary=str(temporary),
                  output=str(tmp_path / 'output'))
    config_path = tmp_path / 'workflow.yaml'
    config_path.write_text(yaml.safe_dump(config))
    original, coordinates = photo.read_bytes(), (annotated / 'annotations.json').read_bytes()
    for attempt in range(2):
        gallery(['--config', str(config_path)])
        with Image.open(annotated / 'phone.jpg') as preview:
            assert preview.size == (100, 80)
        assert (temporary / 'annotation_review/index.html').exists()
        if attempt == 0:
            shutil.rmtree(temporary)
    assert photo.read_bytes() == original
    assert (annotated / 'annotations.json').read_bytes() == coordinates
    assert sorted(p.name for p in annotated.iterdir()) == ['annotations.json', 'phone.jpg']
