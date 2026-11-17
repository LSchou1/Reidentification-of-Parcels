"""Beginner-friendly entry point: prepare -> annotate -> split -> baseline -> train.

Run `.venv/bin/python workflow.py --help` from this folder.
Settings live in workflow.yaml. The longer explanations are in
docs/simple_workflow.md; the implementation is split into three small modules.
"""

import argparse
import json
import math
from pathlib import Path
import re

import yaml

from src.barcode_dataset import dataset_summary
from src.setup_barcode_model import sha256
from src.train_barcodes import run_training
from src.workflow_data import dataset_directory, frozen_dataset, group_counts, prepare_photos, save_json, split_photos
from src.workflow_models import model_files
from src.workflow_reports import analysis_request, comparison_report, run_analysis, write_output_index


MASTER_CLASSES = ["barcode", "label", "parcel"]


def read_config(path):
    """Resolve paths once, then reject mistakes with a useful error message."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping; start from workflow.yaml.")
    required = ["input", "annotations", "output", "temporary", "classes", "run_name", "split", "seed", "barcode_weights", "world_weights",
                "device", "imgsz", "prediction_floor", "nms_iou", "iou", "confidence", "select_confidence", "training"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration settings: {', '.join(missing)}")
    names = config["classes"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(name not in MASTER_CLASSES for name in names):
        raise ValueError("classes must contain a non-empty, unique selection of barcode, label and parcel.")
    if not isinstance(config["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_name"]):
        raise ValueError("run_name may contain letters, numbers, underscores and hyphens.")
    for key in ("input", "annotations", "output", "temporary", "barcode_weights", "world_weights"):
        config[key] = str((path.parent / config[key]).resolve())
    folders = [Path(config[key]) for key in ("input", "annotations", "output", "temporary")]
    for index, first in enumerate(folders):
        for second in folders[index + 1:]:
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("Keep input, annotations, output and temporary files separate.")
    for key in ("prediction_floor", "nms_iou", "iou", "confidence"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be a number above 0 and at most 1.")
    if config["confidence"] < config["prediction_floor"]:
        raise ValueError("confidence cannot be lower than prediction_floor.")
    if type(config["select_confidence"]) is not bool:
        raise ValueError("select_confidence must be true or false.")
    if type(config["seed"]) is not int or type(config["imgsz"]) is not int or config["imgsz"] < 32:
        raise ValueError("Use an integer seed and an integer imgsz of at least 32.")
    group_counts(3, config["split"])
    return config


def run_directory(config):
    # Class toggles get their own results, without losing an earlier experiment.
    return Path(config["output"]) / "experiments" / config["run_name"] / "_".join(config["classes"])


def training_request(config):
    """Bind a checkpoint to the exact dataset, starting weights and settings."""
    _, dataset_hash = frozen_dataset(config["temporary"], config["classes"])
    weights = Path(config["barcode_weights"])
    if not weights.is_file():
        raise ValueError(f"Missing training starting weights: {weights}")
    return {"dataset_manifest_sha256": dataset_hash, "classes": config["classes"],
            "starting_weights_sha256": sha256(weights), "training": config["training"],
            "seed": config["seed"], "device": config["device"], "imgsz": config["imgsz"]}


def require_trained_model(config):
    root = run_directory(config)
    path = root / "training_request.json"
    weights = root / "training/detector/weights/best.pt"
    if not path.is_file() or not weights.is_file():
        raise ValueError("Run train before validating/testing the fine-tuned model.")
    record = json.loads(path.read_text())
    if record["request"] != training_request(config) or record["trained_weights_sha256"] != sha256(weights):
        raise ValueError("Training settings, dataset or checkpoint changed. Use a new run_name.")
    return weights


def train(config, dry_run=False):
    request = training_request(config)
    root = run_directory(config)
    if not dry_run:
        # Inference comes before supervised training in this workflow.
        baseline_path = root / "baseline_val/evaluation.json"
        if not baseline_path.is_file():
            raise ValueError("Run baseline first, so you have a before-training reference.")
        baseline = json.loads(baseline_path.read_text())
        expected = analysis_request(config, "baseline", "val", model_files(config, "baseline"), request["dataset_manifest_sha256"])
        if baseline["workflow"]["request"] != expected:
            raise ValueError("Baseline settings changed. Choose a new run_name and run baseline again.")
    args = argparse.Namespace(
        data=dataset_directory(config["temporary"], config["classes"]) / "data.yaml",
        weights=Path(config["barcode_weights"]), output=root / "training",
        epochs=config["training"]["epochs"], batch=config["training"]["batch"],
        patience=config["training"]["patience"], imgsz=config["imgsz"],
        device=config["device"], seed=config["seed"], dry_run=dry_run)
    result = run_training(args)
    if not dry_run:
        save_json(root / "training_request.json", {"request": request,
                  "trained_weights_sha256": result["trained_weights"]["sha256"]})
    return result


def test_models(config):
    """Check BOTH validation selections before either model sees the test set."""
    root = run_directory(config)
    weights = require_trained_model(config)
    _, dataset_hash = frozen_dataset(config["temporary"], config["classes"])
    for kind in ("baseline", "finetuned"):
        path = root / f"{kind}_val/evaluation.json"
        if not path.is_file():
            raise ValueError("Run baseline and validate before test. Both confidence choices must be frozen first.")
        request = analysis_request(config, kind, "val", model_files(config, kind, weights), dataset_hash)
        if json.loads(path.read_text())["workflow"]["request"] != request:
            raise ValueError("A validation model/setting changed. Start a new run_name before testing.")
    for kind in ("baseline", "finetuned"):
        run_analysis(config, root, kind, "test", weights)
    return comparison_report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("workflow.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("prepare", "Prepare coordinate JSON; then ask the assistant to annotate."),
        ("split", "Make/freeze grouped train, validation and test data."),
        ("baseline", "Pretrained inference and analysis on validation."),
        ("validate", "Fine-tuned inference and analysis on validation."),
        ("test", "Final test of both models, plus comparison report."),
        ("compare", "Regenerate comparison from completed val/test reports."),
        ("status", "Show active classes, dataset and result locations."),
    ]:
        commands.add_parser(name, help=help_text)
    training = commands.add_parser("train", help="Explicit supervised fine-tuning; never runs during inference.")
    training.add_argument("--dry-run", action="store_true", help="Check the training recipe without loading/training a model.")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
        root = run_directory(config)
        if args.command in {"baseline", "train", "validate", "test", "compare"}:
            # Deleting tmp is safe: rebuild the same frozen view before using it.
            if not (dataset_directory(config["temporary"], config["classes"]) / "data.yaml").exists():
                split_photos(config["input"], config["annotations"], config["temporary"], config["output"], config["split"], config["seed"], config["classes"])
        if args.command == "prepare":
            result = prepare_photos(config["input"], config["annotations"], config["temporary"], MASTER_CLASSES)
            print(json.dumps(result, indent=2))
            print("Next: fill annotations.json with reviewed boxes and physical group IDs, then run split.")
        elif args.command == "split":
            print(json.dumps(split_photos(config["input"], config["annotations"], config["temporary"], config["output"], config["split"], config["seed"], config["classes"]), indent=2))
        elif args.command == "baseline":
            run_analysis(config, root, "baseline", "val")
        elif args.command == "train":
            train(config, args.dry_run)
        elif args.command == "validate":
            run_analysis(config, root, "finetuned", "val", require_trained_model(config))
        elif args.command == "test":
            print(f"Comparison: {test_models(config)}")
        elif args.command == "compare":
            frozen_dataset(config["temporary"], config["classes"])
            print(f"Comparison: {comparison_report(root)}")
        else:
            print(f"Active classes: {', '.join(config['classes'])}")
            print(f"Raw photos: {config['input']}\nAnnotations: {config['annotations']}\nResults: {root}")
            if (dataset_directory(config["temporary"], config["classes"]) / "data.yaml").is_file():
                dataset, _ = frozen_dataset(config["temporary"], config["classes"])
                print(json.dumps(dataset_summary(dataset), indent=2))
            else:
                print("No frozen dataset for this selection yet. Start with prepare, annotation, then split.")
        write_output_index(config)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\nSee docs/simple_workflow.md for the next steps.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
