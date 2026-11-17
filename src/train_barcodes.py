"""Initialize, check and fine-tune a barcode/label detector on reviewed YOLO data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from src.barcode_dataset import check_dataset, dataset_summary, initialize, label_path, import_coco
from src.setup_barcode_model import DEFAULT_WEIGHTS, ROOT, sha256


def training_arguments(output, epochs, imgsz, batch, device, patience, seed):
    return {"data": str(output / "resolved_data.yaml"), "epochs": epochs, "imgsz": imgsz,
            "batch": batch, "device": device, "patience": patience, "seed": seed,
            "workers": 0, "project": str(output), "name": "detector", "exist_ok": False,
            "plots": True, "degrees": 15.0, "fliplr": 0.0, "flipud": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.2, "mosaic": 0.0}


def run_training(args):
    dataset = check_dataset(args.data)
    for name in dataset["names"]:
        if not any(a["class"] == name for s in dataset["splits"]["train"] for a in s["annotations"]):
            raise ValueError(f"Training has no reviewed examples of class {name}. Add examples or remove that class from names.")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.patience < 0:
        raise ValueError("Use positive epochs/batch, imgsz >= 32 and patience >= 0.")
    if not args.weights.is_file():
        raise ValueError(f"Missing starting weights: {args.weights}. Run python -m src.setup_barcode_model first.")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Training output already exists: {output}. Choose a new --output to preserve experiments.")
    if output.is_relative_to(dataset["root"]):
        raise ValueError("Store training runs outside the dataset directory.")
    settings = training_arguments(output, args.epochs, args.imgsz, args.batch,
                                  args.device, args.patience, args.seed)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "source_weights": {"file": str(args.weights.resolve()), "sha256": sha256(args.weights)},
                "dataset": dataset_summary(dataset), "settings": settings,
                "images": {split: [{"image": s["image"], "sha256": s["sha256"],
                                     "annotation_sha256": sha256(label_path(s["path"]))}
                                    for s in samples] for split, samples in dataset["splits"].items()}}
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print("Dry run: dataset checked; no model loaded, output written or training started.")
        return manifest
    from ultralytics import YOLO
    import importlib.metadata
    manifest["versions"] = {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "numpy")}
    output.mkdir(parents=True)
    # Absolute image lists make CVAT exports and relative YAML paths unambiguous to Ultralytics.
    resolved = {"path": str(dataset["root"]), "names": dict(enumerate(dataset["names"]))}
    for split in ("train", "val"):
        listing = output / f"{split}.txt"
        listing.write_text("\n".join(str(s["path"]) for s in dataset["splits"][split]) + "\n")
        resolved[split] = str(listing)
    # The test split is deliberately not supplied to training/model selection.
    (output / "resolved_data.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    model = YOLO(str(args.weights), task="detect")
    model.train(**settings)
    best = output / "detector/weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training did not produce the expected checkpoint: {best}")
    manifest["trained_weights"] = {"file": str(best), "sha256": sha256(best)}
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Best weights: {best}")
    print("Next: evaluate the baseline and best.pt on the same validation data. Use test only for final comparison.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an empty YOLO dataset, without inventing annotations.")
    init.add_argument("--dataset", type=Path, default=ROOT / "tmp/manual_dataset")
    init.add_argument("--with-labels", action="store_true", help="Add the class label for whole shipping labels.")
    check = commands.add_parser("check", help="Validate boxes, clean images and split leakage.")
    check.add_argument("--data", type=Path, required=True)
    convert = commands.add_parser("import-coco", help="Convert reviewed COCO JSON boxes to the initialized YOLO layout.")
    convert.add_argument("--annotations", type=Path, required=True)
    convert.add_argument("--images", type=Path, required=True)
    convert.add_argument("--data", type=Path, required=True)
    convert.add_argument("--split", choices=("train", "val", "test"), required=True)
    convert.add_argument("--barcode-category", default="barcode")
    convert.add_argument("--label-category", default="label")
    train = commands.add_parser("train", help="Fine-tune existing weights using supervised annotations.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    train.add_argument("--output", type=Path, default=ROOT / "data/output/next_batch/experiments/manual_training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=1280)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--device", default="cpu", help="cpu, mps, or a CUDA device such as 0.")
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--dry-run", action="store_true", help="Check inputs and print the recipe without training.")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(f"Dataset template: {initialize(args.dataset, args.with_labels)}")
        elif args.command == "check":
            dataset = check_dataset(args.data)
            print(json.dumps(dataset_summary(dataset), indent=2, ensure_ascii=False))
            if not dataset["groups_checked"]:
                print("Only exact image duplication was checked. Fill groups.csv to check shared recordings/parcels too.")
        elif args.command == "import-coco":
            print(json.dumps(import_coco(args.annotations, args.images, args.data, args.split,
                                         args.barcode_category, args.label_category), indent=2))
        else:
            run_training(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
