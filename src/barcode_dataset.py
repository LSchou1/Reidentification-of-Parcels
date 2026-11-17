"""Read and check reviewed YOLO annotations before training or evaluation."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def initialize(directory, with_labels=False):
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"Destination already exists: {directory}")
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (directory / kind / split).mkdir(parents=True)
    config = {"path": ".", "train": "images/train", "val": "images/val",
              "test": "images/test", "names": {0: "barcode"}}
    if with_labels:
        config["names"][1] = "label"
    (directory / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (directory / "groups.csv").write_text("image,group\n", encoding="utf-8")
    (directory / "README.txt").write_text(
        "Put clean images in images/{train,val,test} and reviewed YOLO .txt files in labels/{split}.\n"
        "One row per object: class_id x_center y_center width height (normalized 0..1).\n"
        "A missing .txt is UNANNOTATED and is rejected. An empty .txt means reviewed negative.\n"
        "Annotate all visible instances of every configured class, including unreadable barcodes.\n"
        "Model predictions are suggestions; review and correct them before using them as ground truth.\n"
        "Keep the same physical parcel and neighbouring video frames in one split.\n"
        "Optional groups.csv: image relative to this folder,group (one row for EVERY image).\n"
        "For example: images/train/photo.jpg,recording_01_parcel_03\n"
        "Do not use an unverified barcode value as the parcel identity.\n", encoding="utf-8")
    return directory / "data.yaml"


def load_dataset(data):
    data = Path(data).resolve()
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain path, train, val and names.")
    names = config.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must be consecutive integers starting at 0.")
        names = [names[i] for i in range(len(names))]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or n not in {"barcode", "label", "parcel"} for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Use class names barcode, label and/or parcel, without duplicate names.")
    root = (data.parent / config.get("path", ".")).resolve()
    return config, root, names


def split_images(config, root, split):
    sources = config.get(split)
    if not sources:
        return []
    sources = sources if isinstance(sources, list) else [sources]
    images = []
    for source in sources:
        path = (root / source).absolute()
        if path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS))
        elif path.is_file() and path.suffix == ".txt":
            for line in path.read_text().splitlines():
                if line.strip():
                    images.append((path.parent / line.strip()).absolute())
        else:
            raise ValueError(f"Missing image directory or image list for {split}: {path}")
    if len(images) != len(set(images)):
        raise ValueError(f"Duplicate image paths within {split}.")
    return images


def label_path(image):
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Use a YOLO images/ and labels/ directory layout: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_annotations(path, width, height, names):
    if not path.is_file():
        raise ValueError(f"Missing annotation: {path}. Annotate first; do not treat missing labels as negatives.")
    annotations = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError()
            cls, x, y, w, h = values
            if cls != int(cls) or not 0 <= cls < len(names) or w <= 0 or h <= 0:
                raise ValueError()
            box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            if min(box) < -1e-6 or max(box) > 1 + 1e-6:
                raise ValueError()
        except ValueError:
            raise ValueError(f"Invalid YOLO annotation at {path}:{number}; expected class x y w h in 0..1.") from None
        annotations.append({"class": names[int(cls)],
                            "xyxy": [box[0] * width, box[1] * height,
                                     box[2] * width, box[3] * height]})
    return annotations


def check_dataset(data, required=("train", "val")):
    config, root, names = load_dataset(data)
    records, fingerprints = {}, {}
    for split in ("train", "val", "test"):
        records[split] = []
        for image in split_images(config, root, split):
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable image: {image}")
            height, width = pixels.shape[:2]
            annotations = read_annotations(label_path(image), width, height, names)
            # Hash decoded pixels too, so a metadata-only re-export cannot bypass the split check.
            fingerprint = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                raise ValueError(f"Data leakage: identical image pixels occur in {fingerprints[fingerprint]} and {split}: {image}")
            fingerprints[fingerprint] = split
            records[split].append({"path": image, "image": image.relative_to(root).as_posix(),
                                   "width": width, "height": height, "annotations": annotations,
                                   "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    for split in required:
        if not records.get(split):
            raise ValueError(f"The {split} split is empty. Add reviewed images and annotations first.")
    groups_path = root / "groups.csv"
    groups = {}
    if groups_path.is_file():
        with groups_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image", "group"]:
                raise ValueError("groups.csv must have exactly these columns: image,group")
            for row in reader:
                if not row["image"] or not row["group"] or row["image"] in groups:
                    raise ValueError("groups.csv contains an empty value or duplicate image row.")
                groups[row["image"]] = row["group"]
    group_splits = {}
    if groups:
        for split, samples in records.items():
            for sample in samples:
                group = groups.get(sample["image"])
                if not group:
                    raise ValueError(f"groups.csv has no group for {sample['image']}")
                if group in group_splits and group_splits[group] != split:
                    raise ValueError(f"Data leakage: group {group} occurs in multiple splits.")
                group_splits[group] = split
    return {"root": root, "names": names, "splits": records, "groups_checked": bool(groups)}


def dataset_summary(dataset):
    return {"names": dataset["names"], "groups_checked": dataset["groups_checked"],
            "splits": {split: {"images": len(samples),
                               "objects": sum(len(s["annotations"]) for s in samples),
                               "negatives": sum(not s["annotations"] for s in samples)}
                       for split, samples in dataset["splits"].items()}}


def import_coco(annotations, images, data, split, barcode_category="barcode", label_category="label"):
    """Convert an exhaustively reviewed COCO image/box export, preserving clean originals."""
    config, root, names = load_dataset(data)
    coco = json.loads(Path(annotations).read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    mapping = {barcode_category: "barcode", label_category: "label"}
    selected = {key: names.index(mapping[name]) for key, name in categories.items()
                if name in mapping and mapping[name] in names}
    if not selected:
        raise ValueError(f"No matching COCO categories in {list(categories.values())}. Set --barcode-category/--label-category to the export names.")
    image_dir = root / "images" / split
    if config.get(split) != f"images/{split}":
        raise ValueError("COCO import expects the directory layout created by the init command.")
    image_records = {record["id"]: record for record in coco["images"]}
    if not image_records or len(image_records) != len(coco["images"]):
        raise ValueError("COCO image list is empty or contains duplicate image IDs.")
    rows = {key: [] for key in image_records}
    for annotation in coco["annotations"]:
        if annotation["category_id"] not in selected:
            continue
        image_id = annotation["image_id"]
        if image_id not in image_records:
            raise ValueError("COCO annotation references an unknown image ID.")
        if annotation.get("iscrowd", 0):
            raise ValueError("Crowd annotations require manual review before conversion to individual YOLO boxes.")
        width, height = image_records[image_id]["width"], image_records[image_id]["height"]
        values = annotation["bbox"]
        if len(values) != 4 or not np.isfinite(values).all() or width <= 0 or height <= 0:
            raise ValueError("Invalid COCO image dimensions or bbox.")
        x, y, w, h = values
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
            raise ValueError("COCO bbox is empty or outside the source image.")
        rows[image_id].append(f"{selected[annotation['category_id']]} {(x + w/2)/width:.10f} {(y + h/2)/height:.10f} {w/width:.10f} {h/height:.10f}")
    planned, used_stems = [], set()
    for image_id, record in image_records.items():
        source = (Path(images) / record["file_name"]).resolve()
        if not source.is_relative_to(Path(images).resolve()):
            raise ValueError("COCO file_name must stay inside the supplied --images directory.")
        destination = image_dir / source.name
        if source.suffix.lower() not in EXTENSIONS or destination.stem in used_stems:
            raise ValueError("COCO images must be JPG/PNG and have unique filename stems.")
        used_stems.add(destination.stem)
        if destination.exists() or label_path(destination).exists():
            raise ValueError(f"Import would overwrite existing data: {destination}")
        pixels = cv2.imread(str(source))
        if pixels is None or pixels.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Missing clean image or dimensions differ from COCO: {source}")
        planned.append((source, destination, rows[image_id]))
    for source, destination, lines in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        label_path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        label_path(destination).write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"split": split, "images": len(planned), "objects": sum(len(lines) for _, _, lines in planned)}
