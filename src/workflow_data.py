"""Prepare coordinates and freeze either a grouped or exploratory image split.

Grouped evaluation keeps physical parcel/recording IDs together. Exploratory
experiments allow familiar parcels across splits, while keeping identical pixels
together. Neither mode groups different parcels by their visual appearance.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import cv2
import yaml

from src.barcode_dataset import EXTENSIONS, check_dataset, dataset_summary, label_path
from src.setup_barcode_model import sha256


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_photos(source, annotations, temporary, names):
    """Prepare ONE coordinate JSON; keep clean source pixels only in raw/."""
    source, annotations, temporary = (Path(p).resolve() for p in (source, annotations, temporary))
    if annotations.exists():
        raise ValueError("This annotated batch already exists. Choose a new batch name.")
    for first, second in ((source, annotations), (source, temporary), (annotations, temporary)):
        if first == second or first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("Keep raw photos, annotated photos and temporary files separate.")
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No JPG/JPEG/PNG photos in {source}. Add photos first.")
    preparation = temporary / "preparation.json"
    if preparation.exists():
        raise ValueError("This temporary folder belongs to an existing batch. Choose a new folder.")
    annotations.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary, prefix="prepare_") as staging_parent:
        staging = Path(staging_parent) / "batch"
        staging.mkdir()
        records, coordinates, stems = [], {}, set()
        for image in images:
            relative = image.relative_to(source)
            # Preserve identity when two raw subfolders both contain view_01.jpg.
            filename = "__".join(relative.parts)
            stem = Path(filename).stem.casefold()
            if stem in stems:
                raise ValueError(f"Two photos would share a filename: {relative}")
            stems.add(stem)
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable photo: {image}")
            image_name = relative.as_posix()
            height, width = pixels.shape[:2]
            records.append({"image": image_name, "sha256": sha256(image),
                            "width": width, "height": height,
                            "pixel_sha256": hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()})
            coordinates[image_name] = {"width": width, "height": height,
                "group": relative.parts[0] if len(relative.parts) > 1 else "",
                "reviewed": False, **{name: [] for name in names}}
        save_json(staging / "annotations.json", coordinates)
        staging.rename(annotations)
        save_json(preparation, {"source": str(source), "images": records})
    return {"images": len(records), "annotations": str(annotations / "annotations.json")}


def coordinate_boxes(record, names, filename):
    """Read pixel boxes; an unreviewed/absent class must never become a negative."""
    if record.get("reviewed") is not True:
        raise ValueError(f"Missing annotation review for {filename}. Finish its JSON record first.")
    width, height = record["width"], record["height"]
    boxes = []
    for name in names:
        if name not in record or not isinstance(record[name], list):
            raise ValueError(f"Missing annotation class {name} for {filename}.")
        for box in record[name]:
            if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError(f"Invalid {name} coordinates for {filename}.")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid {name} box outside {filename}.")
            boxes.append({"class": name, "xyxy": box})
    return boxes


def inspection_samples(source, annotations, names):
    """Read reviewed boxes directly, without creating or assuming a data split."""
    source = Path(source).resolve()
    coordinates = json.loads((Path(annotations) / "annotations.json").read_text())
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No photos in {source}.")
    if set(coordinates) != {p.relative_to(source).as_posix() for p in images}:
        raise ValueError("Raw photos and annotations.json must list the same images.")
    samples = []
    for path in images:
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Photo is outside the raw folder: {path}")
        name = path.relative_to(source).as_posix()
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ValueError(f"Unreadable photo: {path}")
        height, width = pixels.shape[:2]
        record = coordinates[name]
        if (record.get("width"), record.get("height")) != (width, height):
            raise ValueError(f"Annotation dimensions differ from the raw photo: {name}")
        samples.append({"image": name, "path": path, "sha256": sha256(path),
                        "width": width, "height": height,
                        "annotations": coordinate_boxes(record, names, name)})
    return samples


def related_groups(records, rows):
    """Connect explicit IDs transitively, including photos showing several parcels.

    Use A|B for a photo showing parcels A and B. That links all A and B photos.
    Byte/pixel-identical copies are also connected, even if their IDs differ.
    """
    by_image = {record["image"]: record for record in records}
    if len(rows) != len(by_image) or {row["image"] for row in rows} != set(by_image):
        raise ValueError("Grouping must list every prepared image exactly once.")
    parents = {image: image for image in by_image}

    def root(image):
        while parents[image] != image:
            image = parents[image]
        return image

    owners = {}
    for row in rows:
        image = row["image"]
        if not isinstance(row["group"], str):
            raise ValueError(f"Group IDs must be text in annotations.json: {image}.")
        ids = [value.strip() for value in row["group"].split("|")]
        if not all(ids):
            raise ValueError(f"Missing group for {image}. Fill its group field in annotations.json with physical parcel or recording IDs.")
        tokens = ["id:" + value for value in ids] + ["pixels:" + by_image[image]["pixel_sha256"]]
        for token in tokens:
            if token in owners:
                parents[root(image)] = root(owners[token])
            else:
                owners[token] = image
    connected = {}
    for image in sorted(by_image):
        connected.setdefault(root(image), []).append(image)
    return sorted(connected.values(), key=lambda images: images[0])


def group_counts(number, ratios):
    """Round ratios to whole groups, leaving at least one group in each split."""
    if set(ratios) != {"train", "val", "test"} or any(value <= 0 for value in ratios.values()):
        raise ValueError("Set positive train, val and test ratios.")
    if abs(sum(ratios.values()) - 1) > 1e-8:
        raise ValueError("The train/val/test ratios must add up to 1.")
    if number < 3:
        raise ValueError("Need at least 3 independent parcel/recording groups for train, val and test. Do not split related photos to work around this.")
    targets = {split: number * ratios[split] for split in ("train", "val", "test")}
    counts = {split: max(1, int(target)) for split, target in targets.items()}
    while sum(counts.values()) > number:
        split = max((key for key in counts if counts[key] > 1), key=lambda key: counts[key] - targets[key])
        counts[split] -= 1
    while sum(counts.values()) < number:
        split = max(counts, key=lambda key: targets[key] - counts[key])
        counts[split] += 1
    return counts


def dataset_directory(temporary, classes, split_mode="grouped"):
    """Each class selection has its own view; master annotations stay untouched."""
    root = Path(temporary) / "datasets"
    if split_mode == "exploratory":
        root /= "exploratory"
    return root / "_".join(classes)


def split_photos(source, annotations, temporary, output, ratios, seed, classes=None, split_mode="grouped"):
    """Copy reviewed images/boxes into a new, fixed train/validation/test dataset."""
    if split_mode not in ("grouped", "exploratory"):
        raise ValueError("split_mode must be grouped or exploratory.")
    pool, annotated, temporary, output = (Path(p).resolve() for p in (source, annotations, temporary, output))
    folders = [pool, annotated, temporary, output]
    for index, first in enumerate(folders):
        for second in folders[index + 1:]:
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("Keep raw photos, annotations, outputs and temporary files separate.")
    coordinates_path = annotated / "annotations.json"
    coordinates = json.loads(coordinates_path.read_text())
    master_names = ["barcode", "label", "parcel"]
    classes = master_names if classes is None else classes
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(master_names):
        raise ValueError(f"Choose unique active classes from {master_names}.")
    destination = dataset_directory(temporary, classes, split_mode)
    preparation_path = temporary / "preparation.json"
    if not preparation_path.exists():
        # tmp/ may be deleted. Rebuild fingerprints from the permanent clean raw photos.
        records = []
        for image in sorted(pool.rglob("*")):
            if image.suffix.lower() not in EXTENSIONS:
                continue
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Unreadable photo: {image}")
            height, width = pixels.shape[:2]
            records.append({"image": image.relative_to(pool).as_posix(), "width": width, "height": height,
                "sha256": sha256(image), "pixel_sha256": hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()})
        save_json(preparation_path, {"images": records})
    preparation = json.loads(preparation_path.read_text())
    records = preparation["images"]
    actual = {p.relative_to(pool).as_posix() for p in pool.rglob("*") if p.suffix.lower() in EXTENSIONS}
    if actual != {r["image"] for r in records} or set(coordinates) != actual:
        raise ValueError("The photo list or JSON image list changed. Prepare a new batch.")
    for record in records:
        name = record["image"]
        if not (pool / name).resolve().is_relative_to(pool) or sha256(pool / name) != record["sha256"]:
            raise ValueError(f"A clean photo changed: {name}. Prepare a new batch.")
        annotation = coordinates[name]
        if (annotation.get("width"), annotation.get("height")) != (record["width"], record["height"]):
            raise ValueError(f"Annotation dimensions differ from the photo: {name}.")
    rows = [{"image": r["image"], "group": coordinates[r["image"]].get("group", "")} for r in records]
    if split_mode == "exploratory":
        # Deliberately allow different views of the same parcel across splits.
        # Keep identical pixels together; these are image units, NOT parcel IDs.
        groups = related_groups(records, [{"image": r["image"], "group": r["image"]} for r in records])
        if len(groups) < 3:
            raise ValueError("Exploratory splitting needs at least 3 distinct images after keeping exact duplicates together.")
    else:
        groups = related_groups(records, rows)
    counts = group_counts(len(groups), ratios)
    # Parse all active classes before creating any generated files.
    boxes_by_image = {r["image"]: coordinate_boxes(coordinates[r["image"]], [name for name in master_names if name in classes], r["image"]) for r in records}
    random.Random(seed).shuffle(groups)
    by_image = {record["image"]: record for record in records}
    # Class selection must not reshuffle test images. Freeze the common plan once.
    plan = {"seed": seed, "ratios": ratios, "groups": groups,
            "annotations_sha256": sha256(coordinates_path),
            "images": {r["image"]: r["sha256"] for r in records}}
    plan_path = output / "dataset_split.json"
    if split_mode == "exploratory":
        plan.update(split_mode=split_mode, source_groups={r["image"]: r["group"] for r in rows})
        plan_path = output / "dataset_split_exploratory.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Photos, annotations, grouping, ratios or seed changed after the split was frozen. Use a new batch; class toggles can reuse the existing plan.")
    if destination.exists():
        frozen_dataset(temporary, classes, split_mode)
        return json.loads((destination / "split_manifest.json").read_text())["summary"]
    manifest = {"seed": seed, "requested_group_ratios": ratios, "classes": classes,
                "images": [], "groups": counts}
    if split_mode == "exploratory":
        manifest.update(split_mode=split_mode, split_unit="exact_duplicate_image_group",
                        source_groups=plan["source_groups"])
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary, prefix="split_") as staging_parent:
        staging = Path(staging_parent) / "dataset"
        config = {"path": ".", "names": dict(enumerate(classes))}
        group_rows, offset = [], 0
        for split, number in counts.items():
            config[split] = f"images/{split}"
            for index, images in enumerate(groups[offset:offset + number], offset):
                group_id = f"group_{index:05d}"
                for image in images:
                    source = pool / image
                    relative = source.relative_to(pool)
                    target = staging / "images" / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    label_path(target).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    # Map by class name. Removing/reordering a class never corrupts IDs.
                    record = by_image[image]
                    boxes = boxes_by_image[image]
                    lines = []
                    for box in boxes:
                        if box["class"] not in classes:
                            continue
                        x1, y1, x2, y2 = box["xyxy"]
                        width, height = record["width"], record["height"]
                        lines.append(f"{classes.index(box['class'])} {(x1+x2)/2/width:.10f} {(y1+y2)/2/height:.10f} {(x2-x1)/width:.10f} {(y2-y1)/height:.10f}")
                    label_path(target).write_text("\n".join(lines) + ("\n" if lines else ""))
                    name = target.relative_to(staging).as_posix()
                    group_rows.append((name, group_id))
                    manifest["images"].append({"image": name, "group": group_id, "split": split,
                        "sha256": by_image[image]["sha256"], "annotation_sha256": sha256(label_path(target))})
            offset += number
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        with (staging / "groups.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "group"])
            writer.writerows(group_rows)
        checked = check_dataset(staging / "data.yaml", required=("train", "val", "test"))
        train_classes = {a["class"] for s in checked["splits"]["train"] for a in s["annotations"]}
        if train_classes != set(checked["names"]):
            raise ValueError("The training split lacks a configured class. Add independent groups with that class, or try another seed before freezing the split.")
        manifest["summary"] = dataset_summary(checked)
        manifest["data_sha256"] = sha256(staging / "data.yaml")
        manifest["groups_sha256"] = sha256(staging / "groups.csv")
        save_json(staging / "split_manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        save_json(plan_path, plan)
    return manifest["summary"]


def frozen_dataset(temporary, classes, split_mode="grouped"):
    """Reject edits after splitting, so model reports keep referring to the same data."""
    data = dataset_directory(temporary, classes, split_mode) / "data.yaml"
    manifest_path = data.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run prepare, annotate the batch, then run split first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split_mode", "grouped") != split_mode:
        raise ValueError("The frozen dataset belongs to a different split_mode.")
    if sha256(data) != manifest["data_sha256"] or sha256(data.parent / "groups.csv") != manifest["groups_sha256"]:
        raise ValueError("The frozen dataset configuration changed. Use a new batch.")
    dataset = check_dataset(data, required=("train", "val", "test"))
    actual = {s["image"]: s for samples in dataset["splits"].values() for s in samples}
    if set(actual) != {r["image"] for r in manifest["images"]}:
        raise ValueError("The frozen dataset image list changed. Use a new batch.")
    for record in manifest["images"]:
        sample = actual[record["image"]]
        if sample["sha256"] != record["sha256"] or sha256(label_path(sample["path"])) != record["annotation_sha256"]:
            raise ValueError(f"Frozen image or annotation changed: {record['image']}. Use a new batch.")
    return dataset, sha256(manifest_path)
