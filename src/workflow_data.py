"""Copy a photo batch, then make a reproducible split by physical parcel/recording.

Different parcels may look alike. Only explicit group IDs and identical pixels
connect images here; the splitter never groups images by their appearance.
"""

import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import cv2
import yaml

from src.barcode_dataset import EXTENSIONS, check_dataset, dataset_summary, label_path, read_annotations
from src.setup_barcode_model import sha256


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_photos(source, annotations, names):
    """Make clean copies. Missing annotations remain missing, never fake negatives."""
    source, annotations = Path(source).resolve(), Path(annotations).resolve()
    pool = annotations
    if pool.exists():
        raise ValueError("This annotated batch already exists. Choose a new batch name; existing annotations are preserved.")
    if source == annotations or source.is_relative_to(annotations) or annotations.is_relative_to(source):
        raise ValueError("Keep raw photos and annotated photos in separate folders.")
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No JPG/JPEG/PNG photos in {source}. Add photos first.")
    annotations.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temporary folder, so a failed check cannot leave half a batch.
    with tempfile.TemporaryDirectory(dir=annotations.parent, prefix="prepare_") as temporary:
        staging = Path(temporary) / "pool"
        records, stems, groups = [], set(), []
        for source_image in images:
            relative = source_image.relative_to(source)
            stem = relative.with_suffix("").as_posix().casefold()
            if stem in stems:
                raise ValueError(f"Two photos would share one annotation file: {relative}")
            stems.add(stem)
            pixels = cv2.imread(str(source_image))
            if pixels is None:
                raise ValueError(f"Unreadable photo: {source_image}")
            target = staging / "images" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target)
            image_name = target.relative_to(staging).as_posix()
            height, width = pixels.shape[:2]
            records.append({"image": image_name, "sha256": sha256(target),
                            "width": width, "height": height,
                            "pixel_sha256": hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()})
            # One top-level folder = one physical parcel or recording group.
            # Flat input has no known identity, so the CSV must be filled in.
            groups.append((image_name, relative.parts[0] if len(relative.parts) > 1 else ""))
            label_path(target).parent.mkdir(parents=True, exist_ok=True)
        (staging / "data.yaml").write_text(yaml.safe_dump(
            {"path": ".", "val": "images", "names": dict(enumerate(names))}, sort_keys=False))
        with (staging / "groups.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "group"])
            writer.writerows(groups)
        save_json(staging / "preparation.json", {"source": str(source), "images": records})
        staging.rename(pool)
    return {"images": len(records), "annotations": str(pool), "annotation_folder": str(pool / "labels")}


def related_groups(records, groups_file):
    """Connect explicit IDs transitively, including photos showing several parcels.

    Use A|B for a photo showing parcels A and B. That links all A and B photos.
    Byte/pixel-identical copies are also connected, even if their IDs differ.
    """
    by_image = {record["image"]: record for record in records}
    with Path(groups_file).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image", "group"]:
            raise ValueError("groups.csv needs exactly the columns image,group.")
        rows = list(reader)
    if len(rows) != len(by_image) or {row["image"] for row in rows} != set(by_image):
        raise ValueError("groups.csv must list every prepared image exactly once.")
    parents = {image: image for image in by_image}

    def root(image):
        while parents[image] != image:
            image = parents[image]
        return image

    owners = {}
    for row in rows:
        image = row["image"]
        ids = [value.strip() for value in (row["group"] or "").split("|")]
        if not all(ids):
            raise ValueError(f"Missing group for {image}. Fill annotated batch groups.csv with physical parcel or recording IDs.")
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


def dataset_directory(output, classes):
    """Each class selection has its own view; master annotations stay untouched."""
    return Path(output) / "datasets" / "_".join(classes)


def split_photos(annotations, output, ratios, seed, classes=None):
    """Copy reviewed images/boxes into a new, fixed train/validation/test dataset."""
    pool, output = Path(annotations).resolve(), Path(output).resolve()
    if pool == output or pool.is_relative_to(output) or output.is_relative_to(pool):
        raise ValueError("Keep annotation masters and generated outputs in separate folders.")
    pool_config = yaml.safe_load((pool / "data.yaml").read_text())
    master_names = list(pool_config["names"].values())
    classes = master_names if classes is None else classes
    if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(master_names):
        raise ValueError(f"Choose unique active classes from {master_names}.")
    destination = dataset_directory(output, classes)
    if destination.exists():
        frozen_dataset(output, classes)
        previous = json.loads((destination / "split_manifest.json").read_text())
        if previous["seed"] != seed or previous["requested_group_ratios"] != ratios:
            raise ValueError("This split is frozen. Use a new batch to change its seed or ratios.")
        return previous["summary"]
    preparation = json.loads((pool / "preparation.json").read_text())
    records = preparation["images"]
    # Verify the clean copies before trusting cached pixel fingerprints.
    actual = {p.relative_to(pool).as_posix() for p in (pool / "images").rglob("*") if p.suffix.lower() in EXTENSIONS}
    if actual != {r["image"] for r in records}:
        raise ValueError("The prepared photo list changed. Prepare a new batch instead of adding files to the annotated batch.")
    for record in records:
        if sha256(pool / record["image"]) != record["sha256"]:
            raise ValueError(f"A clean photo changed: {record['image']}. Prepare a new batch.")
    groups = related_groups(records, pool / "groups.csv")
    counts = group_counts(len(groups), ratios)
    # Check annotations after groups; a blank .txt is allowed only for a reviewed negative.
    check_dataset(pool / "data.yaml", required=("val",))
    random.Random(seed).shuffle(groups)
    by_image = {record["image"]: record for record in records}
    # Class selection must not reshuffle test images. Freeze the common plan once.
    plan = {"seed": seed, "ratios": ratios, "groups": groups,
            "pool_config_sha256": sha256(pool / "data.yaml"),
            "pool_images": {r["image"]: r["sha256"] for r in records},
            "pool_annotations": {r["image"]: sha256(label_path(pool / r["image"])) for r in records},
            "group_csv_sha256": sha256(pool / "groups.csv")}
    plan_path = output / "split_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Photos, annotations, grouping, ratios or seed changed after the split was frozen. Use a new batch; class toggles can reuse the existing plan.")
    manifest = {"seed": seed, "requested_group_ratios": ratios, "classes": classes,
                "images": [], "groups": counts}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix="split_") as temporary:
        staging = Path(temporary) / "dataset"
        config = {"path": ".", "names": dict(enumerate(classes))}
        group_rows, offset = [], 0
        for split, number in counts.items():
            config[split] = f"images/{split}"
            for index, images in enumerate(groups[offset:offset + number], offset):
                group_id = f"group_{index:05d}"
                for image in images:
                    source = pool / image
                    relative = source.relative_to(pool / "images")
                    target = staging / "images" / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    label_path(target).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    # Map by class name. Removing/reordering a class never corrupts IDs.
                    record = by_image[image]
                    boxes = read_annotations(label_path(source), record["width"], record["height"], master_names)
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


def frozen_dataset(output, classes):
    """Reject edits after splitting, so model reports keep referring to the same data."""
    data = dataset_directory(output, classes) / "data.yaml"
    manifest_path = data.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run prepare, annotate the batch, then run split first.")
    manifest = json.loads(manifest_path.read_text())
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
