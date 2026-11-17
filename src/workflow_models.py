"""Run box detection for the pretrained baseline or a fine-tuned joint model.

This module deliberately does no barcode decoding. Detection quality and whether
ZXing can read a detected barcode are different measurements.
"""

import importlib.metadata
from pathlib import Path
import time

import cv2

from src.detect_barcodes import LABEL_WORDS
from src.detect_packages import PACKAGE_WORDS
from src.setup_barcode_model import sha256


def model_files(config, kind, trained_weights=None):
    """Hashes identify the exact weights used, including the separate label baseline."""
    if kind == "baseline":
        paths = {name: Path(config["barcode_weights"] if name == "barcode" else config["world_weights"])
                 for name in config["classes"]}
    else:
        paths = {"joint": Path(trained_weights)}
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} weights: {path}. See docs/simple_workflow.md.")
        result[name] = {"file": str(path.resolve()), "sha256": sha256(path)}
    return result


def predict_boxes(samples, names, config, kind, trained_weights=None):
    """Keep low-confidence proposals so later analysis can try other thresholds."""
    import torch
    from ultralytics import YOLO, YOLOWorld

    files = model_files(config, kind, trained_weights)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    models = []
    if kind == "baseline":
        if "barcode" in names:
            barcode = YOLO(files["barcode"]["file"], task="detect")
            ids = [i for i, name in barcode.names.items() if name.lower().replace("_", "") == "barcode"]
            if not ids:
                raise ValueError("Baseline weights have no barcode class.")
            models.append((barcode, ids, False, "barcode"))
        # Keep the two prompt groups separate: a parcel must not suppress its label.
        for name, prompts in (("label", LABEL_WORDS), ("parcel", PACKAGE_WORDS)):
            if name in names:
                world = YOLOWorld(files[name]["file"])
                world.set_classes(prompts)
                models.append((world, list(range(len(prompts))), True, name))
    else:
        model = YOLO(files["joint"]["file"], task="detect")
        if not set(names).issubset(model.names.values()):
            raise ValueError(f"Trained model must contain all dataset classes: {names}.")
        ids = [i for i, name in model.names.items() if name in names]
        models.append((model, ids, False, None))

    records, discarded, capped = [], 0, []
    for sample in samples:
        image = cv2.imread(str(sample["path"]))
        if image is None:
            raise ValueError(f"Unreadable image: {sample['path']}")
        start = time.perf_counter()
        record = {key: sample[key] for key in ("image", "sha256", "width", "height")}
        record["detections"] = []
        for model, ids, agnostic, forced_class in models:
            prediction = model.predict(
                image, conf=config["prediction_floor"], imgsz=config["imgsz"],
                iou=config["nms_iou"], max_det=300, classes=ids,
                agnostic_nms=agnostic, device=config["device"], verbose=False)[0]
            if len(prediction.boxes) == 300:
                capped.append(sample["image"])
            for box in prediction.boxes:
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                # Very weak proposals at an image boundary can have zero area.
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    discarded += 1
                    continue
                name = forced_class or model.names[int(box.cls[0])]
                record["detections"].append({"class": name, "xyxy": xyxy,
                                              "confidence": float(box.conf[0])})
        record["seconds"] = time.perf_counter() - start
        records.append(record)
        print(f"{sample['image']}: {len(record['detections'])} candidate boxes", flush=True)
    return {"model": {"kind": kind, "weights": files,
                       "label_prompts": LABEL_WORDS if kind == "baseline" and "label" in names else None,
                       "parcel_prompts": PACKAGE_WORDS if kind == "baseline" and "parcel" in names else None},
            "settings": {"confidence": config["prediction_floor"], "imgsz": config["imgsz"],
                         "nms_iou": config["nms_iou"], "device": config["device"],
                         "max_det_per_model": 300, "decoding_performed": False,
                         "discarded_degenerate_predictions": discarded, "cap_reached_images": capped},
            "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "numpy")},
            "images": records}
