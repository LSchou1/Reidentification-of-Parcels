"""A small, readable example of pretrained neural-network parcel detection."""

import argparse
import json
from pathlib import Path

import cv2


# These words describe the ONE category we want: a package/parcel.
# They configure an already-trained model; they do not train a new model.
PACKAGE_WORDS = ["parcel", "package", "cardboard box", "plastic mailing bag"]
IMAGE_SIZE = 960
NMS_IOU = 0.5


def find_images(source):
    """Accept one image or the images directly inside one folder."""
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(source.iterdir()) if source.is_dir() else [source]
    images = [p for p in paths if p.is_file() and p.suffix.lower() in extensions]
    if not images:
        raise ValueError(f"No JPG/PNG images found at {source}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Give input images different names before processing a folder.")
    return images


def save_detection(image, result, output_path):
    """Draw predictions and keep the coordinates for later evaluation."""
    overlay = image.copy()
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        score = float(box.conf[0])
        detections.append({"xyxy": [x1, y1, x2, y2], "confidence": score, "class": "package"})
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 3)
        cv2.putText(overlay, f"package {score:.2f}", (int(x1), max(25, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    if not cv2.imwrite(str(output_path), overlay):
        raise ValueError(f"Could not save {output_path}")
    return detections


def confidence_value(text):
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("Confidence must be between 0 and 1.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/next_batch"), help="One JPG/PNG or a folder.")
    parser.add_argument("--output", type=Path, default=Path("data/output/packages"), help="Folder for results.")
    parser.add_argument("--confidence", type=confidence_value, default=0.05, help="Minimum score (default: 0.05).")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8s-worldv2.pt"))
    args = parser.parse_args(argv)

    try:
        # 1. Find clean images. Ground-truth boxes are never inputs to the detector.
        images = find_images(args.input)
        for path in images:
            output_path = args.output / f"{path.stem}_detected.jpg"
            if output_path.resolve() in {p.resolve() for p in images}:
                raise ValueError("Choose an output folder that will not overwrite input images.")
        if not args.weights.is_file():
            raise ValueError("Model weights are missing. Follow the download step in README.md.")

        # 2. Load the pretrained network and tell it what objects to look for.
        from ultralytics import YOLOWorld

        model = YOLOWorld(str(args.weights))
        model.set_classes(PACKAGE_WORDS)
        args.output.mkdir(parents=True, exist_ok=True)
        records = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read image: {path}")

            # 3. Predict boxes. The library handles resizing and CNN inference.
            # Confidence filters weak predictions. NMS removes duplicate boxes.
            result = model.predict(image, conf=args.confidence, imgsz=IMAGE_SIZE,
                                   iou=NMS_IOU, agnostic_nms=True, device="cpu", verbose=False)[0]

            # 4. Draw the remaining boxes and save the numerical predictions.
            output_path = args.output / f"{path.stem}_detected.jpg"
            boxes = save_detection(image, result, output_path)
            records.append({"image": path.name, "width": image.shape[1],
                            "height": image.shape[0], "detections": boxes})
            print(f"{path.name}: {len(boxes)} package(s) -> {output_path}")

        report = {"model": str(args.weights), "prompts": PACKAGE_WORDS,
                  "confidence_threshold": args.confidence, "image_size": IMAGE_SIZE,
                  "nms_iou": NMS_IOU, "images": records}
        report_path = args.output / "detections.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved predictions to {report_path}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
