"""Find barcodes with a trained YOLO detector, then decode original-resolution crops."""

import argparse
from datetime import datetime, timezone
import html
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import zxingcpp

from src.detect_packages import confidence_value, find_images, PACKAGE_WORDS
from src.setup_barcode_model import CONFIG, DEFAULT_WEIGHTS, ROOT, sha256


LABEL_WORDS = ["shipping label", "address label", "shipping sticker"]


def label_records(image, result, output, stem, confidence):
    """Keep whole-label detections separate from the barcodes they may contain."""
    labels = []
    for box in result.boxes:
        if float(box.conf[0]) < confidence:
            continue
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        crop, crop_xyxy = crop_barcode(image, xyxy, padding=0)
        name = f"labels/{stem}_{len(labels) + 1:02d}.png"
        write_image(output / name, crop)
        labels.append({"class": "label", "xyxy": xyxy, "confidence": float(box.conf[0]),
                       "crop_xyxy": crop_xyxy, "crop_file": name})
    return labels


def crop_barcode(image, xyxy, padding=0.20):
    """Keep margin for barcode quiet zones; never resize the source before cropping."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid barcode bounding box.")
    dx, dy = max(16, (x2 - x1) * padding), max(16, (y2 - y1) * padding)
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    right, bottom = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    if right <= left or bottom <= top:
        raise ValueError("Barcode bounding box is outside the image.")
    return image[top:bottom, left:right].copy(), [left, top, right, bottom]


def decoding_variants(crop):
    """Yield pixels and the transform back to the unmodified crop."""
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    yield "original", crop, identity
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield "grayscale", gray, identity
    yield "otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], identity
    # Interpolation can help sampling; it cannot restore missing bar detail.
    if max(crop.shape[:2]) < 1400:
        yield "upscaled_2x", cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), identity / 2
    # ZXing's right-angle rotation alone is insufficient for diagonally placed parcels.
    # Keep the entire crop and map decoder corners back through the inverse affine transform.
    height, width = gray.shape
    for angle in (-15, 15, -30, 30, -45, 45):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        rotated_width = math.ceil(height * sine + width * cosine)
        rotated_height = math.ceil(height * cosine + width * sine)
        matrix[0, 2] += (rotated_width - width) / 2
        matrix[1, 2] += (rotated_height - height) / 2
        rotated = cv2.warpAffine(gray, matrix, (rotated_width, rotated_height), borderValue=255)
        yield f"rotation_{angle:+d}", rotated, cv2.invertAffineTransform(matrix)


def decode_crop(crop, crop_xyxy, detection_xyxy):
    """Return real decoder results, including failures as an empty list."""
    codes = {}
    for method, pixels, inverse_transform in decoding_variants(crop):
        results = zxingcpp.read_barcodes(np.ascontiguousarray(pixels), try_rotate=True,
                                        try_downscale=True, try_invert=True)
        for result in results:
            if not result.valid:
                continue
            position = result.position
            points = [position.top_left, position.top_right,
                      position.bottom_right, position.bottom_left]
            local = np.array([[[p.x, p.y] for p in points]], dtype=np.float64)
            restored = cv2.transform(local, inverse_transform)[0]
            polygon = [[float(x + crop_xyxy[0]), float(y + crop_xyxy[1])] for x, y in restored]
            center_x = sum(p[0] for p in polygon) / 4
            center_y = sum(p[1] for p in polygon) / 4
            x1, y1, x2, y2 = detection_xyxy
            # A padded crop may contain a neighbouring code. Do not attach it to this box.
            if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                continue
            key = (result.format.name, result.bytes.hex())
            codes.setdefault(key, {"text": result.text, "format": result.format.name,
                                   "raw_bytes_hex": result.bytes.hex(),
                                   "polygon": polygon, "preprocessing": method})
        if codes:
            break
    return list(codes.values())


def package_candidates(xyxy, packages):
    center_x, center_y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
    return [i for i, package in enumerate(packages)
            if package["xyxy"][0] <= center_x <= package["xyxy"][2]
            and package["xyxy"][1] <= center_y <= package["xyxy"][3]]


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not save {path}")


def process_image(image, result, output, stem, packages=None, labels=None):
    """Separate ML confidence from decoding success; preserve original coordinates."""
    overlay = image.copy()
    drawing_scale = max(1, max(image.shape[:2]) / 1600)
    line_width = max(2, round(2 * drawing_scale))
    text_offset = round(10 * drawing_scale)
    text_height = round(28 * drawing_scale)
    packages = [] if packages is None else packages
    labels = [] if labels is None else labels
    for i, label in enumerate(labels):
        x1, y1, x2, y2 = map(round, label["xyxy"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 40, 160), line_width)
        cv2.putText(overlay, f"label {i + 1}: {label['confidence']:.2f}",
                    (x1, max(text_height, y1 - text_offset)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * drawing_scale, (180, 40, 160), line_width)
    for i, package in enumerate(packages):
        x1, y1, x2, y2 = [int(v) for v in package["xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 150, 20), line_width)
        cv2.putText(overlay, f"package {i + 1}", (x1, max(text_height, y1 - text_offset)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * drawing_scale, (220, 150, 20), line_width)
    detections = []
    for i, box in enumerate(result.boxes):
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        crop, crop_xyxy = crop_barcode(image, xyxy)
        codes = decode_crop(crop, crop_xyxy, xyxy)
        crop_name = f"crops/{stem}_{i + 1:02d}.png"
        write_image(output / crop_name, crop)
        candidates = package_candidates(xyxy, packages)
        label_candidates = package_candidates(xyxy, labels)
        record = {"xyxy": xyxy, "confidence": float(box.conf[0]),
                  "class": result.names[int(box.cls[0])], "crop_xyxy": crop_xyxy,
                  "crop_file": crop_name, "status": "decoded" if codes else "detected_not_decoded",
                  "codes": codes, "package_index": candidates[0] if len(candidates) == 1 else None,
                  "package_candidates": candidates,
                  "label_index": label_candidates[0] if len(label_candidates) == 1 else None,
                  "label_candidates": label_candidates}
        detections.append(record)
        color = (60, 180, 20) if codes else (0, 150, 255)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_width)
        # Text is in the JSON/gallery; use ASCII on the OpenCV overlay.
        label = f"barcode {i + 1}: {'decoded' if codes else 'unread'}"
        cv2.putText(overlay, label, (x1, max(text_height, y1 - text_offset)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * drawing_scale, color, line_width)
    preview = overlay
    if max(overlay.shape[:2]) > 1600:
        scale = 1600 / max(overlay.shape[:2])
        preview = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    preview_name = f"{stem}_barcodes.jpg"
    write_image(output / preview_name, preview)
    return detections, preview_name


def write_gallery(report, output):
    cards = []
    for record in report["images"]:
        rows = []
        for i, label in enumerate(record.get("labels", [])):
            crop = html.escape(label["crop_file"], quote=True)
            rows.append(f'<li><a href="{crop}">Fragtlabel {i + 1}</a> · ML-score {label["confidence"]:.3f}</li>')
        for i, detection in enumerate(record["detections"]):
            codes = "<br>".join(f"{html.escape(c['format'])}: <code>{html.escape(c['text'])}</code>"
                                 for c in detection["codes"]) or "Fundet, men ikke aflæst"
            crop = html.escape(detection["crop_file"], quote=True)
            rows.append(f'<li><a href="{crop}">Udsnit {i + 1}</a> · ML-score '
                        f'{detection["confidence"]:.3f}<br>{codes}</li>')
        name = html.escape(record["image"])
        preview = html.escape(record["preview"], quote=True)
        description = "<ul>" + "".join(rows) + "</ul>" if rows else "<p>Ingen stregkode eller fragtlabel fundet.</p>"
        cards.append(f'<article><h2>{name}</h2><a href="{preview}"><img loading="lazy" '
                     f'src="{preview}" alt="Stregkodedetektion på {name}"></a>{description}</article>')
    summary = report["summary"]
    page = ('<!doctype html><html lang="da"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Stregkoder på pakker</title><style>'
            'body{font:16px system-ui;background:#f4f5f7;color:#172433;margin:0;padding:28px}'
            'main{max-width:1250px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}'
            'article{background:white;border:1px solid #dce2e8;border-radius:12px;padding:18px}h2{font-size:18px}'
            'img{width:100%;max-height:500px;object-fit:contain}li{margin-bottom:12px}'
            'code{overflow-wrap:anywhere}a{color:#075baa}</style><main><h1>Stregkoder på pakker</h1>'
            f'<p>{summary["images"]} billeder · {summary["ml_detections"]} ML-fund · '
            f'{summary["decoded_detections"]} fund med aflæst indhold.</p>'
            '<p>Grøn: aflæst. Orange: fundet, men ikke aflæst. Blå: pakke, når pakkedetektion er valgt. '
            'Lilla: fragtlabel, når labeldetektion er valgt. '
            'ML-score er ikke en dokumenteret sandsynlighed. Brug evaluering mod gennemgåede annotationer til at måle kvaliteten.</p>'
            '<p><a href="barcodes.json">Numeriske resultater og modeloplysninger</a></p><div class="grid">'
            + "".join(cards) + '</div></main></html>')
    (output / "index.html").write_text(page, encoding="utf-8")


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/next_batch")
    parser.add_argument("--output", type=Path, default=ROOT / "data/output/barcodes")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=confidence_value, default=0.25)
    parser.add_argument("--imgsz", type=positive_integer, default=1280)
    parser.add_argument("--threads", type=positive_integer, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-labels", action="store_true", help="Also locate whole shipping labels.")
    parser.add_argument("--label-weights", type=Path, help="Optional trained detector containing the class label.")
    parser.add_argument("--label-confidence", type=confidence_value, default=0.1)
    parser.add_argument("--with-packages", action="store_true", help="Also run the existing package detector.")
    parser.add_argument("--package-weights", type=Path, default=ROOT / "models/yolov8s-worldv2.pt")
    parser.add_argument("--package-confidence", type=confidence_value, default=0.05)
    args = parser.parse_args(argv)
    try:
        paths = find_images(args.input)
        if any(p.resolve().is_relative_to(args.output.resolve()) for p in paths):
            raise ValueError("Input images must be outside the output directory. Choose another output folder.")
        if not args.weights.is_file():
            raise ValueError("Barcode weights are missing. Run: python -m src.setup_barcode_model")
        actual_hash = sha256(args.weights)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        if args.weights.resolve() == DEFAULT_WEIGHTS.resolve() and actual_hash != config["sha256"]:
            raise ValueError("Barcode model checksum mismatch. Restore the pinned weights before inference.")
        if args.with_packages and not args.package_weights.is_file():
            raise ValueError("Package weights are missing; supply --package-weights or omit --with-packages.")
        if args.label_weights and not args.with_labels:
            raise ValueError("Use --with-labels together with --label-weights.")
        import torch
        from ultralytics import YOLO, YOLOWorld

        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        model = YOLO(str(args.weights), task="detect")
        barcode_classes = [i for i, name in model.names.items() if name.lower().replace("_", "") == "barcode"]
        if not barcode_classes:
            raise ValueError(f"Weights have no barcode class: {model.names}")
        label_model = None
        label_classes = [i for i, name in model.names.items() if name == "label"]
        label_mode = None
        label_weights = None
        if args.with_labels:
            if args.label_weights:
                if not args.label_weights.is_file():
                    raise ValueError(f"Label model weights are missing: {args.label_weights}")
                label_weights = args.label_weights
                label_model = YOLO(str(label_weights), task="detect")
                label_classes = [i for i, name in label_model.names.items() if name == "label"]
                if not label_classes:
                    raise ValueError("The label model must contain the class label.")
                label_mode = "trained_detector"
            elif label_classes:
                label_model, label_weights, label_mode = model, args.weights, "joint_detector"
            else:
                label_weights = ROOT / "models/yolov8s-worldv2.pt"
                if not label_weights.is_file():
                    raise ValueError("Label baseline weights are missing. Supply --label-weights or install yolov8s-worldv2.pt.")
                label_model = YOLOWorld(str(label_weights))
                label_model.set_classes(LABEL_WORDS)
                label_classes = list(range(len(LABEL_WORDS)))
                label_mode = "open_vocabulary_baseline"
        package_model = None
        if args.with_packages:
            package_model = YOLOWorld(str(args.package_weights))
            package_model.set_classes(PACKAGE_WORDS)
        args.output.mkdir(parents=True, exist_ok=True)
        records = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read {path}")
            start = time.perf_counter()
            packages = []
            if package_model is not None:
                result = package_model.predict(image, conf=args.package_confidence, imgsz=960,
                                               iou=0.5, agnostic_nms=True, device=args.device, verbose=False)[0]
                packages = [{"xyxy": [float(v) for v in b.xyxy[0].tolist()],
                             "confidence": float(b.conf[0]), "class": "package"} for b in result.boxes]
            shared_model = label_model is model
            result = model.predict(image, conf=min(args.confidence, args.label_confidence) if shared_model else args.confidence,
                                   imgsz=args.imgsz, iou=0.45,
                                   classes=barcode_classes + label_classes if shared_model else barcode_classes,
                                   agnostic_nms=False, device=args.device, verbose=False)[0]
            labels = []
            if label_model is not None:
                if shared_model:
                    label_result = result[[int(b.cls[0]) in label_classes for b in result.boxes]]
                else:
                    label_result = label_model.predict(image, conf=args.label_confidence, imgsz=args.imgsz, iou=0.45,
                                                       classes=label_classes, agnostic_nms=True,
                                                       device=args.device, verbose=False)[0]
                labels = label_records(image, label_result, args.output, path.stem, args.label_confidence)
            result = result[[int(b.cls[0]) in barcode_classes and float(b.conf[0]) >= args.confidence for b in result.boxes]]
            detections, preview = process_image(image, result, args.output, path.stem, packages, labels)
            elapsed = time.perf_counter() - start
            records.append({"image": path.name, "sha256": sha256(path), "width": image.shape[1],
                            "height": image.shape[0], "preview": preview, "seconds": elapsed,
                            "packages": packages, "labels": labels, "detections": detections})
            decoded = sum(bool(d["codes"]) for d in detections)
            print(f"{path.name}: {len(detections)} barcode(s), {decoded} decoded", flush=True)
        summary = {"images": len(records), "ml_detections": sum(len(r["detections"]) for r in records),
                   "label_detections": sum(len(r["labels"]) for r in records),
                   "decoded_detections": sum(bool(d["codes"]) for r in records for d in r["detections"])}
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "model": {"file": args.weights.name, "sha256": actual_hash, "classes": model.names,
                            "source": config if actual_hash == config["sha256"] else None},
                  "settings": {"confidence": args.confidence, "imgsz": args.imgsz, "nms_iou": 0.45,
                               "device": args.device, "threads": args.threads, "crop_padding": 0.20,
                               "with_labels": args.with_labels,
                               "label_confidence": args.label_confidence if args.with_labels else None,
                               "decode_rotation_angles": [-15, 15, -30, 30, -45, 45],
                               "with_packages": args.with_packages,
                               "package_confidence": args.package_confidence if args.with_packages else None},
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("ultralytics", "torch", "zxing-cpp", "numpy")},
                  "summary": summary, "images": records}
        if label_model is not None:
            report["label_model"] = {"file": label_weights.name, "sha256": sha256(label_weights),
                                     "mode": label_mode,
                                     "prompts": LABEL_WORDS if label_mode == "open_vocabulary_baseline" else None}
        if args.with_packages:
            report["package_model"] = {"file": args.package_weights.name, "sha256": sha256(args.package_weights),
                                       "prompts": PACKAGE_WORDS, "imgsz": 960, "nms_iou": 0.5}
        (args.output / "barcodes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
        write_gallery(report, args.output)
        print(f"Saved {args.output / 'index.html'}", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
