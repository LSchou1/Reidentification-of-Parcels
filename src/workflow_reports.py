"""Validation, frozen test evaluation, and a before/after model comparison."""

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from src.barcode_dataset import dataset_summary, label_path
from src.evaluate_barcodes import evaluate, join_predictions, score_at, write_report
from src.setup_barcode_model import sha256
from src.workflow_data import frozen_dataset, save_json
from src.workflow_models import model_files, predict_boxes


def choose_confidence(samples, names, floor, overlap, preferred=0.35):
    """Try every distinct prediction cutoff, using validation data only.

    Predictions do not change between two adjacent scores. Prefer a simple
    configured value when it ties for best F1; otherwise use a plateau midpoint.
    """
    thresholds = sorted({floor, 1.0, *[p["confidence"] for s in samples for p in s["predictions"]]})
    rows = [{"confidence": t, **score_at(samples, names, t, overlap)["overall"]} for t in thresholds]
    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["confidence"]))
    preferred_f1 = score_at(samples, names, preferred, overlap)["overall"]["f1"]
    if abs(preferred_f1 - best["f1"]) < 1e-12:
        return preferred, rows
    index = thresholds.index(best["confidence"])
    lower = thresholds[index - 1] if index else floor
    return (lower + best["confidence"]) / 2, rows


def analysis_request(config, kind, split, files, dataset_hash, validation_hash=None):
    """Enough information to reject stale results after a setting/model changes."""
    keys = ("imgsz", "prediction_floor", "nms_iou", "iou", "device", "confidence", "select_confidence")
    return {"kind": kind, "split": split, "classes": config["classes"],
            "dataset_manifest_sha256": dataset_hash, "weights": files,
            "settings": {key: config[key] for key in keys}, "validation_report_sha256": validation_hash,
            "purpose": config.get("purpose", "real_data_experiment")}


def export_analysis_tables(result, output):
    """CSV files are convenient for a thesis spreadsheet; JSON keeps all detail."""
    metric_rows, detection_rows, associations = [], [], []
    for name in ["overall", *result["dataset"]["names"]]:
        values = result["overall"] if name == "overall" else result["per_class"][name]
        overlap = result["iou_summary"]["overall"] if name == "overall" else result["iou_summary"]["per_class"][name]
        ap = {"ap50": result["map50"], "ap50_95": result["map50_95"]} if name == "overall" else result["average_precision"][name]
        metric_rows.append({"class": name, "confidence": result["settings"]["confidence"],
                            **values, "mean_iou": overlap["mean"], **ap})
    for image in result["images"]:
        matches = {match["prediction_index"]: match for match in image["match_details"]}
        predictions = image["predictions"]
        for index, prediction in enumerate(predictions):
            match = matches.get(index)
            detection_rows.append({"image": image["image"], "class": prediction["class"],
                "status": "TP" if match else "FP", "confidence": prediction["confidence"],
                "iou": match["iou"] if match else None,
                **dict(zip(("x1", "y1", "x2", "y2"), prediction["xyxy"]))})
            if prediction["class"] in {"barcode", "label"}:
                # Association uses predictions only, never the reference annotations.
                x1, y1, x2, y2 = prediction["xyxy"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                candidates = {}
                for name in ("parcel", "label"):
                    if name == prediction["class"]:
                        continue
                    found = [i for i, p in enumerate(predictions) if p["class"] == name
                             and p["xyxy"][0] <= cx <= p["xyxy"][2] and p["xyxy"][1] <= cy <= p["xyxy"][3]]
                    candidates[name + "_candidates"] = found
                    candidates[name + "_index"] = found[0] if len(found) == 1 else None
                associations.append({"image": image["image"], "prediction_index": index,
                                     "class": prediction["class"], **candidates})
        for index in image["missed_annotations"]:
            annotation = image["annotations"][index]
            detection_rows.append({"image": image["image"], "class": annotation["class"],
                "status": "FN", "confidence": None, "iou": None,
                **dict(zip(("x1", "y1", "x2", "y2"), annotation["xyxy"]))})
    tables = [("metrics.csv", metric_rows, list(metric_rows[0])),
              ("detections.csv", detection_rows, ["image", "class", "status", "confidence", "iou", "x1", "y1", "x2", "y2"])]
    for filename, rows, fields in tables:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    save_json(output / "associations.json", {"notes": "Geometric association at the selected confidence. Indices refer to evaluation.json image predictions. Multiple candidates remain ambiguous; this does not establish physical parcel identity.", "objects": associations})


def finish_report(result, output):
    """Mark completion only after the plots, tables and downloads are available."""
    export_analysis_tables(result, output)
    path = output / "index.html"
    page = path.read_text()
    if 'href="metrics.csv"' not in page:
        downloads = '<p><a href="metrics.csv">Metrics CSV</a> · <a href="detections.csv">Detections CSV</a> · <a href="predictions.json">Raw predictions</a> · <a href="associations.json">Geometric associations</a></p>'
        path.write_text(page.replace("</h1>", "</h1>" + downloads, 1), encoding="utf-8")
    save_json(output / "complete.json", {"evaluation_sha256": sha256(output / "evaluation.json")})


def run_analysis(config, run_root, kind, split, trained_weights=None):
    """Evaluate a model; choose on val, reuse that exact choice on test."""
    dataset, dataset_hash = frozen_dataset(config["output"], config["classes"])
    files = model_files(config, kind, trained_weights)
    validation_hash, threshold = None, config["confidence"]
    if split == "test":
        validation_path = Path(run_root) / f"{kind}_val/evaluation.json"
        if not validation_path.is_file():
            raise ValueError(f"Evaluate {kind} on validation before running the test set.")
        validation = json.loads(validation_path.read_text())
        expected = analysis_request(config, kind, "val", files, dataset_hash)
        if validation["workflow"]["request"] != expected:
            raise ValueError("Validation settings or model changed. Use a new run_name and validate again before testing.")
        validation_hash = sha256(validation_path)
        threshold = validation["settings"]["confidence"]
    request = analysis_request(config, kind, split, files, dataset_hash, validation_hash)
    output = Path(run_root) / f"{kind}_{split}"
    request_path, predictions_path = output / "request.json", output / "predictions.json"
    evaluation_path = output / "evaluation.json"
    if output.exists():
        if not request_path.is_file() or json.loads(request_path.read_text()) != request:
            raise ValueError(f"Existing results use different settings: {output}. Choose a new run_name.")
        if evaluation_path.is_file():
            completed = output / "complete.json"
            if completed.exists() and json.loads(completed.read_text())["evaluation_sha256"] != sha256(evaluation_path):
                raise ValueError("A completed evaluation was edited. Use a new run_name.")
            result = json.loads(evaluation_path.read_text())
            if not completed.exists():
                finish_report(result, output)
            print(f"Reusing completed report: {output / 'index.html'}")
            return result
    # Saving candidates separately allows a failed plotting step to be retried.
    if predictions_path.is_file():
        predictions = json.loads(predictions_path.read_text())
    else:
        predictions = predict_boxes(dataset["splits"][split], dataset["names"], config, kind, trained_weights)
        save_json(request_path, request)
        save_json(predictions_path, predictions)
    samples = join_predictions(dataset["splits"][split], predictions, dataset["names"])
    sweep = []
    if split == "val" and config["select_confidence"]:
        threshold, sweep = choose_confidence(samples, dataset["names"], config["prediction_floor"], config["iou"], config["confidence"])
    result = evaluate(samples, dataset["names"], threshold, config["iou"], config["prediction_floor"])
    result.update({"created_at": datetime.now(timezone.utc).isoformat(), "split": split,
                   "purpose": config.get("purpose", "real_data_experiment"),
                   "dataset": dataset_summary(dataset), "model": predictions["model"],
                   "image_hashes": {s["image"]: s["sha256"] for s in samples},
                   "annotation_hashes": {s["image"]: sha256(label_path(s["path"])) for s in samples},
                   "workflow": {"request": request, "dataset_manifest_sha256": dataset_hash,
                                "threshold_source": "validation" if split == "test" else "validation selection"}})
    result["notes"][0:0] = [
        "Active classes: " + ", ".join(dataset["names"]) + ". Results are shown separately for each class.",
        "Split groups use physical parcel/recording IDs and duplicate pixels, never visual similarity or unverified barcode content.",
        "Validation results are used for model/threshold selection and can be optimistic. Final performance is measured on the held-out test split.",
        ("Test confidence was frozen from this model's validation report; test annotations were not used to choose it."
         if split == "test" else "Confidence is selected by maximum combined F1 on validation." if config["select_confidence"] else "Confidence is fixed by workflow.yaml; no threshold search was performed."),
        "This analysis measures bounding-box detection, not barcode decoding, OCR, tracking, re-identification or verified parcel identity.",
        "CSV exports: metrics.csv and detections.csv (TP/FP/FN). associations.json links predicted barcode/label centers to predicted parcels; ambiguous matches remain unassigned.",
        f"Inference used NMS IoU {config['nms_iou']}, image size {config['imgsz']} and a limit of 300 detections per model per image.",
    ]
    for name in dataset["names"]:
        if not any(a["class"] == name for sample in samples for a in sample["annotations"]):
            result["notes"].append(f"No {name} reference objects in this split. Its AP is undefined; do not infer class quality from this split.")
    if predictions["settings"].get("cap_reached_images"):
        result["notes"].append("The detection cap was reached on some images. Saved proposals can limit recall; inspect predictions.json.")
    write_report(result, samples, output)
    if sweep:
        save_json(output / "validation_threshold_sweep.json", sweep)
    finish_report(result, output)
    print(f"Saved {output / 'index.html'}", flush=True)
    return result


def compatible_results(baseline, trained):
    """Never compare metrics from different images, annotations or matching rules."""
    for key in ("split", "image_hashes", "annotation_hashes"):
        if baseline[key] != trained[key]:
            raise ValueError(f"Cannot compare models with different {key}.")
    if baseline["dataset"]["names"] != trained["dataset"]["names"]:
        raise ValueError("Cannot compare different active classes.")
    if baseline["workflow"]["dataset_manifest_sha256"] != trained["workflow"]["dataset_manifest_sha256"]:
        raise ValueError("Cannot compare different frozen datasets.")
    for key in ("iou", "prediction_floor"):
        if baseline["settings"][key] != trained["settings"][key]:
            raise ValueError(f"Cannot compare different {key} settings.")
    for key in ("imgsz", "nms_iou"):
        if baseline["workflow"]["request"]["settings"][key] != trained["workflow"]["request"]["settings"][key]:
            raise ValueError(f"Cannot compare different {key} settings.")


def comparison_report(run_root):
    """Save per-class tables and exportable data, with test and val kept separate."""
    run_root = Path(run_root)
    results = {}
    for split in ("val", "test"):
        results[split] = {}
        for kind in ("baseline", "finetuned"):
            path = run_root / f"{kind}_{split}/evaluation.json"
            if not path.is_file():
                raise ValueError("Run baseline, train, validate and test before the final comparison.")
            completed = path.parent / "complete.json"
            if not completed.is_file() or json.loads(completed.read_text())["evaluation_sha256"] != sha256(path):
                raise ValueError("A report is incomplete or was edited. Finish the evaluation or use a new run_name.")
            results[split][kind] = json.loads(path.read_text())
        compatible_results(results[split]["baseline"], results[split]["finetuned"])
    for kind in ("baseline", "finetuned"):
        validation_path = run_root / f"{kind}_val/evaluation.json"
        validation, test = results["val"][kind], results["test"][kind]
        if test["workflow"]["request"]["validation_report_sha256"] != sha256(validation_path):
            raise ValueError("A validation report changed after testing. Use a new run_name.")
        if validation["model"] != test["model"] or validation["settings"]["confidence"] != test["settings"]["confidence"]:
            raise ValueError("Test must use the same model and confidence chosen on validation.")
    output = run_root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows, sections = [], []
    for split in ("test", "val"):
        pair = results[split]
        table_rows = []
        for name in ["overall", *pair["baseline"]["dataset"]["names"]]:
            for kind in ("baseline", "finetuned"):
                report = pair[kind]
                values = report["overall"] if name == "overall" else report["per_class"][name]
                overlaps = report["iou_summary"]["overall"] if name == "overall" else report["iou_summary"]["per_class"][name]
                ap = {"ap50": report["map50"], "ap50_95": report["map50_95"]} if name == "overall" else report["average_precision"][name]
                row = {"split": split, "model": kind, "class": name, "confidence": report["settings"]["confidence"],
                       **values, "mean_iou": overlaps["mean"], **ap}
                rows.append(row)
                cells = [html.escape(name), kind, f"{row['confidence']:.4g}", str(row['tp']), str(row['fp']), str(row['fn'])]
                cells += ["—" if row[key] is None else f"{row[key]:.3f}" for key in ("precision", "recall", "f1", "mean_iou", "ap50", "ap50_95")]
                table_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
        heading = "Final held-out test results" if split == "test" else "Validation results used for selection"
        sections.append(f"<h2>{heading}</h2><div class='scroll'><table><tr>" + "".join(
            f"<th>{label}</th>" for label in ("Class", "Model", "Conf", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean IoU", "AP50 / mAP50", "AP50–95 / mAP50–95")) + "</tr>" + "".join(table_rows) + "</table></div>" +
            f"<p><a href='../baseline_{split}/index.html'>Baseline details</a> · <a href='../finetuned_{split}/index.html'>Fine-tuned details</a></p>" +
            f"<div class='plots'><img alt='Baseline confusion matrix' src='../baseline_{split}/confusion_matrix.png'><img alt='Fine-tuned confusion matrix' src='../finetuned_{split}/confusion_matrix.png'></div>")
    page = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Pretrained vs fine-tuned detector</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#172433}table{border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.plots{display:flex;flex-wrap:wrap}.plots img{width:48%;min-width:280px;max-width:650px}img{max-width:100%}</style>
<h1>Pretrained vs fine-tuned detector</h1>
<p>The same clean images, annotations, active classes, image size, NMS and evaluation IoU were used for both models. Each model's confidence was chosen on validation and frozen for testing.</p>
<p>Read each class separately. Overall precision/recall/F1 pool object counts; mAP averages classes. Mean IoU describes correct matches only. No accuracy metric is calculated.</p>
""" + "".join(sections)
    if all(r.get("purpose") == "synthetic_functional_test_only" for pair in results.values() for r in pair.values()):
        page = page.replace("<h1>Pretrained vs fine-tuned detector</h1>", "<h1>Pretrained vs fine-tuned detector</h1><p><strong>Synthetic functional check only. These numbers do not establish performance on real parcels.</strong></p>")
    training_plot = run_root / "training/detector/results.png"
    if training_plot.is_file():
        page += "<h2>Training and validation curves</h2><img alt='Training curves' src='../training/detector/results.png'>"
    page += "<p><a href='comparison.csv'>Download table as CSV</a> · <a href='comparison.json'>All results as JSON</a></p></html>"
    (output / "index.html").write_text(page, encoding="utf-8")
    save_json(output / "comparison.json", {"reports": results, "rows": rows})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output / "index.html"
