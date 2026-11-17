"""Evaluate detections against reviewed boxes; confidence alone is not accuracy."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import cv2
import numpy as np

from src.barcode_dataset import check_dataset, dataset_summary
from src.detect_packages import confidence_value
from src.setup_barcode_model import DEFAULT_WEIGHTS, ROOT, sha256


def iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions, truth, threshold, same_class=True):
    """Greedily match each prediction once, in descending confidence order."""
    used, matches = set(), []
    for pi in sorted(range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True):
        candidates = [(iou(predictions[pi]["xyxy"], target["xyxy"]), gi)
                      for gi, target in enumerate(truth) if gi not in used
                      and (not same_class or predictions[pi]["class"] == target["class"])]
        if candidates:
            overlap, gi = max(candidates)
            if overlap >= threshold:
                used.add(gi)
                matches.append((pi, gi, overlap))
    return matches


def average_precision(flags, scores, total_truth):
    """101-point interpolated AP; classes without ground truth are undefined."""
    if total_truth == 0:
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = np.asarray(flags, dtype=float)[order]
    if not len(hits):
        return 0.0
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / total_truth
    return float(np.mean([max(precision[recall >= r], default=0.0)
                          for r in np.linspace(0, 1, 101)]))


def counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def summarize_iou(values):
    """Describe only correctly matched boxes; no matches means undefined IoU."""
    return {"matches": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None}


def score_at(samples, names, confidence, overlap):
    totals = {name: [0, 0, 0] for name in names}
    overlaps = {name: [] for name in names}
    matrix = np.zeros((len(names) + 1, len(names) + 1), dtype=int)
    details = []
    for sample in samples:
        predictions = [p for p in sample["predictions"] if p["confidence"] >= confidence]
        truth = sample["annotations"]
        matches = match_boxes(predictions, truth, overlap)
        matched_p = {p for p, _, _ in matches}
        matched_g = {g for _, g, _ in matches}
        for name in names:
            tp = sum(predictions[p]["class"] == name for p in matched_p)
            fp = sum(p["class"] == name for i, p in enumerate(predictions) if i not in matched_p)
            fn = sum(g["class"] == name for i, g in enumerate(truth) if i not in matched_g)
            totals[name] = [a + b for a, b in zip(totals[name], (tp, fp, fn))]
        # Preserve correct class matches, then show remaining class confusions explicitly.
        for pi, gi, _ in matches:
            matrix[names.index(truth[gi]["class"]), names.index(predictions[pi]["class"])] += 1
        for pi, _, value in matches:
            overlaps[predictions[pi]["class"]].append(value)
        remaining_p = [i for i in range(len(predictions)) if i not in matched_p]
        remaining_g = [i for i in range(len(truth)) if i not in matched_g]
        for pi, gi, _ in match_boxes([predictions[i] for i in remaining_p],
                                     [truth[i] for i in remaining_g], overlap, same_class=False):
            p, g = remaining_p[pi], remaining_g[gi]
            matrix[names.index(truth[g]["class"]), names.index(predictions[p]["class"])] += 1
            matched_p.add(p)
            matched_g.add(g)
        for pi, prediction in enumerate(predictions):
            if pi not in matched_p:
                matrix[-1, names.index(prediction["class"])] += 1
        for gi, target in enumerate(truth):
            if gi not in matched_g:
                matrix[names.index(target["class"]), -1] += 1
        correct_p = {p for p, _, _ in matches}
        correct_g = {g for _, g, _ in matches}
        details.append({"image": sample["image"], "predictions": predictions,
                        "annotations": truth, "matches": matches,
                        "match_details": [{"prediction_index": pi, "annotation_index": gi,
                                           "class": predictions[pi]["class"], "iou": value,
                                           "confidence": predictions[pi]["confidence"]}
                                          for pi, gi, value in matches],
                        "false_predictions": [i for i in range(len(predictions)) if i not in correct_p],
                        "missed_annotations": [i for i in range(len(truth)) if i not in correct_g]})
    return {"overall": counts(*np.sum(list(totals.values()), axis=0).tolist()),
            "per_class": {name: counts(*values) for name, values in totals.items()},
            "iou_summary": {"overall": summarize_iou([value for values in overlaps.values() for value in values]),
                            "per_class": {name: summarize_iou(values) for name, values in overlaps.items()}},
            "confusion_matrix": matrix.tolist(), "images": details}


def evaluate(samples, names, confidence=0.25, overlap=0.5, floor=0.0):
    if overlap <= 0 or overlap > 1 or not floor <= confidence <= 1:
        raise ValueError("IoU must be in (0,1]; evaluation confidence cannot be below the inference floor.")
    result = score_at(samples, names, confidence, overlap)
    ap = {}
    for name in names:
        values = []
        for threshold in np.linspace(0.5, 0.95, 10):
            flags, scores, total = [], [], 0
            for sample in samples:
                predictions = [p for p in sample["predictions"] if p["class"] == name]
                truth = [g for g in sample["annotations"] if g["class"] == name]
                matched = {pi for pi, _, _ in match_boxes(predictions, truth, threshold)}
                flags.extend(i in matched for i in range(len(predictions)))
                scores.extend(p["confidence"] for p in predictions)
                total += len(truth)
            values.append(average_precision(flags, scores, total))
        ap[name] = {"ap50": values[0], "ap50_95": float(np.mean(values)) if values[0] is not None else None}
    present = [v for v in ap.values() if v["ap50"] is not None]
    result["average_precision"] = ap
    result["map50"] = float(np.mean([v["ap50"] for v in present])) if present else None
    result["map50_95"] = float(np.mean([v["ap50_95"] for v in present])) if present else None
    thresholds = sorted({floor, confidence, *[v / 20 for v in range(1, 21) if v / 20 >= floor]})
    result["confidence_curve"] = [{"confidence": t, **score_at(samples, names, t, overlap)["overall"]}
                                   for t in thresholds]
    result["settings"] = {"confidence": confidence, "iou": overlap, "prediction_floor": floor,
                          "matrix_rows": "actual", "matrix_columns": "predicted",
                          "matrix_labels": names + ["background"],
                          "ap_method": "101 recall points; confidence-ordered one-to-one same-class matching"}
    result["notes"] = [
        "Confidence is a model score, not a calibrated probability or measured accuracy.",
        "Mean/median IoU describe correct same-class matches at the selected thresholds only; no matches gives null. Read them together with recall and the number of matches.",
        "Background/background is not a true-negative count. Object detection has no fixed number of background objects.",
        "AP uses all stored detections, not only those above the displayed operating threshold.",
        "Classes without reference objects have undefined AP and are excluded from mAP. Empty precision/recall denominators return 0.",
        "Decoder success is not verified payload accuracy; this report measures bounding-box detection.",
        "Use validation for threshold selection. Keep the test split untouched until the final comparison."]
    if floor > 0.001:
        result["notes"].append("AP and curves are limited by the stored prediction floor. Rerun inference at confidence 0.001 for a fuller comparison.")
    return result


def join_predictions(samples, report, names):
    by_name = {}
    for record in report["images"]:
        if record["image"] in by_name:
            raise ValueError("Prediction report contains duplicate image names.")
        by_name[record["image"]] = record
    joined = []
    for sample in samples:
        record = by_name.get(sample["image"], by_name.get(sample["path"].name))
        if record is None:
            raise ValueError(f"Missing predictions for {sample['image']}; missing output is not an empty detection.")
        if record.get("sha256") != sample["sha256"]:
            raise ValueError(f"Source image checksum differs for {sample['image']}. Rerun inference on the annotated source.")
        if (record.get("width"), record.get("height")) != (sample["width"], sample["height"]):
            raise ValueError(f"Image dimensions differ for {sample['image']}")
        predictions = record.get("detections", []) + record.get("labels", [])
        for prediction in predictions:
            if prediction["class"] not in names:
                raise ValueError(f"Prediction class {prediction['class']} is absent from the dataset names.")
            box = prediction["xyxy"]
            score = prediction["confidence"]
            if (len(box) != 4 or not np.isfinite(box).all() or not np.isfinite(score)
                    or not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError(f"Invalid prediction for {sample['image']}")
        joined.append({**sample, "predictions": predictions})
    return joined


def predict_samples(samples, weights, names, floor, imgsz, device):
    if not weights.is_file():
        raise ValueError(f"Model file is missing: {weights}")
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    selected = [i for i, name in model.names.items() if name in names]
    if not selected:
        raise ValueError(f"Model has no classes matching {names}: {model.names}")
    records = []
    discarded = 0
    for sample in samples:
        result = model.predict(str(sample["path"]), conf=floor, imgsz=imgsz, classes=selected,
                               agnostic_nms=False, device=device, verbose=False)[0]
        predictions = []
        for box in result.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            # At very low confidence, clipping a boundary proposal can leave zero area.
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                discarded += 1
                continue
            predictions.append({"class": model.names[int(box.cls[0])], "confidence": float(box.conf[0]), "xyxy": xyxy})
        records.append({k: sample[k] for k in ("image", "sha256", "width", "height")}
                       | {"detections": predictions})
    return {"model": {"file": str(weights), "sha256": sha256(weights), "classes": model.names},
            "settings": {"confidence": floor, "imgsz": imgsz, "device": device,
                         "discarded_degenerate_predictions": discarded}, "images": records}


def write_report(result, samples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(result["confusion_matrix"])
    labels = result["settings"]["matrix_labels"]
    figure, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Forudsagt klasse", ylabel="Faktisk klasse",
           title=f"Confusion matrix · conf {result['settings']['confidence']:g} · IoU {result['settings']['iou']:.2f}")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, "—" if row == col == len(labels) - 1 else str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    curve = result["confidence_curve"]
    for metric in ("precision", "recall", "f1"):
        axes[0].plot([p["confidence"] for p in curve], [p[metric] for p in curve], label=metric)
    axes[0].set(xlabel="Confidencegrænse", ylabel="Målt værdi", ylim=(0, 1.05))
    axes[0].legend()
    true_scores, false_scores = [], []
    for sample in samples:
        correct = {pi for pi, _, _ in match_boxes(sample["predictions"], sample["annotations"], result["settings"]["iou"])}
        for pi, prediction in enumerate(sample["predictions"]):
            (true_scores if pi in correct else false_scores).append(prediction["confidence"])
    axes[1].hist([true_scores, false_scores], bins=np.linspace(0, 1, 11), label=["Korrekte fund", "Falske fund"])
    axes[1].set(xlabel="Modellens confidence", ylabel="Antal fund")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "confidence_analysis.png", dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(7, 4))
    overlaps = [match["iou"] for detail in result["images"] for match in detail["match_details"]]
    if overlaps:
        ax.hist(overlaps, bins=np.linspace(0, 1, 21), color="#326a96", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "Ingen korrekte match: IoU er ikke defineret", ha="center", transform=ax.transAxes)
    ax.axvline(result["settings"]["iou"], color="#b14c33", linestyle="--", label="Valgt IoU-grænse")
    ax.set(xlabel="IoU for korrekte match", ylabel="Antal match", xlim=(0, 1), title="Overlap mellem modelboks og referenceboks")
    ax.legend()
    figure.tight_layout()
    figure.savefig(output / "iou_distribution.png", dpi=160)
    plt.close(figure)
    gallery = []
    for number, (sample, detail) in enumerate(zip(samples, result["images"])):
        image = cv2.imread(str(sample["path"]))
        for target in detail["annotations"]:
            x1, y1, x2, y2 = map(round, target["xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 150, 0), 3)
        false = set(detail["false_predictions"])
        for i, prediction in enumerate(detail["predictions"]):
            x1, y1, x2, y2 = map(round, prediction["xyxy"])
            color = (0, 0, 220) if i in false else (0, 160, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{prediction['class']} {prediction['confidence']:.3f}",
                        (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if max(image.shape[:2]) > 1600:
            image = cv2.resize(image, None, fx=1600 / max(image.shape[:2]), fy=1600 / max(image.shape[:2]))
        name = f"review_{number:04d}.jpg"
        if not cv2.imwrite(str(output / name), image):
            raise ValueError(f"Could not save review image: {name}")
        match_rows = ''.join(f'<tr><td>{match["prediction_index"] + 1}</td>'
                             f'<td>{match["annotation_index"] + 1}</td><td>{html.escape(match["class"])}</td>'
                             f'<td>{match["confidence"]:.3f}</td><td>{match["iou"]:.3f}</td></tr>'
                             for match in detail["match_details"])
        match_table = ('<table><tr><th>Fund</th><th>Reference</th><th>Klasse</th><th>Confidence</th><th>IoU</th></tr>'
                       + match_rows + '</table>') if match_rows else '<p>Ingen korrekte match ved de valgte grænser.</p>'
        gallery.append(f'<article><p>{html.escape(sample["image"])} · '
                       f'{len(detail["false_predictions"])} falske · {len(detail["missed_annotations"])} oversete</p>'
                       f'<a href="{name}"><img loading="lazy" src="{name}"></a>{match_table}</article>')
    rows = []
    for name, values in {"Samlet": result["overall"], **result["per_class"]}.items():
        overlap = result["iou_summary"]["overall"] if name == "Samlet" else result["iou_summary"]["per_class"][name]
        mean_iou = "—" if overlap["mean"] is None else f'{overlap["mean"]:.3f}'
        rows.append(f'<tr><td>{html.escape(name)}</td>' + ''.join(
            f'<td>{values[key]:.3f}</td>' if key in ("precision", "recall", "f1") else f'<td>{values[key]}</td>'
            for key in ("tp", "fp", "fn", "precision", "recall", "f1")) + f'<td>{mean_iou}</td></tr>')
    ap_text = "Ikke defineret" if result["map50"] is None else f"mAP50 {result['map50']:.3f} · mAP50–95 {result['map50_95']:.3f}"
    demonstration = ('<p><strong>Syntetisk funktionstest. Tallene dokumenterer ikke kvalitet på rigtige transportbåndsdata.</strong></p>'
                     if result.get("purpose") == "synthetic_functional_test_only" else '')
    page = ('<!doctype html><html lang="da"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Evaluering af stregkoder og labels</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#172433}'
            'table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}img{max-width:100%}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{border:1px solid #ddd;padding:12px}</style>'
            '<h1>Evaluering af stregkoder og labels</h1>' + demonstration
            + '<p>Confidence er modellens score. Kvaliteten nedenfor er målt mod gennemgåede annotationer.</p>'
            f'<p>Datasplit: {html.escape(result.get("split", "ukendt"))}. {ap_text}.</p>'
            f'<p>Confidencegrænse: {result["settings"]["confidence"]:.6g} · IoU-grænse: {result["settings"]["iou"]:.2f}.</p>'
            '<table><tr><th>Klasse</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>Gns. IoU (korrekte match)</th></tr>'
            + ''.join(rows) + '</table><p>IoU = overlapareal / samlet areal (intersection / union). '
            'Precision = TP / (TP + FP). Recall = TP / (TP + FN). '
            'Gennemsnitlig IoU gælder kun korrekt matchede bokse; uden match vises —. '
            'Læs derfor altid IoU sammen med recall og antallet af match.</p>'
            '<img src="confusion_matrix.png"><img src="iou_distribution.png"><img src="confidence_analysis.png">'
            '<p>Matrix: faktisk klasse på rækkerne, forudsagt klasse i kolonnerne. Background-kolonnen viser oversete objekter; '
            'background-rækken viser falske fund. Der beregnes ingen TN eller almindelig accuracy.</p><ul>'
            + ''.join(f'<li>{html.escape(note)}</li>' for note in result["notes"])
            + '</ul><p>Blå bokse: reference. Grøn: korrekt fund. Rød: forkert fund. En reference uden korrekt fund er overset.</p>'
            '<p><a href="evaluation.json">Alle tal og indstillinger</a></p><div class="grid">' + ''.join(gallery) + '</div></html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--predictions", type=Path, help="Existing barcodes.json; images are verified by checksum.")
    source.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data/output/next_batch/evaluation")
    parser.add_argument("--confidence", type=confidence_value, default=0.25)
    parser.add_argument("--iou", type=confidence_value, default=0.5)
    parser.add_argument("--prediction-floor", type=confidence_value, default=0.001)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        dataset = check_dataset(args.data, required=(args.split,))
        if args.imgsz < 32:
            raise ValueError("imgsz must be at least 32.")
        samples = dataset["splits"][args.split]
        if any(s["path"].resolve().is_relative_to(args.output.resolve()) for split in dataset["splits"].values() for s in split):
            raise ValueError("Evaluation output must not contain dataset images.")
        if args.predictions:
            report = json.loads(args.predictions.read_text(encoding="utf-8"))
        else:
            report = predict_samples(samples, args.weights or DEFAULT_WEIGHTS, dataset["names"],
                                     args.prediction_floor, args.imgsz, args.device)
        settings = report.get("settings", {})
        floor = max(settings.get("confidence", 0), settings.get("label_confidence", 0) or 0)
        joined = join_predictions(samples, report, dataset["names"])
        result = evaluate(joined, dataset["names"], args.confidence, args.iou, floor)
        result.update({"created_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
                       "dataset": dataset_summary(dataset), "model": report.get("model"),
                       "image_hashes": {s["image"]: s["sha256"] for s in samples}})
        result["notes"].append(f"Discarded {settings.get('discarded_degenerate_predictions', 0)} zero-area proposals after clipping at image boundaries.")
        if not dataset["groups_checked"]:
            result["notes"].append("No complete groups.csv was supplied. Exact duplicate pixels were checked, but related frames/parcels still require manual split review.")
        write_report(result, joined, args.output)
        (args.output / "predictions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result["overall"], indent=2))
        print(f"Saved {args.output / 'index.html'}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
