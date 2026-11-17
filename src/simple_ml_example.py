"""A small class-based example. Run with: python -m src.simple_ml_example --help.

The ML implementation and evaluation stay in the existing workflow modules.
This file makes the main steps easy to follow, like the supplied BarcodeScanner.
"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ParcelDetector:
    def __init__(self, configFile=ROOT / "workflow.yaml"):
        from workflow import read_config

        # Tunable variabler ligger samlet i workflow.yaml.
        self.configFile = Path(configFile).resolve()
        self.settings = read_config(self.configFile)
        self.confidence = self.settings["confidence"]
        self.imageSize = self.settings["imgsz"]
        self.classes = self.settings["classes"]  # barcode, label og/eller parcel

    def loadImage(self, filepath):
        import cv2

        # Behold original størrelse: koordinater skal passe til originalbilledet.
        image = cv2.imread(str(filepath))
        if image is None:
            raise ValueError(f"Cannot read image: {filepath}")
        return image

    def findObjects(self, filepath, image):
        from src.setup_barcode_model import sha256
        from src.workflow_models import predict_boxes

        height, width = image.shape[:2]
        sample = {"path": Path(filepath), "image": Path(filepath).name,
                  "width": width, "height": height, "sha256": sha256(filepath)}

        # Pretrained inferens: modellen ændrer ingen vægte her.
        # Hjælperen bruger barcode-YOLO og YOLO-World til label/parcel.
        settings = dict(self.settings, imgsz=self.imageSize)
        predictions = predict_boxes([sample], self.classes, settings, "baseline")
        boxes = predictions["images"][0]["detections"]
        return [box for box in boxes if box["confidence"] >= self.confidence]

    def drawBoxes(self, image, boxes):
        import cv2

        result = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = [round(value) for value in box["xyxy"]]
            text = f"{box['class']} {box['confidence']:.2f}"
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(result, text, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return result

    def detectImage(self, filepath):
        import cv2

        image = self.loadImage(filepath)
        boxes = self.findObjects(filepath, image)
        result = self.drawBoxes(image, boxes)
        output = Path(self.settings["output"]) / "example" / f"{Path(filepath).stem}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result):
            raise ValueError(f"Cannot save image: {output}")
        print(f"{len(boxes)} detections. Preview: {output}")
        return boxes

    def runStep(self, step, dryRun=False):
        from workflow import main

        # Genbrug arbejdsgangen: samme grupper, split, metrics og kontrol af testdata.
        arguments = ["--config", str(self.configFile), step]
        if dryRun:
            arguments.append("--dry-run")
        return main(arguments)

    def splitData(self):
        # Kræver annotationer og fysiske pakke-ID'er i annotations.json først.
        return self.runStep("split")

    def runBaseline(self):
        # Inferens FØR egen træning; analyse på validation-data.
        return self.runStep("baseline")

    def trainModel(self, dryRun=False):
        # Kun train-data opdaterer vægte. Validation vælger checkpoint/early stop.
        return self.runStep("train", dryRun)

    def validateModel(self):
        # Vælg confidence på validation, før testdata åbnes.
        return self.runStep("validate")

    def testModels(self):
        # Begge systemer får samme testdata; gem også sammenligningen.
        return self.runStep("test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "workflow.yaml")
    commands = parser.add_subparsers(dest="step", required=True)
    commands.add_parser("image", help="Pretrained preview only; no training or evaluation.").add_argument("filepath", type=Path)
    for name in ("split", "baseline", "validate", "test"):
        commands.add_parser(name)
    commands.add_parser("train").add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    detector = ParcelDetector(args.config)
    if args.step == "image":
        detector.detectImage(args.filepath)
        return 0
    steps = {"split": detector.splitData, "baseline": detector.runBaseline,
             "train": lambda: detector.trainModel(args.dry_run),
             "validate": detector.validateModel, "test": detector.testModels}
    return steps[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
