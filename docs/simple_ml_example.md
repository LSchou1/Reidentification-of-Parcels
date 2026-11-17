# A small ML example for the bachelor project

`src/simple_ml_example.py` follows the structure of your `BarcodeScanner`:
one class, tunable settings, named steps, and comments explaining the choices.
It is an educational entry point to the existing workflow, rather than another
independent implementation of splitting, training or evaluation.

Your supplied scanner finds patterns using gradients, coherence and morphology.
The ML version finds objects using learned model weights. `loadImage`,
`findObjects`, `drawBoxes` and `detectImage` show that sequence explicitly.
`findObjects` calls the existing predictor in `src/workflow_models.py`; that is
where the models are loaded and their `predict()` methods are called. The
baseline uses barcode YOLO plus YOLO-World prompts for labels and parcels.
Inference uses pretrained weights and does not train on your photographs.

Run commands from the project folder, using the existing virtual environment:

```bash
.venv/bin/python -m src.simple_ml_example --help
.venv/bin/python -m src.simple_ml_example image data/raw/next_batch/your_photo.png
```

Replace `your_photo.png` with a real filename. This writes a preview to
`data/output/next_batch/example/`. It returns class names, confidence scores and
pixel coordinates `[x1, y1, x2, y2]`. It does not decode barcode contents or
calculate evaluation metrics. Preview confidence comes from `workflow.yaml`;
the formal evaluation can select a different threshold using validation data.
The helper currently loads the baseline models on each call, so this small
example is not a throughput benchmark.

You can also read the sequence directly in Python:

```python
from src.simple_ml_example import ParcelDetector

detector = ParcelDetector()
boxes = detector.detectImage("data/raw/next_batch/your_photo.png")
```

Change classes, model paths, confidence, network input size and training settings
in **one place: `workflow.yaml`**. Clean source images keep their original size.

After annotation and completion of `data/annotated/next_batch/groups.csv`, the
supervised experiment is still an explicit sequence:

```bash
.venv/bin/python -m src.simple_ml_example split
.venv/bin/python -m src.simple_ml_example baseline
.venv/bin/python -m src.simple_ml_example train --dry-run
.venv/bin/python -m src.simple_ml_example train
.venv/bin/python -m src.simple_ml_example validate
.venv/bin/python -m src.simple_ml_example test
```

Do not run this sequence until groups identify repeated views of the same
physical parcels. The existing splitter keeps connected groups together and
requires enough independent groups for train, validation and test. A random
frame split would give a misleading impression of performance on new parcels.

`baseline` evaluates pretrained inference on validation data. `train` updates
weights on training data and uses validation for checkpoint selection and early
stopping. `validate` evaluates the fine-tuned model and selects its confidence
threshold. `test` evaluates both frozen systems on the same held-out images and
saves a comparison. Each step delegates to `workflow.py`, keeping its existing
checks and reports. Train dry-run checks the recipe without loading a model;
it still needs the frozen split and starting weights.

The analysis includes precision, recall, F1, AP/mAP, IoU and a confusion matrix;
accuracy is omitted. A high confidence score does not prove a correct box.
Evaluation compares predictions with reference annotations. For the report,
describe the baseline as a combination of pretrained detectors and the trained
version as one joint detector: their comparison changes both training and the
system composition. No performance claim follows from having this example.
