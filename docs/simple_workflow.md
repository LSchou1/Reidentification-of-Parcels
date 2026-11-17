# Simple workflow: photos → annotations → inference → training → comparison

Run commands from `bachelor-detection` with its `.venv`. Settings are in
`workflow.yaml`; paths are relative to that file. Reports open as local HTML.

Select the batch folder names and object classes in that file:

```yaml
batches: [next_batch]             # Or [next_batch, batch_02]
input: data/raw/{batch}
annotations: data/annotated/{batch}
output: data/output/{batch}
temporary: tmp/{batch}
classes: [barcode, label, parcel]
```

Only change `batches` to select data; keep `{batch}` in the paths. All workflow
commands and annotation-gallery export run each selected batch separately, in
the listed order. Each batch keeps its own annotations, fixed split, reports and
trained model. This selection does not pool batches into one model or calculate
combined metrics. If a batch fails, the command stops before later batches.
Every selected batch needs reviewed annotations for every active class when
running analysis or training. Both current batches support all three classes.
The 23 `real_photos` images now contain 31 parcel boxes, 27 barcode boxes and
10 shipping-label boxes. Their shared conservative group must still be resolved
into enough independent groups before splitting for training/validation/test.

## Three data folders

```text
data/
  raw/next_batch/                  # Unchanged original photographs
  annotated/next_batch/
    photo_001.jpg                 # Reference boxes drawn on these pictures
    photo_002.jpg
    annotations.json              # The ONLY coordinate file you edit
  output/next_batch/
    index.html                    # Links to this batch's model experiments
    inference/index.html          # Whole-batch inference, metrics and pictures
    experiments/                  # One named folder per model experiment
    dataset_split.json            # Permanent split record, needed for reproducibility
tmp/next_batch/                   # Editor, thumbnails and generated training copies
```

Earlier experiments, retired projects and annotation history are kept locally
in `tmp/archive/`. They are not part of the active workflow.

**The current 100-photo batch is already prepared and annotated.** Its final
boxes contain 485 parcels, 123 barcodes and 31 whole shipping labels. Keep those
boxes; the remaining step before splitting is confirming physical group IDs.

## 1. Add a NEW batch and prepare it

Put original photos in `data/raw/<batch-name>/`. Subfolders can identify physical
parcels or recordings. Set `batches: [<batch-name>]` in `workflow.yaml`, then run:

```sh
.venv/bin/python workflow.py prepare
```

This creates one `annotations.json`, keyed by paths relative to the raw folder.
When preparing several new batches, list only those new batches in `batches`.
Different parcel subfolders may therefore both contain `view_01.jpg`. The
annotation gallery renders boxed previews into the annotated folder, using
subfolder prefixes in preview filenames to avoid collisions.
The preparation command refuses to overwrite an existing annotated batch.
Technical preparation records go in `tmp/<batch>/`.

Ask the assistant to annotate the batch. Model suggestions require review before
being used as reference boxes. The user prefers assistant-led annotation, with
manual corrections in the gallery when necessary.

## 2. Understand the ONE coordinate file

Each filename has a record like this; boxes use original-image pixels:

```json
{
  "photo_001.png": {
    "width": 2448,
    "height": 2048,
    "group": "physical_parcel_A",
    "reviewed": true,
    "barcode": [[100, 200, 280, 300]],
    "label": [[70, 120, 500, 650]],
    "parcel": [[20, 50, 1500, 1800]]
  }
}
```

Every box is `[x1, y1, x2, y2]`: top-left and bottom-right corners. An empty class
list means no visible objects of that class **after review**. `reviewed: false`
means unfinished annotation and is rejected for training. A missing class field
also means unannotated, not a negative example. When adding a class to an
existing batch, review its images before filling that class's boxes or negatives.

Parcel, shipping-label and barcode boxes can overlap. Annotate every visible
instance, including unreadable barcodes. A shipping label is the whole physical
label; a barcode box covers the visible bar pattern. Keep one visibility policy
throughout the batch. `excluded` regions and review notes are optional metadata.

The photographs in `raw/` stay clean. Pictures in `annotated/` show the reference
boxes. Training reads the raw photographs and the JSON coordinates separately;
it never uses the pictures with drawn boxes. JSON keys name the raw files,
while the matching reference preview uses the same stem with a `.jpg` suffix.

## 3. Keep related photographs together, then split

You can inspect the annotated batch **before** filling groups or splitting:

```sh
.venv/bin/python workflow.py inspect --confidence 0.05
```

Open `data/output/next_batch/inference/index.html`. The same HTML contains
predictions for every active class, per-class precision/recall/F1 and AP,
IoU, confusion matrix, confidence curves and pictures showing correct, false
and missed detections. CSV and JSON downloads are linked from that page.
The low 0.05 threshold keeps weak proposals visible for this inspection;
change it in the command to inspect another value. Saved predictions are reused.
Without `--confidence`, the command uses the configured confidence.

This uses clean raw photos and the final coordinate JSON. It performs no
training or automatic threshold selection and does not modify the annotations,
split or future validation settings. Its whole-batch results are exploratory;
use the separate validation/test workflow below for the model comparison.
Re-running inspection replaces that inspection report; named experiments remain
separate. This command reads `workflow.yaml`, unlike the lower-level
`src.detect_barcodes` command with separate flags.

Fill each photo's `group` in `annotations.json`. Every view of the same physical
parcel must stay in the same group, including views from different recordings.
For a frame showing two parcels, use `parcel_A|parcel_B`; the code connects both
groups transitively. Similar-looking parcels do not automatically share identity.
Byte/pixel-identical copies are also kept together.

```sh
.venv/bin/python workflow.py split
.venv/bin/python workflow.py status
```

The default ratios are 70/15/15 by independent group, with seed 42. Actual image
percentages may differ. At least three independent groups are required, and more
are preferable. If reused parcels connect the whole collection, collect new,
independent parcels for validation/test; do not split neighbouring frames apart.

The code converts JSON boxes to YOLO `.txt` files automatically under `tmp/<batch>/`.
It freezes image hashes, coordinates, group assignments, seed and class-specific
views. Changing classes reuses the same group assignment and remaps class IDs.
Do not edit a frozen view; use a new batch/output for later data corrections.
Every active class needs examples in the training split.

## 4. Measure pretrained inference BEFORE supervised training

```sh
.venv/bin/python workflow.py baseline
```

The baseline searches whole images with the pretrained barcode detector and
YOLO-World label/parcel prompts. It evaluates validation data only. Saving
proposals down to confidence 0.001 supports confidence/AP curves; this is not the
recommended operating threshold. By default, validation selects combined F1's
best threshold; the fixed preferred confidence is 0.35 when it ties.

## 5. Train, validate and finally test

```sh
.venv/bin/python workflow.py train --dry-run
.venv/bin/python workflow.py train
.venv/bin/python workflow.py validate
.venv/bin/python workflow.py test
```

Dry-run checks the recipe without loading/training a model. Supervised training
starts from the local barcode weights and adapts a joint detector to the active
classes. Only training images update weights. Validation selects the checkpoint,
early stopping and confidence threshold. Test data is used after those choices
are frozen, with both systems evaluated on the same held-out images.

The default maximum is 100 epochs, batch 4, patience 20, CPU and input size 1280.
Change these in `workflow.yaml`. Change `run_name` for another experiment.
`classes: [barcode, label, parcel]` enables all classes; remove names to omit them.
Master JSON boxes remain intact when a class is disabled.

Reports contain per-class precision, recall, F1, AP/mAP, IoU, confusion matrices,
confidence curves, CSV/JSON and visual errors. Accuracy is omitted. Mean IoU
covers matched detections; read false negatives/recall alongside it. Confidence
is a model score, not proof that a detection is correct.

`test` generates comparison HTML/CSV/JSON; `workflow.py compare` regenerates it
from completed reports. The baseline combines pretrained detectors, whereas the
fine-tuned model is one joint detector. Describe the comparison accordingly;
it does not isolate training as the only changed factor.

## 6. Review annotations and open results

```sh
.venv/bin/python -m src.export_annotation_gallery
```

Open `tmp/next_batch/annotation_review/index.html` to select, move, resize,
add or delete boxes. This disposable editor can be regenerated at any time.
Open `data/output/next_batch/index.html` for completed model experiments.
Export `annotations.json`, then ask the assistant to import it into the batch.
Browser drafts do not modify the saved coordinate file automatically.

`tmp/<batch>/thumbnails/` contains smaller gallery previews. The gallery falls
back to full-size previews if the thumbnail cache is deleted.
`data/annotated/<batch>/` contains the full-size pictures with reference boxes.
`tmp/` contains generated YOLO datasets and scratch files. It can be deleted.
The permanent `output/<batch>/dataset_split.json` preserves the exact assignment;
the next workflow run rebuilds the same training copies from photos and JSON.
Raw photos, reference pictures, coordinate JSON, checkpoints and reports remain
outside `tmp/`.

For a small class-based code example, see `src/simple_ml_example.py` and
`docs/simple_ml_example.md`. The complete sequence stays in `workflow.py`;
`src/workflow_data.py` handles JSON, groups and splitting, and
`src/workflow_reports.py` handles evaluation and model comparison.
