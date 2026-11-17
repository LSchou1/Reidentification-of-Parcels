# Folder-based workflow for future photo batches

Run commands from `bachelor-detection` using its `.venv`. Start by editing
`workflow.yaml`; all its paths are relative to that configuration file.
The entry point is `workflow.py`. Each command is a separate step you can inspect.
No command starts a web server. The reports open directly as local HTML files.

## Annotation preference — agreed with the user

The user plans to ask the assistant to annotate each new batch, then correct
boxes in the review gallery when needed. Model predictions require review before
being used as reference annotations.
Prepare clean copies first, then annotate those copies, keeping the originals.
Annotate every visible instance of the classes that may be used later, create
empty `.txt` files only for reviewed negatives, and flag uncertain boundaries.
Keep overlay images separate from the clean model inputs. Review the difficult
cases before a final benchmark. Quick annotation still needs consistent rules.

The master YOLO annotation IDs in the annotated batch are always:

| ID | Object | Bounding box |
|---|---|---|
| 0 | `barcode` | Visible 1D bar pattern, also unreadable or partially visible |
| 1 | `label` | Whole physical shipping/address/tracking label |
| 2 | `parcel` | Visible extent of each physical parcel, including partial parcels |

These are separate, overlapping boxes. A parcel may contain several labels and
barcodes, or none. Use the same visibility policy throughout the batch. Test and
dimension stickers are not shipping labels under the current annotation policy.

## 1. Add photos and make clean copies

Put related photos into a folder named for their **physical parcel or recording**:

```text
data/raw/next_batch/
  physical_parcel_A/
    view_01.jpeg
    view_02.jpeg
  physical_parcel_B/
    view_01.jpeg
  recording_03/
    frame_0001.png
    frame_0002.png
```

Use the same physical-parcel ID if a parcel returns in another recording. Two
different parcels that look alike can belong to different groups. Grouping does
not use appearance or decoded barcode text. An unverified code is not a parcel ID.

```sh
.venv/bin/python workflow.py prepare
```

This creates independent clean copies in `data/annotated/next_batch/images/`,
matching coordinate files in `labels/`, and a `groups.csv` alongside them.
It creates **no annotations**. Ask the assistant to annotate the prepared batch.
For an already annotated export, copy its YOLO files to the matching labels paths;
class IDs must match the master table. Other formats need conversion first.

The first input subfolder supplies the default group ID. With a flat input
folder, fill the group column in the annotated batch's `groups.csv` before
splitting. For a photo showing multiple known parcels, use `parcel_A|parcel_B`. The splitter connects
those groups transitively. Pixel-identical copies are also kept together.

## 2. Split once, before inspecting model performance

```sh
.venv/bin/python workflow.py split
.venv/bin/python workflow.py status
```

The default is approximately **70% training, 15% validation, 15% test by group**,
with seed 42. Groups can have different numbers of photos, so the report shows the
actual image/object counts. The seed makes assignment reproducible. Check that
each split represents the conditions and classes you intend to evaluate.

At least three independent groups are needed; many more are preferable for useful
evaluation. Three groups only establish that the software can form three splits.
If all photos are connected through the same parcels, the code rejects the split.
It never breaks a physical group just to reach the requested percentages.

Images, annotations, configuration and group assignments are frozen by hashes.
Training requires examples of every active class. Missing annotations, leakage
and edits to a frozen view are rejected. Correct annotations **before** freezing;
use a new batch for later corrections or additional photos.

The split is made before baseline analysis so the future test set stays held back.
The sequence still measures pretrained inference before supervised training.

## 3. Measure the pretrained baseline

```sh
.venv/bin/python workflow.py baseline
```

The baseline uses the existing barcode model plus YOLO-World parcel/label prompts
for the enabled classes. It performs inference on **validation only**, saving
proposals down to confidence 0.001. Each class is searched in the whole image;
a missed parcel cannot prevent a barcode prediction.

By default, confidence is selected by the highest **combined F1 on validation**.
Every distinct stored score is considered. If the configured 0.35 ties for best
F1, it is retained; otherwise a point inside the best interval is used. Combined
F1 pools object counts; always read per-class results as well. To use one fixed
threshold, set `select_confidence: false` and edit `confidence`.

NMS IoU removes duplicate predictions within a target class. The evaluation IoU
matches model boxes with reference boxes. These two settings are different.
Overlapping parcel/label/barcode boxes are preserved across classes.

## 4. Train explicitly

```sh
.venv/bin/python workflow.py train --dry-run
.venv/bin/python workflow.py train
```

The dry run validates and prints the recipe without loading a model, writing a
training output or updating weights. The real command requires the completed
baseline. Training starts from `barcode_weights`; the detector head is adapted
to the active classes. A new parcel/label class must be learned from your data.

Only **train and validation** are supplied to the optimizer/model selection.
The test split is never supplied. The best checkpoint, training losses,
validation curves, settings, versions and data hashes are retained. Settings
such as epochs and batch size are in `workflow.yaml`.

## 5. Validate the fine-tuned model

```sh
.venv/bin/python workflow.py validate
```

This evaluates `best.pt` on validation and chooses its confidence using the same
selection rule as the baseline. Each model may receive a different threshold,
because their scores need not have the same calibration. Make model and threshold
decisions on validation. Validation scores are selection results and can be
optimistic; they are not the final claim about new operational data.

## 6. Final test and comparison

```sh
.venv/bin/python workflow.py test
```

Both validation selections must be complete and current before either test run.
The test uses each model's **frozen validation threshold**. It evaluates both
models on the exact same held-out images and creates a comparison report.
Repeating a completed command reuses matching reports; it does not silently
retune or overwrite them. Changed settings/weights require a new `run_name`.

To regenerate just the comparison from completed reports:

```sh
.venv/bin/python workflow.py compare
```

Do not use final test results to repeatedly choose training settings or confidence.
If you do inspect test results during development, record that fact and reserve a
new untouched test collection for the final claim.

Each inference/evaluation folder contains:

- `index.html`: image review, per-class metrics and explanatory notes.
- `evaluation.json`: counts, precision, recall, F1, AP/mAP, IoU and match details.
- `metrics.csv`: per-class and overall numbers for a spreadsheet/report.
- `detections.csv`: individual TP/FP/FN rows with scores, coordinates and matched IoU.
- `predictions.json`: all stored candidate boxes, exact models and settings.
- `confusion_matrix.png`, `iou_distribution.png`, `confidence_analysis.png`.
- `associations.json`: predicted barcode/label-to-parcel containment candidates.
  Multiple candidates remain ambiguous. Indices are local to one image and do
  not establish physical identity or tracking.

Validation also saves `validation_threshold_sweep.json`. The comparison provides
HTML, CSV and JSON, validation/test tables kept separate, each model's operating
threshold, per-class AP and counts, confusion matrices and a training-curve link.
**Accuracy is not calculated.** IoU summaries describe correct matches only.
Barcode payload correctness, OCR, tracking and re-identification need separate
ground truth and are not measured by these detection reports.

The baseline is a system of pretrained detectors; the fine-tuned model is a joint
detector. Their comparison measures the before/after systems. It does not isolate
fine-tuning as the only changed factor, especially for the new classes.

## Enable, disable or reorder classes

Edit just this list in `workflow.yaml`:

```yaml
classes: [barcode, label, parcel]  # All three
# classes: [barcode, label]       # Disable parcel
# classes: [barcode]              # Barcode only
# classes: [parcel]               # Parcel only
```

Then run `split` for that selection. It creates a filtered dataset view and remaps
class IDs automatically. Master boxes are preserved; **do not delete or renumber
the original annotation files**. Every selection reuses the common split plan,
including the same held-out groups. Each selection has its own results folder.
Removing a class can make a previously positive image a valid negative for the
remaining classes. Newly enabling a class still requires complete master
annotations for that class; a configuration entry cannot create missing boxes.

For another training recipe on the same data, change `run_name` to e.g.
`experiment_02`, then run baseline/train/validate/test. For a new batch, change
`input`, `annotations` and `output` to matching new batch folders and start with prepare.

```text
data/
  raw/next_batch/                 # Original photographs
  annotated/next_batch/
    images/                      # Clean working copies; no boxes drawn in pixels
    labels/                      # Matching YOLO .txt coordinates
    annotations.manual.json      # Final editable pixel coordinates
    annotations.coco.json        # COCO export
    groups.csv                   # Physical parcel/recording IDs
    data.yaml                    # Class names and image location
  output/
    next_batch/
      annotation_review/index.html
      split_plan.json
      datasets/barcode_label_parcel/   # Generated train/val/test copies
      experiments/experiment_01/barcode_label_parcel/
        baseline_val/
        training/detector/weights/best.pt
        finetuned_val/
        baseline_test/
        finetuned_test/
        comparison/index.html
    archive/                     # Earlier experiments and recovery files
```

## Where to read the code

- `workflow.py`: commands and stage order; start reading here.
- `workflow.yaml`: the settings you normally change.
- `src/workflow_data.py`: clean copies, group linking, splitting and class filtering.
- `src/workflow_models.py`: pretrained/joint inference and model provenance.
- `src/workflow_reports.py`: validation selection, test freezing, CSV and comparison.
- `src/train_barcodes.py` and `src/evaluate_barcodes.py`: the reused trainer/metrics.

The earlier 23-photo barcode/label collection remains development material. It
has not been turned into an independent three-class train/test benchmark, and its
current annotations do not include parcel boxes.
