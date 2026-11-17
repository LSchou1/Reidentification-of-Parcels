# Bachelor detection

Detect parcels, whole shipping labels and barcodes. Measure a pretrained baseline,
then train on your annotated data and compare the results.

## Three data folders

```text
data/
  raw/                         Original photographs, organised by batch
  annotated/                   Pictures with reference boxes and coordinate JSON
  output/                      Galleries, checkpoints, fixed split record and analysis
tmp/                           Disposable caches, scratch work and retired copies
```

The current 100-image batch is `next_batch`. The earlier 23-photo collection is
`real_photos`. Earlier reports, retired projects and annotation history are in `tmp/archive/`.

For the current batch:

- Originals: `data/raw/next_batch/`
- Pictures with reference boxes: directly in `data/annotated/next_batch/`
- The only coordinate file: `data/annotated/next_batch/annotations.json`
- Annotation gallery: `tmp/next_batch/annotation_review/index.html`
- Generated training datasets: `tmp/next_batch/datasets/`
- Training and analysis results: `data/output/next_batch/experiments/`

Each annotated batch contains only photos and one `annotations.json` file.
The pictures here show your reference boxes; the JSON holds their pixel coordinates
and physical group IDs. Training always reads clean images from `raw/` and boxes
from the JSON. It never reads the pictures with drawn boxes.
The code generates YOLO `.txt` files under `tmp/<batch>/` when splitting.

## Run the workflow

Run commands from this `bachelor-detection` folder with its existing `.venv`.
Change paths, active classes and training settings in [`workflow.yaml`](workflow.yaml).
See [the workflow guide](docs/simple_workflow.md) for grouping and evaluation.

```sh
# Show the current paths and dataset status.
.venv/bin/python workflow.py status

# NEW batches only: copy raw photos, then ask the assistant to annotate them.
.venv/bin/python workflow.py prepare

# After annotation: fill physical group IDs in annotations.json before splitting.
.venv/bin/python workflow.py split
.venv/bin/python workflow.py baseline
.venv/bin/python workflow.py train --dry-run
.venv/bin/python workflow.py train
.venv/bin/python workflow.py validate
.venv/bin/python workflow.py test
```

**The current batch is already prepared and annotated.** Its final annotations
contain 485 parcels, 123 barcodes and 31 shipping labels. Start with grouping,
then `split`; do not run `prepare` again for this batch.

Keep every view of the same physical parcel in one group, including appearances
in different recordings. Use `parcel_A|parcel_B` when a frame shows both. At least
three independent groups are needed for train/validation/test. If these frames
all connect through reused parcels, collect independent parcels for validation
and testing instead of splitting related frames across those sets.

`classes: [barcode, label, parcel]` enables all three classes. Removing a class
filters a generated dataset without deleting its master annotations. Validation
selects checkpoints and confidence thresholds. Test data stays held back until
those choices are frozen. Reports include IoU, confusion matrices, precision,
recall, F1, AP/mAP, CSV/JSON and a before/after comparison. No accuracy is calculated.

Open `data/output/next_batch/index.html` for model experiment links.
The annotation editor is in `tmp/next_batch/annotation_review/index.html`.
`tmp/` contains disposable generated files. The fixed split record stays with
the permanent results so training copies can be rebuilt after deleting `tmp/`.

## Review annotations

Open `tmp/next_batch/annotation_review/index.html` in a browser. Select,
move, resize, add or delete boxes, then export the JSON and ask the assistant to
import it. Browser drafts do not update the YOLO files automatically.
Regenerate the current gallery and coordinate exports with:

```sh
.venv/bin/python -m src.export_annotation_gallery
```

## Simple bachelor example

Start with [`src/simple_ml_example.py`](src/simple_ml_example.py) and its
[short guide](docs/simple_ml_example.md). It shows one readable class for loading
an image, finding objects and drawing boxes, plus explicit training/evaluation steps.

## Code

- `workflow.py`: commands in workflow order.
- `workflow.yaml`: the settings you normally change.
- `src/workflow_data.py`: JSON coordinates, grouping and reproducible splits.
- `src/workflow_models.py`: model inference.
- `src/workflow_reports.py`: analysis and model comparison.

Optional old notebooks and Docker setup have been moved to `tmp/optional/`.
They are not required for this local project and can be removed.

Lower-level commands are documented in [training_evaluation.md](docs/training_evaluation.md).

## Environment and checks

The local environment uses Python 3.12. For a fresh macOS environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Model files are in `models/`; `workflow.yaml` names the weights to use. The
workflow checks local weights and does not download them automatically.

```sh
.venv/bin/python -m pytest -q
```

Photos, annotations, experiment outputs and recovery archives are ignored by Git.
Keep a separate backup of your data.
